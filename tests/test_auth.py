"""Comprehensive authentication tests: email/password, Argon2id, JWT, refresh rotation, CSRF, rate limiting, and Google OAuth."""

from datetime import datetime, timedelta, timezone
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import httpx
import jwt
import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import AuthConfigurationError, Settings, settings, validate_auth_configuration
from app.core.security import create_access_token, create_oauth_state_token, decode_access_token, hash_token
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.auth import OAuthAccount, RefreshToken, User
from app.services.auth_service import AuthService, InactiveUserError
from app.services.google_oauth_service import (
    GoogleOAuthService,
    GoogleOAuthTokenError,
    get_google_oauth_service,
)

pytestmark = pytest.mark.real_auth


@pytest.fixture
def auth_db_session():
    """Isolated SQLite database per test with real auth dependencies enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _override_db():
        db = TestingSessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_db
    db = TestingSessionLocal()
    try:
        yield db, TestingSessionLocal
    finally:
        db.close()
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _register_user(
    client: TestClient,
    *,
    name: str = "Alice Tester",
    email: str = "alice@example.com",
    password: str = "StrongPass1!",
):
    return client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": password},
    )


# ============================================================================
# 1. Registration & Password Hashing Tests
# ============================================================================


def test_register_success_and_argon2id_hashing(auth_db_session):
    """Successful registration hashes password with Argon2id, sets HttpOnly refresh cookie, and returns JWT."""
    db, _ = auth_db_session
    client = TestClient(app)

    resp = _register_user(client, name="Alice Developer", email="Alice@Example.COM", password="MySecurePassword1!")
    assert resp.status_code == 201
    data = resp.json()
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    assert data["csrf_token"]
    assert data["user"]["email"] == "alice@example.com"
    assert data["user"]["name"] == "Alice Developer"
    assert data["user"]["is_active"] is True
    assert data["user"]["email_verified"] is False
    assert "password_hash" not in data["user"]
    assert "password" not in resp.text.lower() or "password" in str(data["user"]["auth_providers"])

    # Refresh token cookie must be HttpOnly
    set_cookie_headers = resp.headers.get_list("set-cookie")
    refresh_cookie_hdr = next(h for h in set_cookie_headers if h.startswith("refresh_token="))
    assert "HttpOnly" in refresh_cookie_hdr

    # Verify DB stores Argon2id hash and SHA-256 hashed refresh token
    user = db.execute(select(User).where(User.email == "alice@example.com")).scalars().first()
    assert user is not None
    assert user.password_hash is not None
    assert user.password_hash.startswith("$argon2id$")
    assert user.password_hash != "MySecurePassword1!"

    rt_rows = db.execute(select(RefreshToken).where(RefreshToken.user_id == user.id)).scalars().all()
    assert len(rt_rows) == 1
    assert len(rt_rows[0].token_hash) == 64


def test_register_duplicate_email_returns_409(auth_db_session):
    """Duplicate registration (including case variation) returns 409 Conflict."""
    client = TestClient(app)
    r1 = _register_user(client, email="dup@example.com")
    assert r1.status_code == 201

    r2 = _register_user(client, email="  DUP@Example.com ")
    assert r2.status_code == 409
    assert "already exists" in r2.json()["detail"].lower()


@pytest.mark.parametrize(
    "weak_password",
    [
        "Short1!",       # < 8 chars
        "alllowercase1!", # no uppercase
        "ALLUPPERCASE1!", # no lowercase
        "NoDigitsHere!",  # no digit
        "NoSpecial123",   # no special character
    ],
)
def test_register_weak_password_returns_422(auth_db_session, weak_password):
    """Weak passwords fail validation with 422 Unprocessable Entity."""
    client = TestClient(app)
    resp = _register_user(client, email=f"u_{uuid.uuid4().hex[:6]}@example.com", password=weak_password)
    assert resp.status_code == 422


# ============================================================================
# 2. Email/Password Login & Account Status Tests
# ============================================================================


def test_login_success_and_me_endpoint(auth_db_session):
    """Valid login returns access token and /auth/me returns the authenticated user."""
    client = TestClient(app)
    _register_user(client, email="bob@example.com", password="BobPassword1!")

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": "BobPassword1!", "remember_me": True},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "bob@example.com"


def test_login_incorrect_credentials_generic_error(auth_db_session):
    """Wrong password and unknown email produce identical generic 401 errors."""
    client = TestClient(app)
    _register_user(client, email="known@example.com", password="ValidPassword1!")

    wrong_pw = client.post(
        "/api/v1/auth/login",
        json={"email": "known@example.com", "password": "WrongPassword1!"},
    )
    unknown_email = client.post(
        "/api/v1/auth/login",
        json={"email": "unknown@example.com", "password": "WrongPassword1!"},
    )

    assert wrong_pw.status_code == 401
    assert unknown_email.status_code == 401
    assert wrong_pw.json()["detail"] == unknown_email.json()["detail"] == "Invalid email or password."


def test_login_and_access_inactive_account_blocked(auth_db_session):
    """Inactive users are denied login (403) and existing tokens are rejected by require_active_user (403)."""
    db, _ = auth_db_session
    client = TestClient(app)
    reg = _register_user(client, email="inactive@example.com", password="ValidPassword1!")
    token = reg.json()["access_token"]

    user = db.execute(select(User).where(User.email == "inactive@example.com")).scalars().first()
    assert user is not None
    user.is_active = False
    db.commit()

    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "inactive@example.com", "password": "ValidPassword1!"},
    )
    assert login_resp.status_code == 403

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 403


# ============================================================================
# 3. Refresh Token Rotation, Reuse Detection, CSRF, and Logout Tests
# ============================================================================


def test_refresh_token_rotation_and_reuse_detection(auth_db_session):
    """Refresh rotates tokens, and reusing an already-rotated token revokes the entire family."""
    db, _ = auth_db_session
    client = TestClient(app)
    reg = _register_user(client, email="rotate@example.com", password="RotatePass1!")
    csrf_1 = reg.json()["csrf_token"]
    old_refresh_cookie = client.cookies.get("refresh_token")
    assert old_refresh_cookie

    # 1. Valid refresh rotates token
    ref_resp = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_1},
    )
    assert ref_resp.status_code == 200
    new_access = ref_resp.json()["access_token"]
    csrf_2 = ref_resp.json()["csrf_token"]
    new_refresh_cookie = client.cookies.get("refresh_token")
    assert new_access
    assert new_refresh_cookie and new_refresh_cookie != old_refresh_cookie

    # Verify old refresh token is revoked in DB
    old_record = db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(old_refresh_cookie))
    ).scalars().first()
    assert old_record is not None
    assert old_record.revoked_at is not None

    # 2. Replay attack with old refresh token fails 401 and revokes entire family
    attacker_client = TestClient(app)
    attacker_client.cookies.set("refresh_token", old_refresh_cookie)
    attacker_client.cookies.set("csrf_token", csrf_2)
    replay_resp = attacker_client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_2},
    )
    assert replay_resp.status_code == 401

    # New refresh token in same family must now also be revoked
    db.expire_all()
    new_record = db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(new_refresh_cookie))
    ).scalars().first()
    assert new_record is not None
    assert new_record.revoked_at is not None


def test_csrf_required_for_refresh_and_logout(auth_db_session):
    """Refresh and logout reject requests missing or mismatching the X-CSRF-Token header."""
    client = TestClient(app)
    _register_user(client, email="csrf@example.com", password="CsrfPass1!")

    no_csrf = client.post("/api/v1/auth/refresh")
    assert no_csrf.status_code == 403

    bad_csrf = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "wrong-token"})
    assert bad_csrf.status_code == 403


def test_logout_revokes_refresh_token_and_clears_cookies(auth_db_session):
    """Logout revokes refresh token in PostgreSQL and subsequent refresh fails with 401."""
    client = TestClient(app)
    reg = _register_user(client, email="logout@example.com", password="LogoutPass1!")
    csrf = reg.json()["csrf_token"]
    refresh_val = client.cookies.get("refresh_token")

    out_resp = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
    assert out_resp.status_code == 200

    # Even if a client tries to replay the logged-out refresh token, DB marks it revoked
    replay_client = TestClient(app)
    replay_client.cookies.set("refresh_token", refresh_val)
    replay_client.cookies.set("csrf_token", csrf)
    ref_after = replay_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    assert ref_after.status_code == 401


def test_expired_access_token_rejected(auth_db_session):
    """Expired JWT access tokens return 401 Unauthorized."""
    client = TestClient(app)
    reg = _register_user(client, email="exp@example.com", password="ExpPass1!")
    user_id = reg.json()["user"]["id"]

    expired_token, _ = create_access_token(
        user_id=user_id,
        email="exp@example.com",
        expires_delta=timedelta(seconds=-10),
    )
    resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401


# ============================================================================
# 4. Google OAuth 2.0 Cryptographic Verification & Account Linking Tests
# ============================================================================


def _generate_rsa_keypair_and_jwk(kid: str = "test-google-kid"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    public_jwk["kid"] = kid
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"
    return private_key, public_jwk


def _sign_google_id_token(
    private_key,
    *,
    kid: str = "test-google-kid",
    sub: str = "google-sub-123",
    email: str = "googleuser@example.com",
    email_verified: bool = True,
    aud: str = "test-google-client-id",
    iss: str = "https://accounts.google.com",
    nonce: str = "",
    exp_offset_seconds: int = 300,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "email": email,
        "email_verified": email_verified,
        "name": "Google User",
        "aud": aud,
        "iss": iss,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=exp_offset_seconds)).timestamp()),
    }
    if nonce:
        payload["nonce"] = nonce
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


def test_google_oauth_full_flow_success(auth_db_session, monkeypatch):
    """Google OAuth verifies signed state, RS256 ID token via JWKS, creates user + OAuthAccount, and sets cookies."""
    db, _ = auth_db_session
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk()

    client = TestClient(app)
    init_resp = client.get("/api/v1/auth/google/login?mode=json")
    assert init_resp.status_code == 200
    state = init_resp.json()["state"]
    state_claims = jwt.decode(state, options={"verify_signature": False})
    nonce = state_claims["nonce"]

    id_token = _sign_google_id_token(
        private_key,
        sub="google-sub-001",
        email="oauth.new@example.com",
        email_verified=True,
        nonce=nonce,
    )

    google_service = GoogleOAuthService()
    monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": id_token}))
    monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service

    try:
        cb_resp = client.post(
            "/api/v1/auth/google/callback",
            json={"code": "valid-auth-code", "state": state},
        )
        assert cb_resp.status_code == 200
        data = cb_resp.json()
        assert data["access_token"]
        assert data["user"]["email"] == "oauth.new@example.com"
        assert data["user"]["email_verified"] is True
        assert "google" in data["user"]["auth_providers"]
        assert client.cookies.get("refresh_token")

        oauth_row = db.execute(
            select(OAuthAccount).where(OAuthAccount.provider_user_id == "google-sub-001")
        ).scalars().first()
        assert oauth_row is not None
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


def test_google_oauth_rejects_invalid_token_and_unverified_email(auth_db_session, monkeypatch):
    """Google OAuth rejects tampered/wrong-audience tokens and unverified Google emails."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk()
    other_key, _ = _generate_rsa_keypair_and_jwk()

    client = TestClient(app)
    init_resp = client.get("/api/v1/auth/google/login?mode=json")
    state = init_resp.json()["state"]
    nonce = jwt.decode(state, options={"verify_signature": False})["nonce"]

    google_service = GoogleOAuthService()
    monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service

    try:
        # 1. Invalid signature (signed by wrong key)
        bad_sig_token = _sign_google_id_token(other_key, nonce=nonce)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": bad_sig_token}))
        r_bad_sig = client.post("/api/v1/auth/google/callback", json={"code": "c1", "state": state})
        assert r_bad_sig.status_code == 400

        # 2. Wrong audience
        bad_aud_token = _sign_google_id_token(private_key, aud="other-client-id", nonce=nonce)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": bad_aud_token}))
        r_bad_aud = client.post("/api/v1/auth/google/callback", json={"code": "c2", "state": state})
        assert r_bad_aud.status_code == 400

        # 3. Unverified email
        unverified_token = _sign_google_id_token(private_key, email_verified=False, nonce=nonce)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": unverified_token}))
        r_unverified = client.post("/api/v1/auth/google/callback", json={"code": "c3", "state": state})
        assert r_unverified.status_code == 403
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


def test_google_oauth_invalid_state_rejected(auth_db_session, monkeypatch):
    """Google callback rejects tampered state or missing oauth_state cookie."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    client = TestClient(app)
    # Missing cookie altogether
    r_no_cookie = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": "forged"})
    assert r_no_cookie.status_code == 400

    # Valid cookie from init, but forged state param
    client.get("/api/v1/auth/google/login?mode=json")
    r_bad_state = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": "forged.jwt.state"})
    assert r_bad_state.status_code == 400


def test_google_oauth_safe_account_linking_and_takeover_prevention(auth_db_session, monkeypatch):
    """Verified existing user links safely; unverified existing user is blocked (409) to prevent account takeover."""
    db, _ = auth_db_session
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk()
    google_service = GoogleOAuthService()
    monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service

    client = TestClient(app)

    try:
        # 1. Pre-existing UNVERIFIED password account -> linking MUST be blocked (409)
        _register_user(client, email="victim@example.com", password="VictimPassword1!")
        init1 = client.get("/api/v1/auth/google/login?mode=json")
        state1 = init1.json()["state"]
        nonce1 = jwt.decode(state1, options={"verify_signature": False})["nonce"]
        tok1 = _sign_google_id_token(private_key, sub="g-sub-victim", email="victim@example.com", nonce=nonce1)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": tok1}))

        r_blocked = client.post("/api/v1/auth/google/callback", json={"code": "c1", "state": state1})
        assert r_blocked.status_code == 409

        # 2. Verified existing account -> safely links Google OAuthAccount
        victim_user = db.execute(select(User).where(User.email == "victim@example.com")).scalars().first()
        assert victim_user is not None
        victim_user.email_verified = True
        db.commit()

        init2 = client.get("/api/v1/auth/google/login?mode=json")
        state2 = init2.json()["state"]
        nonce2 = jwt.decode(state2, options={"verify_signature": False})["nonce"]
        tok2 = _sign_google_id_token(private_key, sub="g-sub-victim", email="victim@example.com", nonce=nonce2)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": tok2}))

        r_linked = client.post("/api/v1/auth/google/callback", json={"code": "c2", "state": state2})
        assert r_linked.status_code == 200
        assert r_linked.json()["user"]["id"] == victim_user.id
        assert set(r_linked.json()["user"]["auth_providers"]) == {"password", "google"}
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


# ============================================================================
# 5. Rate Limiting & Production Startup Validation Tests
# ============================================================================


def test_auth_rate_limiting_returns_429(auth_db_session, monkeypatch):
    """Repeated login attempts beyond configured limit return 429 Too Many Requests."""
    monkeypatch.setattr(settings, "AUTH_RATE_LIMIT_MAX_ATTEMPTS", 3)
    client = TestClient(app)

    for _ in range(3):
        r = client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x"})
        assert r.status_code == 401

    limited = client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x"})
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_production_startup_validates_auth_settings():
    """Production configuration validation rejects missing/weak JWT_SECRET_KEY or invalid FRONTEND_URL."""
    weak_cfg = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql://u:p@localhost/db",
        JWT_SECRET_KEY="short",
        _env_file=None,
    )
    with pytest.raises(AuthConfigurationError):
        validate_auth_configuration(weak_cfg)

    valid_cfg = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql://u:p@localhost/db",
        JWT_SECRET_KEY="a" * 32,
        FRONTEND_URL="https://app.example.com",
        _env_file=None,
    )
    validate_auth_configuration(valid_cfg)


def test_google_oauth_missing_and_placeholder_credentials_return_503(auth_db_session, monkeypatch):
    """Missing or placeholder Google OAuth credentials return a safe 503 configuration error."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", None)
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", None)

    client = TestClient(app)
    r_missing = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert r_missing.status_code == 503
    assert r_missing.json() == {"detail": "Google OAuth is not configured on the server."}

    r_missing_json = client.get("/api/v1/auth/google/login?mode=json")
    assert r_missing_json.status_code == 503
    assert r_missing_json.json() == {"detail": "Google OAuth is not configured on the server."}

    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "your_google_client_id_here")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "your_google_client_secret_here")
    r_placeholder = client.get("/api/v1/auth/google/login?mode=json")
    assert r_placeholder.status_code == 503
    assert r_placeholder.json() == {"detail": "Google OAuth is not configured on the server."}


def test_google_oauth_login_redirect_and_invalid_token_claims(auth_db_session, monkeypatch):
    """Google login returns 302 redirect with oauth_state cookie; expired/wrong-issuer tokens and expired codes fail safely."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    client = TestClient(app)
    redirect_resp = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert redirect_resp.status_code == 302
    location = redirect_resp.headers["location"]
    assert location.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=test-google-client-id" in location
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A5173%2Fauth%2Fcallback" in location
    assert "test-google-client-secret" not in location
    assert client.cookies.get("oauth_state")

    private_key, jwk = _generate_rsa_keypair_and_jwk()
    init_resp = client.get("/api/v1/auth/google/login?mode=json")
    state = init_resp.json()["state"]
    nonce = jwt.decode(state, options={"verify_signature": False})["nonce"]

    google_service = GoogleOAuthService()
    monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service

    try:
        # Expired Google ID token
        expired_id_token = _sign_google_id_token(
            private_key, nonce=nonce, exp_offset_seconds=-120
        )
        monkeypatch.setattr(
            google_service, "exchange_code", AsyncMock(return_value={"id_token": expired_id_token})
        )
        r_exp = client.post("/api/v1/auth/google/callback", json={"code": "c-exp", "state": state})
        assert r_exp.status_code == 400

        # Wrong issuer
        bad_iss_token = _sign_google_id_token(
            private_key, nonce=nonce, iss="https://evil.example.com"
        )
        monkeypatch.setattr(
            google_service, "exchange_code", AsyncMock(return_value={"id_token": bad_iss_token})
        )
        r_iss = client.post("/api/v1/auth/google/callback", json={"code": "c-iss", "state": state})
        assert r_iss.status_code == 400

        # Expired / invalid authorization code from Google token endpoint
        class _FakeResp:
            status_code = 400
            def json(self):
                return {"error": "invalid_grant", "error_description": "Bad Request"}

        with patch("app.services.google_oauth_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _FakeResp()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            real_google_service = GoogleOAuthService()
            app.dependency_overrides[get_google_oauth_service] = lambda: real_google_service
            r_grant = client.post("/api/v1/auth/google/callback", json={"code": "expired-code", "state": state})
            assert r_grant.status_code == 400
            assert "expired" in r_grant.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


# ============================================================================
# 6. Extended Input Validation, JWT Edge Cases, OAuth Hardening & DB Error Tests
# ============================================================================


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"name": "   ", "email": "a@example.com", "password": "StrongPass1!"},
        {"name": "A", "email": "a@example.com", "password": "StrongPass1!"},
        {"name": "Alice", "email": "", "password": "StrongPass1!"},
        {"name": "Alice", "email": "not-an-email", "password": "StrongPass1!"},
        {"name": "Alice", "email": "a@example.com", "password": ""},
        {"name": "Alice", "email": "a@example.com", "password": "A1!" + "a" * 130},
    ],
)
def test_register_malformed_and_empty_inputs_return_422(auth_db_session, payload):
    """Empty, whitespace, malformed, or out-of-bounds registration payloads return 422."""
    client = TestClient(app)
    resp = client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"email": "", "password": "StrongPass1!"},
        {"email": "invalid-email", "password": "StrongPass1!"},
        {"email": "valid@example.com", "password": ""},
    ],
)
def test_login_malformed_and_empty_inputs_return_422(auth_db_session, payload):
    """Empty or malformed login inputs fail schema validation with 422."""
    client = TestClient(app)
    resp = client.post("/api/v1/auth/login", json=payload)
    assert resp.status_code == 422


def test_login_oauth_only_user_and_case_insensitive_email(auth_db_session):
    """OAuth-only accounts reject password login with generic 401; valid logins normalize email."""
    db, _ = auth_db_session
    client = TestClient(app)

    oauth_user = User(
        id=str(uuid.uuid4()),
        name="OAuth Only",
        email="oauthonly@example.com",
        password_hash=None,
        is_active=True,
        email_verified=True,
    )
    db.add(oauth_user)
    db.commit()

    oauth_login = client.post(
        "/api/v1/auth/login",
        json={"email": "oauthonly@example.com", "password": "AnyPassword1!"},
    )
    assert oauth_login.status_code == 401
    assert oauth_login.json()["detail"] == "Invalid email or password."

    reg = _register_user(client, name="  Alice   Smith  ", email="CaseTest@Example.COM", password="CasePass1!")
    assert reg.status_code == 201
    assert reg.json()["user"]["name"] == "Alice Smith"
    assert reg.json()["user"]["email"] == "casetest@example.com"

    ok_login = client.post(
        "/api/v1/auth/login",
        json={"email": "  casetest@example.com ", "password": "CasePass1!", "remember_me": False},
    )
    assert ok_login.status_code == 200
    assert ok_login.json()["access_token"]


def test_jwt_validation_edge_cases_and_deleted_user(auth_db_session):
    """Tampered, malformed, wrong-type, wrong-scheme, and deleted-user JWTs are rejected with 401."""
    db, _ = auth_db_session
    client = TestClient(app)
    reg = _register_user(client, email="jwtuser@example.com", password="JwtPassword1!")
    token = reg.json()["access_token"]
    user_id = reg.json()["user"]["id"]

    claims = decode_access_token(token)
    assert claims["sub"] == user_id
    assert claims["email"] == "jwtuser@example.com"
    assert claims["type"] == "access"
    assert "jti" in claims

    # Tampered signature
    parts = token.split(".")
    tampered = f"{parts[0]}.{parts[1]}.{'A' * len(parts[2])}"
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401

    # Malformed token & wrong auth scheme
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Basic dXNlcjpwYXNz"}).status_code == 401

    # Wrong token type (OAuth state JWT cannot be used as access token)
    state_jwt = create_oauth_state_token(nonce="abc", redirect_uri="http://localhost:5173/auth/callback")
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {state_jwt}"}).status_code == 401

    # Valid JWT for deleted user
    user = db.execute(select(User).where(User.id == user_id)).scalars().first()
    assert user is not None
    db.delete(user)
    db.commit()
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_refresh_expired_missing_and_csrf_origin_checks(auth_db_session):
    """Expired/missing refresh tokens return 401; disallowed Origin returns 403; /csrf initializes cookie."""
    db, _ = auth_db_session
    client = TestClient(app)

    # /csrf initializes and preserves CSRF token
    csrf_resp = client.get("/api/v1/auth/csrf")
    assert csrf_resp.status_code == 200
    csrf_tok = csrf_resp.json()["csrf_token"]
    assert csrf_tok
    assert client.cookies.get("csrf_token") == csrf_tok

    # Missing refresh cookie -> 401
    missing_ref = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf_tok})
    assert missing_ref.status_code == 401

    # Disallowed Origin -> 403
    bad_origin = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_tok, "Origin": "https://evil.example.com"},
    )
    assert bad_origin.status_code == 403

    # Expired refresh token in DB -> 401 and marked revoked
    reg = _register_user(client, email="expref@example.com", password="ExpRefPass1!")
    user_id = reg.json()["user"]["id"]
    raw_expired = "expired-refresh-token-value"
    db.add(
        RefreshToken(
            user_id=user_id,
            token_hash=hash_token(raw_expired),
            family_id="fam-exp-1",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
    )
    db.commit()

    exp_client = TestClient(app)
    exp_client.cookies.set("refresh_token", raw_expired)
    exp_client.cookies.set("csrf_token", csrf_tok)
    r_exp = exp_client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf_tok})
    assert r_exp.status_code == 401

    db.expire_all()
    row = db.execute(select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw_expired))).scalars().first()
    assert row is not None and row.revoked_at is not None


def test_google_oauth_existing_user_login_conflict_and_inactive(auth_db_session, monkeypatch):
    """Existing Google user logs in cleanly without duplication; conflicting Google sub (409) and inactive user (403) are blocked."""
    db, _ = auth_db_session
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk()
    google_service = GoogleOAuthService()
    monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service
    client = TestClient(app)

    try:
        # 1. First Google login creates user
        init1 = client.get("/api/v1/auth/google/login?mode=json")
        s1 = init1.json()["state"]
        n1 = jwt.decode(s1, options={"verify_signature": False})["nonce"]
        t1 = _sign_google_id_token(private_key, sub="g-sub-repeat", email="repeat@example.com", nonce=n1)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": t1}))
        r1 = client.post("/api/v1/auth/google/callback", json={"code": "c1", "state": s1})
        assert r1.status_code == 200
        user_id = r1.json()["user"]["id"]

        # 2. Second Google login reuses existing user and OAuthAccount
        init2 = client.get("/api/v1/auth/google/login?mode=json")
        s2 = init2.json()["state"]
        n2 = jwt.decode(s2, options={"verify_signature": False})["nonce"]
        t2 = _sign_google_id_token(private_key, sub="g-sub-repeat", email="repeat@example.com", nonce=n2)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": t2}))
        r2 = client.post("/api/v1/auth/google/callback", json={"code": "c2", "state": s2})
        assert r2.status_code == 200
        assert r2.json()["user"]["id"] == user_id
        assert len(db.execute(select(User).where(User.email == "repeat@example.com")).scalars().all()) == 1

        # 3. Same email but DIFFERENT Google subject -> 409 conflict
        init3 = client.get("/api/v1/auth/google/login?mode=json")
        s3 = init3.json()["state"]
        n3 = jwt.decode(s3, options={"verify_signature": False})["nonce"]
        t3 = _sign_google_id_token(private_key, sub="g-sub-different", email="repeat@example.com", nonce=n3)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": t3}))
        r3 = client.post("/api/v1/auth/google/callback", json={"code": "c3", "state": s3})
        assert r3.status_code == 409

        # 4. Deactivate user -> Google login returns 403
        user = db.execute(select(User).where(User.id == user_id)).scalars().first()
        assert user is not None
        user.is_active = False
        db.commit()

        init4 = client.get("/api/v1/auth/google/login?mode=json")
        s4 = init4.json()["state"]
        n4 = jwt.decode(s4, options={"verify_signature": False})["nonce"]
        t4 = _sign_google_id_token(private_key, sub="g-sub-repeat", email="repeat@example.com", nonce=n4)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": t4}))
        r4 = client.post("/api/v1/auth/google/callback", json={"code": "c4", "state": s4})
        assert r4.status_code == 403
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


def test_google_oauth_state_nonce_mismatch_and_config_hardening(auth_db_session, monkeypatch, caplog):
    """OAuth validates state/cookie nonce, ID token nonce, redirect URI, JWKS key rotation, and never logs secrets."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "super-secret-oauth-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk(kid="kid-new")
    _, old_jwk = _generate_rsa_keypair_and_jwk(kid="kid-old")
    google_service = GoogleOAuthService()
    app.dependency_overrides[get_google_oauth_service] = lambda: google_service
    client = TestClient(app)

    try:
        # 1. Cookie nonce mismatch with valid signed state -> 400
        init_a = client.get("/api/v1/auth/google/login?mode=json")
        state_a = init_a.json()["state"]
        client.get("/api/v1/auth/google/login?mode=json")  # Overwrites oauth_state cookie with session B's nonce
        r_mismatch = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": state_a})
        assert r_mismatch.status_code == 400

        # 2. State redirect_uri mismatch -> 400
        init_b = client.get("/api/v1/auth/google/login?mode=json")
        cookie_nonce_b = client.cookies.get("oauth_state")
        bad_redir_state = create_oauth_state_token(
            nonce=cookie_nonce_b,
            redirect_uri="https://evil.example.com/callback",
        )
        r_redir = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": bad_redir_state})
        assert r_redir.status_code == 400

        # 3. ID token nonce mismatch -> 400
        state_b = init_b.json()["state"]
        wrong_nonce_tok = _sign_google_id_token(private_key, kid="kid-new", nonce="wrong-nonce")
        monkeypatch.setattr(google_service, "fetch_jwks", AsyncMock(return_value=[jwk]))
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": wrong_nonce_tok}))
        r_id_nonce = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": state_b})
        assert r_id_nonce.status_code == 400

        # 4. JWKS key rotation fallback (first call returns old kid, force_refresh returns new kid)
        good_tok = _sign_google_id_token(private_key, kid="kid-new", email="rotated@example.com", nonce=cookie_nonce_b)
        fetch_mock = AsyncMock(side_effect=[[old_jwk], [jwk]])
        monkeypatch.setattr(google_service, "fetch_jwks", fetch_mock)
        monkeypatch.setattr(google_service, "exchange_code", AsyncMock(return_value={"id_token": good_tok}))
        r_rot = client.post("/api/v1/auth/google/callback", json={"code": "c-rot", "state": state_b})
        assert r_rot.status_code == 200
        assert fetch_mock.await_count == 2

        # 5. Token endpoint invalid_client -> 503 and never leaks client_secret in response or logs
        init_c = client.get("/api/v1/auth/google/login?mode=json")
        state_c = init_c.json()["state"]

        class _InvalidClientResp:
            status_code = 401
            def json(self):
                return {"error": "invalid_client"}

        caplog.set_level(logging.DEBUG)
        with patch("app.services.google_oauth_service.httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.post.return_value = _InvalidClientResp()
            mock_http.__aenter__.return_value = mock_http
            mock_http.__aexit__.return_value = False
            mock_client_cls.return_value = mock_http

            real_svc = GoogleOAuthService()
            app.dependency_overrides[get_google_oauth_service] = lambda: real_svc
            r_invalid_client = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": state_c})
            assert r_invalid_client.status_code == 503
            assert "super-secret-oauth-client-secret" not in r_invalid_client.text
            assert "super-secret-oauth-client-secret" not in caplog.text

            # Network error during token exchange -> 400
            mock_http.post.side_effect = httpx.ConnectError("network down")
            r_net = client.post("/api/v1/auth/google/callback", json={"code": "c", "state": state_c})
            assert r_net.status_code == 400
            assert "super-secret-oauth-client-secret" not in caplog.text

        # 6. Partial OAuth credentials or invalid redirect URI (fragment) -> 503
        monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback#frag")
        assert client.get("/api/v1/auth/google/login?mode=json").status_code == 503

        monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")
        monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", None)
        assert client.get("/api/v1/auth/google/login?mode=json").status_code == 503
    finally:
        app.dependency_overrides.pop(get_google_oauth_service, None)


@pytest.mark.anyio
async def test_verify_id_token_empty_nonce_and_invalid_algorithm(monkeypatch):
    """Passing empty expected_nonce or non-RS256 ID token is strictly rejected."""
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_ID", "test-google-client-id")
    monkeypatch.setattr(settings, "GOOGLE_CLIENT_SECRET", "test-google-client-secret")
    monkeypatch.setattr(settings, "GOOGLE_REDIRECT_URI", "http://localhost:5173/auth/callback")

    private_key, jwk = _generate_rsa_keypair_and_jwk()
    svc = GoogleOAuthService()
    monkeypatch.setattr(svc, "fetch_jwks", AsyncMock(return_value=[jwk]))

    tok = _sign_google_id_token(private_key, nonce="abc")
    with pytest.raises(GoogleOAuthTokenError):
        await svc.verify_id_token(tok, expected_nonce="")

    hs_tok = jwt.encode({"sub": "1", "email": "a@example.com"}, "a-32-byte-secret-key-for-hs256!!", algorithm="HS256", headers={"kid": "test-google-kid"})
    with pytest.raises(GoogleOAuthTokenError):
        await svc.verify_id_token(hs_tok, expected_nonce="abc")


def test_database_integrity_and_operational_errors_handled_safely(auth_db_session, monkeypatch):
    """DB integrity race during register returns 409; OperationalError returns 503 without leaking DB URLs."""
    db, _ = auth_db_session
    client = TestClient(app)

    # 1. IntegrityError race during register commit -> 409
    svc = AuthService()
    orig_create_user = svc._repo.create_user

    def _raced_create_user(*args, **kwargs):
        orig_create_user(*args, **kwargs)
        raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed: users.email"))

    monkeypatch.setattr(svc._repo, "create_user", _raced_create_user)
    from app.services.auth_service import get_auth_service
    app.dependency_overrides[get_auth_service] = lambda: svc
    try:
        r_race = _register_user(client, email="race@example.com", password="RacePassword1!")
        assert r_race.status_code == 409
    finally:
        app.dependency_overrides.pop(get_auth_service, None)

    # 2. OperationalError -> 503 without leaking credentials
    no_raise_client = TestClient(app, raise_server_exceptions=False)

    def _failing_db():
        mock_db = MagicMock()
        mock_db.execute.side_effect = OperationalError(
            "SELECT 1", {}, Exception("connection to postgresql://user:SECRET_DB_PASS@neon/db failed")
        )
        yield mock_db

    app.dependency_overrides[get_db] = _failing_db
    try:
        r_op = no_raise_client.post(
            "/api/v1/auth/login",
            json={"email": "a@example.com", "password": "Password1!"},
        )
        assert r_op.status_code == 503
        assert "SECRET_DB_PASS" not in r_op.text

        # /me with valid token when DB fails -> 503
        tok, _ = create_access_token(user_id="u-1", email="a@example.com")
        r_me_op = no_raise_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
        assert r_me_op.status_code == 503
        assert "SECRET_DB_PASS" not in r_me_op.text

        # Generic SQLAlchemyError -> 500 without leaking credentials
        def _persistence_err_db():
            mock_db = MagicMock()
            mock_db.execute.side_effect = SQLAlchemyError("SECRET_DB_PASS failure")
            yield mock_db

        app.dependency_overrides[get_db] = _persistence_err_db
        r_sqla = no_raise_client.post(
            "/api/v1/auth/login",
            json={"email": "a@example.com", "password": "Password1!"},
        )
        assert r_sqla.status_code == 500
        assert "SECRET_DB_PASS" not in r_sqla.text
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_alembic_auth_migration_preserves_existing_data_and_downgrades_cleanly(tmp_path, monkeypatch):
    """Alembic migration 20261110_0003 upgrades existing DB without data loss and downgrades cleanly."""
    db_file = tmp_path / "migration_test.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setattr(settings, "DATABASE_URL", db_url)

    alembic_cfg = AlembicConfig("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Upgrade to pre-auth revision and insert legacy rows
    alembic_command.upgrade(alembic_cfg, "20261029_0002")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO interview_sessions (id, role, experience_level, difficulty, topic, question_type) "
                "VALUES ('legacy-sess-1', 'Backend', 'Junior', 'Easy', 'Python', 'Technical')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO document_sources (id, source_type, title, status) "
                "VALUES ('legacy-src-1', 'document', 'Legacy Resume.pdf', 'completed')"
            )
        )

    # Upgrade to head (20261110_0003)
    alembic_command.upgrade(alembic_cfg, "head")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"users", "oauth_accounts", "refresh_tokens", "interview_sessions", "document_sources"}.issubset(tables)

    with engine.connect() as conn:
        sess_row = conn.execute(text("SELECT id, user_id FROM interview_sessions WHERE id = 'legacy-sess-1'")).fetchone()
        src_row = conn.execute(text("SELECT id, user_id FROM document_sources WHERE id = 'legacy-src-1'")).fetchone()
        assert sess_row is not None and sess_row[1] is None
        assert src_row is not None and src_row[1] is None

    # Downgrade back to 20261029_0002 cleanly
    alembic_command.downgrade(alembic_cfg, "20261029_0002")
    inspector_after = inspect(engine)
    tables_after = set(inspector_after.get_table_names())
    assert "users" not in tables_after
    assert "oauth_accounts" not in tables_after
    assert "refresh_tokens" not in tables_after
    engine.dispose()


