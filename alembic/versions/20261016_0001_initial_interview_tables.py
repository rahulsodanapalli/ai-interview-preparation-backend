"""initial interview tables

Revision ID: 20261016_0001
Revises: 
Create Date: 2026-10-16 11:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20261016_0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create interview_sessions table
    op.create_table(
        'interview_sessions',
        sa.Column('id', sa.String(length=36), nullable=False, comment='Unique UUID session identifier'),
        sa.Column('role', sa.String(length=100), nullable=False, comment='Target job role for the interview'),
        sa.Column('experience_level', sa.String(length=50), nullable=False, comment='Candidate experience level (Fresher, Junior, Mid-Level, Senior)'),
        sa.Column('difficulty', sa.String(length=50), nullable=False, comment='Configured difficulty level (Easy, Medium, Hard)'),
        sa.Column('topic', sa.String(length=100), nullable=False, comment='Interview topic or domain'),
        sa.Column('question_type', sa.String(length=50), nullable=False, comment='Type of questions (Technical, Scenario-Based, Behavioral)'),
        sa.Column('total_questions', sa.Integer(), nullable=False, server_default='5', comment='Total questions configured for the session'),
        sa.Column('completed_questions', sa.Integer(), nullable=False, server_default='0', comment='Count of completed and evaluated questions'),
        sa.Column('current_question_number', sa.Integer(), nullable=False, server_default='0', comment='Sequence number of the current active question'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='in_progress', comment="Session lifecycle status ('in_progress', 'completed')"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp when session was created'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp of last update'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_interview_sessions_id'), 'interview_sessions', ['id'], unique=False)
    op.create_index(op.f('ix_interview_sessions_status'), 'interview_sessions', ['status'], unique=False)

    # 2. Create interview_questions table
    op.create_table(
        'interview_questions',
        sa.Column('id', sa.String(length=36), nullable=False, comment='Unique UUID question identifier'),
        sa.Column('session_id', sa.String(length=36), nullable=False, comment='Foreign key to parent interview session'),
        sa.Column('order', sa.Integer(), nullable=False, comment='1-indexed sequence order of the question within the session'),
        sa.Column('question', sa.Text(), nullable=False, comment='The generated interview question text'),
        sa.Column('topic', sa.String(length=100), nullable=False, comment='Topic of the question'),
        sa.Column('difficulty', sa.String(length=50), nullable=False, comment='Difficulty of the question'),
        sa.Column('question_type', sa.String(length=50), nullable=True, comment='Question category type'),
        sa.Column('expected_answer_points', sa.JSON(), nullable=False, comment='Expected criteria points for evaluating answers'),
        sa.Column('follow_up_question', sa.Text(), nullable=True, comment='Probing follow-up question'),
        sa.Column('candidate_answer', sa.Text(), nullable=True, comment='Submitted candidate answer text'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp when question was generated'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp when question was last updated'),
        sa.ForeignKeyConstraint(['session_id'], ['interview_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'order', name='uq_session_question_order'),
    )
    op.create_index(op.f('ix_interview_questions_id'), 'interview_questions', ['id'], unique=False)
    op.create_index(op.f('ix_interview_questions_session_id'), 'interview_questions', ['session_id'], unique=False)

    # 3. Create answer_evaluations table
    op.create_table(
        'answer_evaluations',
        sa.Column('id', sa.String(length=36), nullable=False, comment='Unique UUID evaluation identifier'),
        sa.Column('question_id', sa.String(length=36), nullable=False, comment='Foreign key referencing the evaluated question'),
        sa.Column('session_id', sa.String(length=36), nullable=False, comment='Foreign key referencing the parent interview session'),
        sa.Column('candidate_answer', sa.Text(), nullable=False, comment='Snapshot of candidate answer evaluated'),
        sa.Column('score', sa.Integer(), nullable=False, comment='Evaluation score from 0 to 100'),
        sa.Column('rating', sa.String(length=50), nullable=False, comment='Categorical rating (Needs Improvement, Fair, Good, Excellent)'),
        sa.Column('correct_points', sa.JSON(), nullable=False, comment='Concepts the candidate answered correctly'),
        sa.Column('missing_points', sa.JSON(), nullable=False, comment='Expected concepts omitted or poorly addressed'),
        sa.Column('strengths', sa.JSON(), nullable=False, comment='Strengths observed in candidate response'),
        sa.Column('improvements', sa.JSON(), nullable=False, comment='Actionable suggestions for improvement'),
        sa.Column('sample_answer', sa.Text(), nullable=False, comment='Concise, high-quality reference answer'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp when evaluation was recorded'),
        sa.ForeignKeyConstraint(['question_id'], ['interview_questions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['interview_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('question_id', name='uq_answer_evaluations_question_id'),
    )
    op.create_index(op.f('ix_answer_evaluations_id'), 'answer_evaluations', ['id'], unique=False)
    op.create_index(op.f('ix_answer_evaluations_question_id'), 'answer_evaluations', ['question_id'], unique=True)
    op.create_index(op.f('ix_answer_evaluations_session_id'), 'answer_evaluations', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_answer_evaluations_session_id'), table_name='answer_evaluations')
    op.drop_index(op.f('ix_answer_evaluations_question_id'), table_name='answer_evaluations')
    op.drop_index(op.f('ix_answer_evaluations_id'), table_name='answer_evaluations')
    op.drop_table('answer_evaluations')

    op.drop_index(op.f('ix_interview_questions_session_id'), table_name='interview_questions')
    op.drop_index(op.f('ix_interview_questions_id'), table_name='interview_questions')
    op.drop_table('interview_questions')

    op.drop_index(op.f('ix_interview_sessions_status'), table_name='interview_sessions')
    op.drop_index(op.f('ix_interview_sessions_id'), table_name='interview_sessions')
    op.drop_table('interview_sessions')
