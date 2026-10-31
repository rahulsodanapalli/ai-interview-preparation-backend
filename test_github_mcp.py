import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    env_file = Path.cwd() / ".env"

    if not env_file.is_file():
        print("ERROR: .env file not found:", env_file)
        return

    load_dotenv(dotenv_path=env_file)

    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        print("ERROR: GITHUB_PERSONAL_ACCESS_TOKEN missing from .env")
        return

    print("GitHub token found: YES")

    server_env = os.environ.copy()
    server_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token

    params = StdioServerParameters(
        command="github-mcp-server",
        args=["stdio", "--read-only"],
        env=server_env,
    )

    try:
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                print("MCP CONNECTION: SUCCESS")

                tools = await session.list_tools()
                print("TOOLS DISCOVERED:", len(tools.tools))

                tool = next(
                    (t for t in tools.tools if t.name == "get_file_contents"),
                    None,
                )
                if tool is None:
                    print("ERROR: get_file_contents not found")
                    return

                schema = getattr(tool, "inputSchema", None)
                if schema is None:
                    schema = getattr(tool, "input_schema", {})

                properties = schema.get("properties", {})
                args = {
                    "owner": "octocat",
                    "repo": "Hello-World",
                    "path": "README",
                }

                if "ref" in properties:
                    args["ref"] = "master"
                elif "branch" in properties:
                    args["branch"] = "master"

                print("Calling get_file_contents...")
                result = await session.call_tool("get_file_contents", args)

                is_error = getattr(result, "is_error", False) or getattr(result, "isError", False)
                print(
                    "MCP TOOL CALL:",
                    "FAILED" if is_error else "SUCCESS",
                )

                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        print("Text:", item.text[:2000])
                    if hasattr(item, "resource") and hasattr(item.resource, "text") and item.resource.text:
                        print("Resource text:", item.resource.text[:2000])

    except Exception as exc:
        print("MCP TEST FAILED:", type(exc).__name__, str(exc))


if __name__ == "__main__":
    asyncio.run(main())
