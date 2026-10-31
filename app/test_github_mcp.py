import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(Path(".env"))

token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not token:
    raise RuntimeError("GitHub token is missing from BackEnd/.env")

params = StdioServerParameters(
    command="github-mcp-server",
    args=["stdio", "--read-only"],
    env={
        **os.environ,
        "GITHUB_PERSONAL_ACCESS_TOKEN": token,
    },
)

async def main():
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.list_tools()

            print("MCP CONNECTION: SUCCESS")
            print("TOOLS DISCOVERED:", len(result.tools))

            tool = next(
                (t for t in result.tools if t.name == "get_file_contents"),
                None,
            )
            if tool:
                print("Tool found:", tool.name)
                print("Input schema:", tool.input_schema)

if __name__ == "__main__":
    asyncio.run(main())
