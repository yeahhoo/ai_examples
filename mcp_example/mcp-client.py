import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client
from groq import Groq

from mcp_use import MCPAgent, MCPClient
from langchain_groq import ChatGroq

API_KEY = os.environ.get("GROQ_API_KEY")

async def chat(agent, client):
    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("==================================\n")

    try:
        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Check for clear history command
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                # Run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        # Clean up
        if client and client.sessions:
            await client.close_all_sessions()

async def main():
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    client = MCPClient.from_config_file(
        os.path.join(os.path.dirname(__file__), "mcp-servers.json")
    )

    # Create LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=API_KEY,
    )

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, verbose=True, max_steps=30, memory_enabled=True)
    await chat(agent, client)

if __name__ == "__main__":
    asyncio.run(main())