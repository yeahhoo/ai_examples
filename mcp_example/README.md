# MCP integration with LLM

A simple example of MCP server that is integrated with LLM to perform "application specific" operations that LLM don't have a possibility to do.
In this case MCP server it's just a document system that can perform some CRUD operation on documents stored in the local folder .documents.
The MCP server is written with FastMCP library and currently supports tools and prompts.
The MCP client serves two purposes: it acts as a chat and it's also MCP client connected to the MCP server. 
So in case LLM needs to perform a specific operation it can do it via the MCP.
The MCP client uses Groq as LLM, MCPAgent and mcp-use libs to perform MCP and chat-related activities. Basically it's just a slightly modified example of mcp-use: https://github.com/mcp-use/mcp-use/tree/main/libraries/python/examples


## Launching the code

First you need to have Groq API-KEY

```bash
export GROQ_API_KEY=<your_key>
```

Then you need to install all the dependencies

```bash
python3 -m venv .venv # initiate a temporary environment venv
source .venv/bin/activate # activate temporary environment venv
pip install fastmcp groq requests mcp-use langchain groq langchain_groq
```

Then you need to start the server:

```bash
python mcp-server.py
```

Then run the client:

```bash
python mcp-client.py
```

## Playing with chat

After that you can launch some requests that will demonstrate the power of LLM plus MCP. 
For example after launching the prompts below you will see that the documents with the related content will be created in the .documents folder.

```txt
Create a new document with name 'first_fly.txt' about the first fly in space
or
could you generate a text about first formula 1 race and store the document with name "formula1.txt"?
or
could you summarize the document "formula1.txt" in 2 sentences?
```

## Troubleshoting MCP server

You can also troubleshoot MCP server separately, but for that you will need NodeJS installed. After that launch MCP inspector

```bash
npx @modelcontextprotocol/inspector python mcp-server.py
```

