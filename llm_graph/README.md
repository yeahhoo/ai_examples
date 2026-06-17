# Example of LangGraph agent

The folder contains a simple script for making agent using LangGraph.

## Prerequisites

1. init virtual env

```bash
python -m venv .venv
```

2. activate virtual env

```bash
linux: source .venv/bin/activate
windows: .venv\Scripts\activate
```

3. install dependencies and store them in a file

```bash
pip install langchain langgraph langchain-openai langchain_groq langchain-google-genai groq pydantic pydantic-settings
pip freeze > requirements.txt
```

## Clean-up

Once you are done with the work you can launch:

```bash
deactivate
pip freeze | xargs pip uninstall -y
pip cache purge
```