from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from settings import settings 
from groq import Groq
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. create graph state
class State(TypedDict):
    value: int


# 2. define nodes
def add(state: State) -> State:
    return {
        "value": state["value"] + 2,
    }


# 3. create graph
graph = StateGraph(State)

graph.add_node('add', add)

# 5. connect nodes with edges
graph.add_edge(START, "add")
graph.add_edge("add", END)

# 6. compile graph
app = graph.compile()


# del /s /q *.pyc
if __name__ == "__main__":
    # 7. launch
    result = app.invoke({"value": 8})

    print(result)
    print(settings.LANGSMITH_API_KEY)
    # llm = ChatGroq(
    #     model="llama-3.1-8b-instant",
    #     api_key=settings.LANGSMITH_API_KEY,
    # )

    #os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

    llm_google = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key = settings.LANGSMITH_API_KEY, temperature=0.2, max_tokens=None, timeout=None, max_retries=2)
    
    #response = llm_google.invoke('hello world')
    #print(response)