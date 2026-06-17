import random
from typing import TypedDict, Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# Graph State
class State(TypedDict):
    graph_state: str


# Graph Nodes
def node_1(state: State) -> State:
    print('__node_1__')
    return {'graph_state': state['graph_state'] + 'I am'}


def node_2(state: State) -> State:
    print('__node_2__')
    return {'graph_state': state['graph_state'] + ' happy!'}


def node_3(state: State) -> State:
    print('__node_3__')
    return {'graph_state': state['graph_state'] + ' sad!'}


# Graph Edges
def decide_mood(state: State) -> Literal['node_2', 'node_3']:
    print('__decide_mood__')
    user_input = state['graph_state']

    if random.random() < 0.5:
        return 'node_2'
    return 'node_3'


# Build Graph
builder = StateGraph(State)

builder.add_node('node_1', node_1)
builder.add_node('node_2', node_2)
builder.add_node('node_3', node_3)

builder.add_edge(START, 'node_1')
builder.add_conditional_edges('node_1', decide_mood)
builder.add_edge('node_2', END)
builder.add_edge('node_3', END)

graph = builder.compile()

if __name__ == '__main__':
    result = graph.invoke({'graph_state': 'Hello! My name is Sasha. '})

    print(result['graph_state'])