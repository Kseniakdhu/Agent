from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage

import joblib
import numpy as np
import pandas as pd


llm = ChatOpenAI(
    model_name='gpt-4o-mini',
    temperature=0,
    openai_api_base='https://api.vsegpt.ru/v1',
    openai_api_key= # KEY
)

def load_model(path="iris_model.pkl"):
    global model, trained_feature_names
    try:
        model = joblib.load(path)
        trained_feature_names = getattr(model, "feature_names_in_", None)
        return True
    except Exception:
        model = None
        trained_feature_names = None
        return False

def iris_model_tool(text: str) -> str:
    if model is None:
        return "Ошибка: модель не загружена"
    parts = str(text).replace(",", " ").split()
    if len(parts) != 4:
        return "Error: provide 4 numeric features, e.g. '5.1 3.5 1.4 0.2'"
    try:
        feats = np.array([float(p) for p in parts]).reshape(1, -1)
    except Exception:
        return "Error: features must be numeric"
    if trained_feature_names is not None:
        pred = model.predict(pd.DataFrame(feats, columns=trained_feature_names))[0]
    else:
        pred = model.predict(feats)[0]
    names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    try:
        return f"Вид ириса: {names[int(pred)]}"
    except Exception:
        return f"Вид ириса: {pred}"
    

class State(TypedDict):
    message: Annotated[List, "add_message"]
    
def call_model(state: State):
    if len(state['message']) == 1 and isinstance(state['message'][0], HumanMessage):
        message = [
            SystemMessage(content=(
                "You are a smart machine and you have to classify iris flowers by parameters. "
                "You have a tool (iris_model) which is already trained and makes a prediction. "
                "You will receive a message from the user as input; if it contains numeric features or explicit request to predict the iris species, call the tool and return its answer. "
                "Otherwise just answer the question without calling the tool."
            )),
            state['message'][0],
        ]
    else:
        message = state['message']

    response = llm.invoke(message)
    if isinstance(response, (list, tuple)):
        resp_val = list(response)
    else:
        resp_val = [response]
    return {'message': resp_val}
        
def tools_node(state: State):
    last = state['message'][-1]
    text = getattr(last, 'content', str(last))
    parts = str(text).replace(',', ' ').split()
    if model is None:
        res = 'Ошибка: модель не загружена'
    elif len(parts) >= 4:
        try:
            feats = np.array([float(p) for p in parts[:4]]).reshape(1, -1)
            if trained_feature_names is not None:
                pred = model.predict(pd.DataFrame(feats, columns=trained_feature_names))[0]
            else:
                pred = model.predict(feats)[0]
            names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            try:
                res = f"Вид ириса: {names[int(pred)]}"
            except Exception:
                res = f"Вид ириса: {pred}"
        except Exception:
            res = 'Error: unable to parse numeric features'
    else:
        res = 'No numeric features found; nothing to predict.'
    state.setdefault('message', []).append(SystemMessage(content=res))
    return {'message': state['message']}

graph = StateGraph(State)
graph.add_node('model', call_model)
graph.add_node('tools', tools_node)
graph.set_entry_point('model')

def should_continue(state: State):
    last_message = state['message'][-1]
    if hasattr(last_message, 'tools_calls') and last_message.tools_calls:
        return 'tools'
    return END

graph.add_conditional_edges('model', should_continue)
graph.add_edge('tools', 'model')
agent_graph = graph.compile() 

	