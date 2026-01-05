
# filename: weather_langgraph_api.py
from typing import Annotated, List, TypedDict, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
import os
import json
from dotenv import load_dotenv
import httpx
import uvicorn

from langfuse import Langfuse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv('WEATHER_API_KEY')

# --------- Define a simple conversational state ----------
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat history as a list of messages"]

# helper functions
def _format_context(messages, k=3):
    lines = []
    for m in messages[-k*2:]:  # rough slice; adjust as needed
        role = ("User" if isinstance(m, HumanMessage)
                else "Assistant" if isinstance(m, AIMessage)
                else "Tool" if isinstance(m, ToolMessage)
                else "System")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# --------- Define a custom tool ----------
@tool
def get_weather(city: str) -> str:
    """Return a real weather report for a given city."""
    real_time_url = "https://api.tomorrow.io/v4/weather/realtime"

    print("Fetching real-time data for location: ", city)
    try:
        with httpx.Client(base_url=real_time_url, timeout=100) as client:
            response = client.get(
                real_time_url,
                params={"location": city, "apikey": api_key}
            )
            if response.status_code != 200:
                print("Failed to fetch real-time data: ", response.text)
                return False, {"error": response.text}
            return True, response.json()
    except Exception as e:
        print("Unexpected error while fetching real-time data: ", e)
        return False, {"error": str(e)}

# --------- Create the Ollama LLMs ----------
# IMPORTANT: Pick a router model that supports tool/function calling in Ollama.
ROUTER_MODEL = "qwen3:1.7b"       # e.g., llama3.2:3b, llama3.1:8b, qwen2.5:7b
SUMMARIZER_MODEL = "gemma3:4b"    # can be same or different; no tools bound here

router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0.2).bind_tools([get_weather])
summarizer_llm = ChatOllama(model=SUMMARIZER_MODEL, temperature=0.2)
insecure_client = httpx.Client(verify=False)
langfuse = Langfuse(host=os.getenv("LANGFUSE_HOST"),
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    httpx_client=insecure_client
                    )
with open(".env", mode="a", encoding="utf-8") as env:
    user_expert_prompt = langfuse.get_prompt(name="router", type="chat").prompt #label="production" by defualt
    print(user_expert_prompt)
    content = json.dumps(user_expert_prompt[0]["content"])
    user_expert_prompt = f"\nROUTER_SYSTEM_PROMPT={content}\n"
    env.write(user_expert_prompt)

    user_expert_prompt = langfuse.get_prompt(
        name="summariser", type="chat").prompt
    content = json.dumps(user_expert_prompt[0]["content"])
    user_expert_prompt = f"\nSUMMARISER_SYSTEM_PROMPT={content}\n"
    env.write(user_expert_prompt)
load_dotenv() #load env again after updating
# Optional system prompts to make behavior crisp
ROUTER_SYSTEM = SystemMessage(content=(
    os.getenv("ROUTER_SYSTEM_PROMPT")
))
SUMMARIZER_SYSTEM = SystemMessage(content=(
    os.getenv("SUMMARISER_SYSTEM_PROMPT")
))

# --------- Define the Router node ----------
def router_node(state: ConversationState, config: RunnableConfig = None) -> ConversationState:
    """
    Calls the router LLM. It may produce a normal response or a tool call.
    We always append the AIMessage; routing decides the next step.
    """
    messages = state.get("messages", [])
    # Prepend a system prompt once (optional)
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [ROUTER_SYSTEM] + messages

    ai: AIMessage = router_llm.invoke(messages, config=config)
    messages = messages + [ai]
    return {"messages": messages}

# --------- Conditional routing after Router ----------
def route_after_router(state: ConversationState):
    """
    If the last AI message contains tool calls, go to tools; else, go to summarizer.
    Robustly checks both `AIMessage.tool_calls` and `additional_kwargs['tool_calls']`.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        tool_calls = getattr(last, "tool_calls", None) or getattr(last, "additional_kwargs", {}).get("tool_calls")
        if tool_calls:
            return "tools"
    return "summarizer"

# --------- Define the Summarizer node ----------
def summarizer_node(state: ConversationState, config: RunnableConfig = None) -> ConversationState:
    """
    Reads recent ToolMessages (if any) plus the user's last question,
    and produces the final assistant reply.
    """
    messages = state.get("messages", [])
    print(f"Passed to summarizer: \n{messages}")

    # Collect recent tool results since the last AIMessage
    tool_outputs: List[str] = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Each ToolMessage.content is the tool's returned string
            if isinstance(msg.content, str):
                tool_outputs.append(msg.content)
            else:
                # Some tools may return structured data
                tool_outputs.append(str(msg.content))
        elif isinstance(msg, AIMessage):
            # Stop at the prior AIMessage (i.e., before tools ran)
            break

    # Get the last user query
    user_query = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Build a focused prompt for the summarizer
    tool_summary_text = "\n".join(reversed(tool_outputs)) if tool_outputs else "(No tool results available.)"
    history = _format_context(messages, k=3)
    user_prompt = HumanMessage(content=f"User Query:\n{user_query}\n\nTool Results:\n{tool_summary_text}\n\nContext:\n{history}")

    # Ensure a system prompt precedes the summarizer call
    summarizer_messages = [SUMMARIZER_SYSTEM, user_prompt]

    ai: AIMessage = summarizer_llm.invoke(summarizer_messages, config=config)
    messages = messages + [ai]
    return {"messages": messages}

# --------- Build the graph ----------
graph = StateGraph(ConversationState)

graph.add_node("router", router_node)
graph.add_node("tools", ToolNode(tools=[get_weather]))  # Executes tool calls
graph.add_node("summarizer", summarizer_node)

# Wire up edges
graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_after_router)  # "tools" or "summarizer"
graph.add_edge("tools", "summarizer")
graph.add_edge("summarizer", END)

# Compile with an in-memory checkpoint (optional)
graph_app = graph.compile(checkpointer=InMemorySaver())

api = FastAPI(title="Weather LangGraph API", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # or ["POST"] if you want to restrict later
    allow_headers=["*"],   # include "content-type", "authorization" etc. if you want to be explicit
)

class GenerateRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = "demo-session-1"
    stream: Optional[bool] = False

class EventItem(BaseModel):
    node: str
    text: str

class GenerateResponse(BaseModel):
    assistant_reply: str
    events: Optional[List[EventItem]] = None

@api.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    Accepts a user input and optional thread_id.
    Preserves conversation state via checkpointer.
    If stream=True, returns intermediate node events along with final reply.
    """
    # Load previous conversation from checkpointer (StateSnapshot)
    prev_snapshot = graph_app.get_state({"configurable": {"thread_id": req.thread_id}})
    old_msgs: List[BaseMessage] = []
    if prev_snapshot is not None and getattr(prev_snapshot, "values", None):
        old_msgs = prev_snapshot.values.get("messages", [])

    # Append new user message
    state: ConversationState = {"messages": old_msgs + [HumanMessage(content=req.user_input)]}

    events_out: List[EventItem] = []
    final: Optional[Dict[str, Any]] = None

    if req.stream:
        # Stream intermediate steps
        for event in graph_app.stream(state, config={"configurable": {"thread_id": req.thread_id}}):
            for k, v in event.items():
                # Try to surface last message content for each node
                last_text = ""
                try:
                    last_text = v["messages"][-1].content or ""
                except Exception:
                    last_text = str(v)
                events_out.append(EventItem(node=k, text=last_text))
                if k == "summarizer":
                    final = v

        if final is None:
            final = graph_app.invoke(state, config={"configurable": {"thread_id": req.thread_id}})
    else:
        # Single final result
        final = graph_app.invoke(state, config={"configurable": {"thread_id": req.thread_id}})

    assistant_reply = ""
    if final and isinstance(final, dict) and "messages" in final:
        try:
            assistant_reply = final["messages"][-1].content or ""
        except Exception:
            assistant_reply = str(final)

    return GenerateResponse(assistant_reply=assistant_reply, events=events_out or None)

if __name__ == "__main__":
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=True)
