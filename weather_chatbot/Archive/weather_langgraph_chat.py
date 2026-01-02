# filename: weather_langgraph_chat.py
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
import os
from dotenv import load_dotenv
import httpx

load_dotenv()
api_key= os.getenv('WEATHER_API_KEY')
# --------- Define a simple conversational state ----------
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], "Chat history as a list of messages"]

#helper funtions 

def _format_context(messages, k=3):
    lines = []
    for m in messages[-k*2:]:  # rough slice; adjust as needed
        role = ("User" if isinstance(m, HumanMessage)
                else "Assistant" if isinstance(m, AIMessage)
                else "Tool" if isinstance(m, ToolMessage)
                else "System")
        # Only include text content
        
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
        with httpx.Client(
            base_url=real_time_url, timeout=100) as client:
            response = client.get(
                real_time_url,
                params={"location": city, "apikey": api_key}
            )
            if response.status_code != 200:
                print(
                    "Failed to fetch real-time data: ", response.text)
                return False, {"error": response.text}
            return True, response.json()
    except Exception as e:
        print(
            "Unexpected error while fetching real-time data: ", e)
        return False, {"error": str(e)}


# --------- Create the Ollama LLMs ----------
# IMPORTANT: Pick a router model that supports tool/function calling in Ollama.
ROUTER_MODEL = "qwen3:1.7b"       # e.g., llama3.2:3b, llama3.1:8b, qwen2.5:7b
SUMMARIZER_MODEL = "gemma3:4b"    # can be same or different; no tools bound here

router_llm = ChatOllama(model=ROUTER_MODEL, temperature=0.2).bind_tools([get_weather])
summarizer_llm = ChatOllama(model=SUMMARIZER_MODEL, temperature=0.2)

# Optional system prompts to make behavior crisp
ROUTER_SYSTEM = SystemMessage(content=(
    "You are a routing assistant. "
    "When a user asks for something that requires a tool, emit a tool call. "
    "If a tool is not needed, respond normally."
))
SUMMARIZER_SYSTEM = SystemMessage(content=(
    "You are a summarizer. Read the user's question and the latest tool results, "
    "and produce a clear, concise final answer."
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
app = graph.compile(checkpointer=InMemorySaver())


# --------- Run a simple session ----------
if __name__ == "__main__":
    thread_id = "demo-session-1"
    state: ConversationState = {"messages": []}
    while(True):
        user_input = input('User: ').strip()
        
        # Load previous conversation from checkpointer
        prev = app.get_state({"configurable": {"thread_id": thread_id}})
        messages = prev.values.get("messages", []) + [HumanMessage(content=user_input)]
        state = {"messages": messages}

        print("---- Streaming events ----")
        final = None
        for event in app.stream(state, config={"configurable": {"thread_id": thread_id}}):
            for k, v in event.items():
                print(f"[{k}]: {v}")
                if k == "summarizer":
                    final = v

        if final is None: #fallback
            final = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
        print("\n---- Final state ----")
        print(f"{final}")
        print("\n---- Final assistant reply ----")
        print(final["messages"][-1].content or "(No content from model)")
