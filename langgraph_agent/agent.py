# langgraph_agent/agent.py
import os
import time
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from shared.tools import web_search

@tool
def search_tool(query: str) -> str:
    """
    Search the web for information on a given query topic.

    Args:
        query (str): The search term or topic.

    Returns:
        str: Search results text.
    """
    return web_search(query)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    research_notes: str
    final_report: str

def researcher_node(state: AgentState):
    print("\n[LangGraph] 🔍 Activating Researcher Node...")
    messages = list(state.get("messages", []))

    model = ChatAnthropic(
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.2
    )
    model_with_tools = model.bind_tools([search_tool])
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode([search_tool])

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]['name']
        print(f"[LangGraph] 🔀 Tool call detected: '{tool_name}'. Routing to 'tools' node.")
        return "tools"
    print("[LangGraph] 🔀 No tool calls. Routing to 'writer' node.")
    return "writer"

def writer_node(state: AgentState):
    print("\n[LangGraph] ✍️ Activating Writer Node...")
    topic = state["topic"]

    research_notes = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            research_notes = msg.content
            break

    if not research_notes:
        research_notes = "No research notes were gathered."

    print("\n[LangGraph] ⏳ Sleeping for 15 seconds to respect API rate limits...")
    time.sleep(15)

    model = ChatAnthropic(
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.5
    )

    prompt = (
        f"You are a Technical Writer. Write a professional report on '{topic}' based on the following research notes:\n\n"
        f"{research_notes}\n\n"
        "Your report must be structured in professional Markdown and include:\n"
        "1. Title (H1)\n"
        "2. Executive Summary\n"
        "3. Detailed Analysis (Mechanism & Benefits)\n"
        "4. Industry Players & Progress\n"
        "5. Technical Challenges & Future Outlook\n"
        "Use a professional, objective tone."
    )

    response = model.invoke([
        SystemMessage(content="You are a professional technical writer who excels at creating structured reports."),
        HumanMessage(content=prompt)
    ])

    return {
        "research_notes": research_notes,
        "final_report": response.content
    }

def run_langgraph_agent(topic: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {"tools": "tools", "writer": "writer"}
    )
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("writer", END)

    app = workflow.compile()

    initial_state = {
        "topic": topic,
        "messages": [
            SystemMessage(content=(
                "You are an expert researcher. Use the web_search tool to gather facts. "
                "You are strictly limited to calling web_search ONCE. "
                "After the search, summarize your findings into structured research notes covering:\n"
                "1. Core concept and mechanism.\n"
                "2. Key advantages.\n"
                "3. Major players/companies.\n"
                "4. Critical technical/market challenges.\n"
                "Do NOT call tools again after your first search."
            )),
            HumanMessage(content=f"Please research '{topic}'.")
        ]
    }

    result_state = app.invoke(initial_state)

    return {
        "research_notes": result_state.get("research_notes", ""),
        "final_report": result_state.get("final_report", "")
    }
