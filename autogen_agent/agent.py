# autogen_agent/agent.py
import os
import time
import autogen
from autogen import register_function
from shared.tools import web_search

def run_autogen_agent(topic: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    llm_config = {
        "config_list": [{"model": model, "api_key": api_key, "api_type": "anthropic"}],
        "temperature": 0.2,
    }

    # ---------------------------------------------------------------
    # Phase 1: Research Agent + UserProxy for tool execution
    # ---------------------------------------------------------------
    print(f"\n[AutoGen] 🔍 Activating Researcher Agent for topic: '{topic}'...")

    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message=(
            "You are a Senior Researcher. Use the web_search tool ONCE to gather facts about the topic. "
            "After the search, compile your findings into structured research notes covering:\n"
            "1. Core concept and mechanism.\n"
            "2. Key advantages.\n"
            "3. Major players/companies.\n"
            "4. Critical technical/market challenges.\n"
            "Do NOT call web_search more than once. When your notes are complete, end your reply with TERMINATE."
        ),
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    register_function(
        web_search,
        caller=researcher,
        executor=user_proxy,
        name="web_search",
        description="Search the web for information on a given query topic.",
    )

    user_proxy.initiate_chat(
        researcher,
        message=f"Research the topic: '{topic}'. Use web_search once, then provide structured research notes."
    )

    # Extract the last Researcher message as research notes
    research_notes = ""
    for msg in reversed(user_proxy.chat_messages[researcher]):
        if msg["role"] == "assistant":
            research_notes = msg["content"].replace("TERMINATE", "").strip()
            break

    # ---------------------------------------------------------------
    # Rate Limit Spacer
    # ---------------------------------------------------------------
    print("\n[AutoGen] ⏳ Sleeping for 15 seconds to respect API rate limits...")
    time.sleep(15)

    # ---------------------------------------------------------------
    # Phase 2: Writer Agent
    # ---------------------------------------------------------------
    print(f"\n[AutoGen] ✍️ Activating Writer Agent to draft report...")

    writer = autogen.AssistantAgent(
        name="Writer",
        system_message=(
            "You are a professional Technical Writer who excels at creating clear, structured reports. "
            "Write in clean Markdown. End your reply with TERMINATE when done."
        ),
        llm_config={**llm_config, "temperature": 0.5},
    )

    user_proxy2 = autogen.UserProxyAgent(
        name="UserProxy2",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        code_execution_config=False,
    )

    user_proxy2.initiate_chat(
        writer,
        message=(
            f"Write a professional report on '{topic}' based on the following research notes:\n\n"
            f"{research_notes}\n\n"
            "Structure the report in Markdown with:\n"
            "1. Title (H1)\n"
            "2. Executive Summary\n"
            "3. Detailed Analysis (Mechanism & Benefits)\n"
            "4. Industry Players & Progress\n"
            "5. Technical Challenges & Future Outlook\n"
            "Use a professional, objective tone."
        )
    )

    # Extract the Writer's report
    final_report = ""
    for msg in reversed(user_proxy2.chat_messages[writer]):
        if msg["role"] == "assistant":
            final_report = msg["content"].replace("TERMINATE", "").strip()
            break

    return {
        "research_notes": research_notes,
        "final_report": final_report
    }
