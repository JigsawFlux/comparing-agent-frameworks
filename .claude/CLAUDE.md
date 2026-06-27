# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Implement, test, and benchmark the same two-agent pipeline (Researcher + Technical Writer) across
three orchestration frameworks using Claude (Anthropic) as the LLM:

1. **LangGraph** — stateful, cyclic graph control (foundation for ReAct, Plan-and-Execute, ReWOO, Reflexion patterns)
2. **CrewAI** — declarative, role-based orchestration (foundation for Hierarchical multi-agent patterns)
3. **AutoGen** — conversation-based multi-agent orchestration (foundation for Peer-to-peer and Consensus patterns)

## Running the Project

Activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run a single framework:
```bash
python run.py --framework langgraph --topic "solid-state batteries"
python run.py --framework crewai --topic "quantum computing"
python run.py --framework autogen --topic "nuclear fusion"
```

Run all three and print a telemetry comparison table:
```bash
python run.py --framework all --topic "solid-state batteries"
```

Each run writes `notes_{framework}.md` and `report_{framework}.md` to the project root.

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` and add your key:
```
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-sonnet-4-6
```

## Architecture

Every `run_*_agent` function follows the same two-phase contract:

```
shared/tools.py            # DuckDuckGo web_search() with mock fallback dict
    ↑ imported by all three agents

langgraph_agent/agent.py   → run_langgraph_agent(topic) → {"research_notes", "final_report"}
crewai_agent/agent.py      → run_crewai_agent(topic)    → {"research_notes", "final_report"}
autogen_agent/agent.py     → run_autogen_agent(topic)   → {"research_notes", "final_report"}

run.py                     # CLI: invokes the above, writes output files, prints telemetry table
```

**Phase 1 — Researcher**: calls `web_search` once, returns structured notes.
**Phase 2 — Writer**: consumes notes, returns a Markdown report.
A `time.sleep(15)` between phases is intentional rate-limit protection — do not remove.

### Framework-specific details

**LangGraph**: `StateGraph` with nodes `researcher → (tools loop) → writer`. State is
`AgentState(messages, topic, research_notes, final_report)`. The `should_continue` conditional
edge checks `last_message.tool_calls`. Uses `ChatAnthropic` — Claude handles `SystemMessage`
correctly, so instructions live in a `SystemMessage` at the top of `initial_state`.

**CrewAI**: Declarative `Agent/Task/Crew` with `Process.sequential`. Model string uses
`anthropic/{model}` prefix (LiteLLM convention) — handled automatically in `run_crewai_agent`.
A `sleep_callback` on `research_task` enforces the 15-second rate-limit gap.

**AutoGen**: Two-phase conversation pattern. Phase 1 pairs a `Researcher` (`AssistantAgent`) with
a `UserProxyAgent` (`human_input_mode="NEVER"`) that executes `web_search` via `register_function`.
Terminates on `"TERMINATE"` in the last message. Phase 2 uses a separate `Writer`/`UserProxy2`
pair with `max_consecutive_auto_reply=1`. AutoGen prints verbose conversation logs by default —
this is expected behaviour.

## Known Constraints

- All agents are constrained to **exactly one** `web_search` call via prompt instructions.
- The mock fallback in `shared/tools.py` covers: `"solid-state batteries"`, `"quantum computing"`,
  `"nuclear fusion"`, `"generative ai agents"`. Other topics fall back to a generic stub.
- AutoGen's `pyautogen` uses `api_type: "anthropic"` in `config_list` for Claude.

## Context for Next Week

These three frameworks were chosen as the natural implementation homes for the agentic patterns
explored in the follow-up post:

| Pattern | Framework |
| :--- | :--- |
| ReAct, Plan-and-Execute, ReWOO, Reflexion | LangGraph |
| Hierarchical Agent | CrewAI (`Process.hierarchical`) |
| Peer-to-peer Network | AutoGen (`GroupChat`) |
| Consensus / Joint | AutoGen (multi-agent debate) |
| Human-in-the-loop | LangGraph (`interrupt_before/after`) |
