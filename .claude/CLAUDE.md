# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Implement, test, and benchmark the same two-agent pipeline (Researcher + Technical Writer) across three orchestration styles using Google Gemini:

1. **Native SDK** — imperative, direct scripting with `google-genai`
2. **LangGraph** — stateful, cyclic graph control
3. **CrewAI** — declarative, role-based orchestration

## Running the Project

Activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run a single framework:
```bash
python run.py --framework native --topic "solid-state batteries"
python run.py --framework langgraph --topic "quantum computing"
python run.py --framework crewai --topic "nuclear fusion"
```

Run all three and print a telemetry comparison table:
```bash
python run.py --framework all --topic "solid-state batteries"
```

Each run writes `notes_{framework}.md` and `report_{framework}.md` to the project root.

Install dependencies (if `.venv` is not already set up):
```bash
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` and add your key:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-flash-lite-latest
```

**Model choice matters**: `gemini-2.5-flash` has a 20 RPD free-tier quota (easily exhausted by agent loops). Use `gemini-flash-lite-latest` (Gemini 1.5 Flash 8B) for 1,500 RPD.

## Architecture

Every `run_*_agent` function follows the same two-phase contract:

```
shared/tools.py          # DuckDuckGo web_search() with mock fallback dict
    ↑ imported by all three agents

native_agent/agent.py    → run_native_agent(topic)    → {"research_notes", "final_report"}
langgraph_agent/agent.py → run_langgraph_agent(topic) → {"research_notes", "final_report"}
crewai_agent/agent.py    → run_crewai_agent(topic)    → {"research_notes", "final_report"}

run.py                   # CLI: invokes the above, writes output files, prints telemetry table
```

**Phase 1 — Researcher**: calls `web_search` once, returns structured notes.  
**Phase 2 — Writer**: consumes notes, returns a Markdown report.  
A `time.sleep(15)` between phases is intentional rate-limit protection — do not remove.

### Framework-specific details

**Native** (`google-genai` SDK): the SDK handles the tool-call execution loop automatically when Python functions are passed as `tools=`. Two sequential `client.models.generate_content()` calls.

**LangGraph**: `StateGraph` with nodes `researcher → (tools loop) → writer`. State is `AgentState(messages, topic, research_notes, final_report)`. The `should_continue` conditional edge checks `last_message.tool_calls` to route back through `ToolNode` or forward to `writer_node`.

- Do **not** use `SystemMessage` with Gemini via LangChain — the Gemini API requires strict `user → model → tool → model` turn alternation; `SystemMessage` breaks this and produces a 400 error. Embed all instructions inside the initial `HumanMessage`.
- `safe_extract_text()` in `langgraph_agent/agent.py` normalises LangChain/Gemini content that is returned as a `list` of parts instead of a plain string (prevents `TypeError: write() argument must be str, not list`).

**CrewAI**: declarative `Agent/Task/Crew` with `Process.sequential`. The model string must be prefixed with `gemini/` (LiteLLM convention) — handled automatically in `run_crewai_agent`. A `sleep_callback` on `research_task` enforces the 15-second gap.

## Known Constraints

- All agents are constrained to **exactly one** `web_search` call to stay within RPM limits. Enforced via prompt instructions, not code.
- The mock fallback in `shared/tools.py` covers: `"solid-state batteries"`, `"quantum computing"`, `"nuclear fusion"`, `"generative ai agents"`. Other topics fall back to a generic stub.

## Benchmark Results

*Model: `gemini-flash-lite-latest` (Gemini 1.5 Flash 8B)*

| Framework     | Status  | Time (s)    | Notes Length | Report Length |
| :------------ | :------ | :---------- | :----------- | :------------ |
| **NATIVE**    | Success | **24.43s**  | 4,105 chars  | 5,276 chars   |
| **LANGGRAPH** | Success | **34.17s**  | 2,754 chars  | 4,721 chars   |
| **CREWAI**    | Success | **36.67s**  | 2,655 chars  | 4,993 chars   |
