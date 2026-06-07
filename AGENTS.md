# AGENTS.md

## Project Direction

Build a local-first UI for Open Deep Research so users can run deep research, configure providers, save reports, and continue chatting with the completed research.

Recommended stack:

- Frontend: React + Vite + TypeScript
- UI: Tailwind CSS + shadcn/ui
- Data fetching: TanStack Query
- Backend: FastAPI
- Local storage: SQLite

Keep the existing LangGraph research agent as the core research engine. The UI should call backend endpoints that start runs, stream status, save outputs, and load previous research sessions.

## Secret Handling

Do not commit real API keys.

`.env` is ignored and should stay local-only. `.env.example` may show placeholder setup examples.

The frontend should let users enter provider keys from a Settings page, but those keys should not be stored in frontend source code. Prefer local backend-side storage:

- Best: OS keyring through Python `keyring`
- Good: encrypted SQLite
- Dev-only: ignored local config file

For runtime execution, use `GET_API_KEYS_FROM_CONFIG=true` and pass keys through LangGraph run config instead of hardcoding them.

## Saved Research

Save research data locally so users can revisit past work.

Suggested SQLite tables:

- `research_runs`
- `messages`
- `reports`
- `sources`
- `notes`
- `settings`
- `api_key_refs`

Each run should preserve:

- Original user query
- Model/search settings
- Intermediate notes
- Search sources
- Final report
- Follow-up chat messages

## New Feature: Post-Report Chat

After a final report is generated, the user should be able to keep chatting with the agent about that report.

This should be a lightweight chat flow, not a full deep-research rerun by default.

Recommended behavior:

- User asks a follow-up after the report is done.
- Backend loads the saved report, notes, sources, and prior chat messages for that `research_run_id`.
- Backend sends that context to the selected model.
- Model answers using the existing saved research.
- The follow-up message and answer are saved locally.

Use this mode for questions like:

- "Summarize this report in 5 bullets."
- "What sources support this claim?"
- "Expand the historical background."
- "Turn this into a presentation outline."
- "Add a concise executive summary."

## Follow-Up Research Mode

Add a separate explicit mode for follow-ups that need new information.

Use two user-facing modes:

- Ask About Report: uses existing report, notes, and sources only. Fast and cheap.
- Research Follow-up: runs the research graph again with the previous report as context. Slower, but can fetch new sources.

Do not run full research again unless the user explicitly chooses Research Follow-up or asks for new/current sources.

## Backend Endpoints To Add

Suggested endpoints:

- `POST /api/runs` - start a new deep research run
- `GET /api/runs` - list saved runs
- `GET /api/runs/{run_id}` - load a run with report, notes, and sources
- `POST /api/runs/{run_id}/chat` - ask a lightweight follow-up about the saved report
- `POST /api/runs/{run_id}/research-followup` - start a deeper follow-up research run
- `GET /api/settings` - load local settings
- `PUT /api/settings` - update model/search settings
- `POST /api/settings/keys` - save/update provider keys locally

## UI Pages

Suggested pages:

- Dashboard: previous research runs
- New Research: prompt, model choices, search provider, limits
- Run View: live progress, sources, notes, final report
- Report Chat: post-report follow-up conversation
- Settings: API keys, model defaults, search settings, storage options

## Implementation Notes

Keep the post-report chat endpoint separate from the main research graph at first. It is simpler, cheaper, and more predictable.

Later, if needed, add a LangGraph branch for follow-up research.

