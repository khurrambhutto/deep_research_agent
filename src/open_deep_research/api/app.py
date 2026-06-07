"""FastAPI application for the local Open Deep Research backend."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from open_deep_research.api.schemas import (
    ApiKeysUpdateRequest,
    ApiKeysUpdateResponse,
    ChatRequest,
    ChatResponse,
    ResearchFollowupRequest,
    RunCreateRequest,
    RunDetailResponse,
    RunEventResponse,
    RunSummaryResponse,
    SettingsResponse,
    SettingsUpdateRequest,
)
from open_deep_research.run_service import RunService
from open_deep_research.secrets import SecretStore
from open_deep_research.storage import ResearchStorage


storage = ResearchStorage()
secrets = SecretStore(storage)
run_service = RunService(storage, secrets)

app = FastAPI(title="Open Deep Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def reset_abandoned_runs() -> None:
    """Clear stale in-process work after reloads or restarts."""
    storage.reset_running_runs(
        "Backend restarted before this in-process research task finished. "
        "Start a new run."
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Return backend health."""
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunSummaryResponse, status_code=202)
async def create_run(
    request: RunCreateRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Start a new deep research run."""
    run = run_service.create_run(request.query, request.settings)
    background_tasks.add_task(run_service.execute_run, run["id"])
    return run


@app.get("/api/runs", response_model=list[RunSummaryResponse])
async def list_runs() -> list[dict]:
    """List saved research runs."""
    return storage.list_runs()


@app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: str) -> dict:
    """Load a saved research run with report, notes, sources, and messages."""
    run = storage.get_run_detail(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Research run not found.")
    return run


@app.get("/api/runs/{run_id}/events/history", response_model=list[RunEventResponse])
async def get_run_events(run_id: str, after_id: int | None = None) -> list[dict]:
    """Load saved run progress events."""
    if not storage.get_run(run_id):
        raise HTTPException(status_code=404, detail="Research run not found.")
    return storage.list_events(run_id, after_id=after_id)


@app.get("/api/runs/{run_id}/events")
async def stream_run_events(
    run_id: str,
    request: Request,
    after_id: int | None = None,
) -> StreamingResponse:
    """Stream run progress events as server-sent events."""
    if not storage.get_run(run_id):
        raise HTTPException(status_code=404, detail="Research run not found.")

    async def event_stream() -> AsyncIterator[str]:
        last_id = after_id
        terminal_statuses = {"completed", "failed", "needs_clarification"}
        while True:
            if await request.is_disconnected():
                break

            events = storage.list_events(run_id, after_id=last_id)
            for event in events:
                last_id = event["id"]
                yield (
                    f"event: {event['event_type']}\n"
                    f"data: {json.dumps(event)}\n\n"
                )

            run = storage.get_run(run_id)
            if run and run["status"] in terminal_statuses:
                break

            if not events:
                yield 'event: heartbeat\ndata: {"status": "waiting"}\n\n'
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/runs/{run_id}/chat", response_model=ChatResponse)
async def chat_about_report(run_id: str, request: ChatRequest) -> dict[str, str]:
    """Ask a lightweight follow-up question about a completed report."""
    try:
        answer = await run_service.chat_about_report(
            run_id,
            request.message,
            model=request.model,
            settings=request.settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"run_id": run_id, "answer": answer}


@app.post(
    "/api/runs/{run_id}/research-followup",
    response_model=RunSummaryResponse,
    status_code=202,
)
async def research_followup(
    run_id: str,
    request: ResearchFollowupRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Start an explicit full research follow-up using the previous report as context."""
    try:
        run = run_service.start_research_followup(
            run_id,
            request.query,
            settings=request.settings,
        )
        background_tasks.add_task(run_service.execute_run, run["id"])
        return run
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings() -> dict:
    """Load local backend settings and API key availability."""
    return {
        "settings": storage.get_settings(),
        "api_keys": secrets.list_available_keys(),
    }


@app.put("/api/settings", response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest) -> dict:
    """Update local backend settings."""
    settings = storage.update_settings(request.settings)
    return {"settings": settings, "api_keys": secrets.list_available_keys()}


@app.post("/api/settings/keys", response_model=ApiKeysUpdateResponse)
async def update_api_keys(request: ApiKeysUpdateRequest) -> dict:
    """Save or update provider API keys in OS keyring."""
    try:
        saved = secrets.save_api_keys(request.keys)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"saved": saved}
