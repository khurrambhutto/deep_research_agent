"""Focused backend tests for local storage and API contracts."""

from __future__ import annotations

import asyncio

from fastapi import BackgroundTasks, HTTPException

from open_deep_research.api.schemas import ChatRequest, RunCreateRequest
from open_deep_research.run_service import RunService
from open_deep_research.storage import ResearchStorage


class DummySecrets:
    """Test secret provider that never returns real keys."""

    def get_api_keys(self) -> dict[str, str]:
        return {}

    def list_available_keys(self) -> list[dict]:
        return []

    def save_api_keys(self, keys: dict[str, str]) -> list[dict]:
        return [{"provider": key, "stored": bool(value)} for key, value in keys.items()]


class NoopRunService(RunService):
    """Run service that creates rows but does not call external models."""

    async def execute_run(self, run_id: str) -> None:
        self.storage.append_event(
            run_id,
            "test_noop",
            "Test run execution skipped.",
            {"status": "running"},
        )


def test_storage_round_trip(tmp_path):
    storage = ResearchStorage(tmp_path / "research.sqlite3")
    run = storage.create_run("test query", {"search_api": "none"})

    storage.save_report(run["id"], "final report")
    storage.append_message(run["id"], "assistant", "final report")
    storage.replace_notes(run["id"], ["note"], ["raw note"], "brief")
    storage.replace_sources(
        run["id"],
        [{"title": "Source", "url": "https://example.com", "snippet": "Summary"}],
    )
    storage.update_run_status(run["id"], "completed", completed=True)
    storage.append_event(run["id"], "completed", "Done.", {"status": "completed"})

    detail = storage.get_run_detail(run["id"])
    assert detail["status"] == "completed"
    assert detail["report"] == "final report"
    assert len(detail["messages"]) == 2
    assert [note["kind"] for note in detail["notes"]] == [
        "research_brief",
        "note",
        "raw_note",
    ]
    assert detail["sources"][0]["url"] == "https://example.com"
    assert storage.list_events(run["id"])[-1]["event_type"] == "completed"


def test_reset_running_runs_records_events(tmp_path):
    storage = ResearchStorage(tmp_path / "research.sqlite3")
    run = storage.create_run("test query", {})

    count = storage.reset_running_runs("restart")

    assert count == 1
    assert storage.get_run(run["id"])["status"] == "failed"
    assert storage.list_events(run["id"])[-1]["event_type"] == "failed"


def test_tavily_source_extraction_shape(tmp_path):
    storage = ResearchStorage(tmp_path / "research.sqlite3")
    service = RunService(storage, DummySecrets())
    raw_notes = [
        """
--- SOURCE 1: LangGraph Docs ---
URL: https://langchain-ai.github.io/langgraph/

SUMMARY:
LangGraph documentation summary.

--------------------------------------------------------------------------------

--- SOURCE 2: LangChain Blog ---
URL: https://blog.langchain.dev/

SUMMARY:
Blog summary.
"""
    ]

    sources = service._extract_sources(raw_notes)

    assert sources == [
        {
            "title": "LangGraph Docs",
            "url": "https://langchain-ai.github.io/langgraph/",
            "snippet": "SUMMARY:\nLangGraph documentation summary.",
        },
        {
            "title": "LangChain Blog",
            "url": "https://blog.langchain.dev/",
            "snippet": "SUMMARY:\nBlog summary.",
        },
    ]


def test_api_health_settings_runs_and_events(monkeypatch, tmp_path):
    import open_deep_research.api.app as api_app

    storage = ResearchStorage(tmp_path / "research.sqlite3")
    secrets = DummySecrets()
    run_service = NoopRunService(storage, secrets)
    monkeypatch.setattr(api_app, "storage", storage)
    monkeypatch.setattr(api_app, "secrets", secrets)
    monkeypatch.setattr(api_app, "run_service", run_service)

    assert asyncio.run(api_app.health()) == {"status": "ok"}
    assert asyncio.run(api_app.get_settings()) == {
        "settings": {},
        "api_keys": [],
    }

    run = asyncio.run(
        api_app.create_run(
            RunCreateRequest(query="test query", settings={"search_api": "none"}),
            BackgroundTasks(),
        )
    )
    assert run["status"] == "running"
    asyncio.run(run_service.execute_run(run["id"]))

    runs = asyncio.run(api_app.list_runs())
    assert len(runs) == 1

    detail = asyncio.run(api_app.get_run(run["id"]))
    assert detail["messages"][0]["content"] == "test query"

    events = asyncio.run(api_app.get_run_events(run["id"]))
    assert [event["event_type"] for event in events] == [
        "created",
        "test_noop",
    ]


def test_api_chat_rejects_unfinished_run(monkeypatch, tmp_path):
    import open_deep_research.api.app as api_app

    storage = ResearchStorage(tmp_path / "research.sqlite3")
    secrets = DummySecrets()
    run_service = NoopRunService(storage, secrets)
    monkeypatch.setattr(api_app, "storage", storage)
    monkeypatch.setattr(api_app, "secrets", secrets)
    monkeypatch.setattr(api_app, "run_service", run_service)

    run = asyncio.run(
        api_app.create_run(
            RunCreateRequest(query="test query"),
            BackgroundTasks(),
        )
    )
    try:
        asyncio.run(
            api_app.chat_about_report(run["id"], ChatRequest(message="summarize"))
        )
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "This run does not have a final report yet."
    else:
        raise AssertionError("Expected unfinished report chat to be rejected.")
