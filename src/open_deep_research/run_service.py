"""Application services for research runs and report chat."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from open_deep_research.configuration import Configuration

load_dotenv()

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.secrets import SecretStore
from open_deep_research.storage import ResearchStorage
from open_deep_research.utils import get_api_key_for_model


SOURCE_RE = re.compile(
    r"--- SOURCE (?P<num>\d+): (?P<title>.*?) ---\s*URL: (?P<url>\S+)\s*(?P<body>.*?)(?=\n\n--- SOURCE|\Z)",
    re.DOTALL,
)


class RunService:
    """Coordinate graph execution, persistence, settings, and chat."""

    def __init__(self, storage: ResearchStorage, secrets: SecretStore) -> None:
        self.storage = storage
        self.secrets = secrets

    def create_run(self, query: str, settings: dict[str, Any]) -> dict[str, Any]:
        merged_settings = self._merged_settings(settings)
        return self.storage.create_run(query, merged_settings)

    async def execute_run(self, run_id: str) -> None:
        try:
            run = self.storage.get_run(run_id)
            if not run:
                return

            config = self._runnable_config(run["settings"])
            os.environ["GET_API_KEYS_FROM_CONFIG"] = "true"
            self.storage.append_event(
                run_id,
                "started",
                "Research graph execution started.",
                {"status": "running"},
            )

            result = await deep_researcher.ainvoke(
                {"messages": [HumanMessage(content=run["query"])]},
                config,
            )
        except Exception as exc:
            self.storage.update_run_status(run_id, "failed", str(exc), completed=True)
            self.storage.append_event(
                run_id,
                "failed",
                f"Research failed: {exc}",
                {"status": "failed", "error": str(exc)},
            )
            self.storage.append_message(
                run_id,
                "assistant",
                f"Research failed: {exc}",
                mode="research",
            )
            return

        final_report = str(result.get("final_report") or "").strip()
        messages = result.get("messages", [])
        last_message_content = self._last_message_content(messages)
        notes = [str(note) for note in result.get("notes", []) if note]
        raw_notes = [str(note) for note in result.get("raw_notes", []) if note]
        research_brief = result.get("research_brief")

        self.storage.replace_notes(run_id, notes, raw_notes, research_brief)
        self.storage.replace_sources(run_id, self._extract_sources(raw_notes))
        self.storage.append_event(
            run_id,
            "artifacts_saved",
            "Research notes and sources saved.",
            {
                "notes": len(notes),
                "raw_notes": len(raw_notes),
                "sources": len(self._extract_sources(raw_notes)),
            },
        )

        if final_report:
            self.storage.save_report(run_id, final_report)
            self.storage.append_message(
                run_id,
                "assistant",
                final_report,
                mode="research",
            )
            self.storage.update_run_status(run_id, "completed", completed=True)
            self.storage.append_event(
                run_id,
                "completed",
                "Final report generated.",
                {"status": "completed"},
            )
            return

        if last_message_content:
            self.storage.append_message(
                run_id,
                "assistant",
                last_message_content,
                mode="research",
            )
            self.storage.update_run_status(run_id, "needs_clarification")
            self.storage.append_event(
                run_id,
                "needs_clarification",
                "Research needs clarification from the user.",
                {"status": "needs_clarification"},
            )
            return

        self.storage.update_run_status(
            run_id,
            "failed",
            "Research graph completed without a report or assistant message.",
            completed=True,
        )
        self.storage.append_event(
            run_id,
            "failed",
            "Research graph completed without a report or assistant message.",
            {"status": "failed"},
        )

    async def chat_about_report(
        self,
        run_id: str,
        message: str,
        model: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> str:
        run = self.storage.get_run_detail(run_id)
        if not run:
            raise ValueError("Research run not found.")
        if not run.get("report"):
            raise ValueError("This run does not have a final report yet.")

        merged_settings = self._merged_settings(settings or {})
        selected_model = (
            model
            or merged_settings.get("final_report_model")
            or Configuration().final_report_model
        )
        max_tokens = int(
            merged_settings.get("final_report_model_max_tokens")
            or Configuration().final_report_model_max_tokens
        )
        config = self._runnable_config(merged_settings)
        os.environ["GET_API_KEYS_FROM_CONFIG"] = "true"
        chat_model = init_chat_model(
            model=selected_model,
            max_tokens=max_tokens,
            api_key=get_api_key_for_model(selected_model, config),
            tags=["langsmith:nostream"],
        )

        prompt = self._report_chat_prompt(run)
        self.storage.append_message(run_id, "user", message, mode="report_chat")
        self.storage.append_event(
            run_id,
            "report_chat_user",
            "User asked a follow-up question about the report.",
            {"status": run["status"]},
        )
        response = await chat_model.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=message),
            ]
        )
        answer = str(response.content).strip()
        self.storage.append_message(run_id, "assistant", answer, mode="report_chat")
        self.storage.append_event(
            run_id,
            "report_chat_answer",
            "Assistant answered a report follow-up.",
            {"status": run["status"]},
        )
        return answer

    def start_research_followup(
        self,
        run_id: str,
        query: str,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = self.storage.get_run_detail(run_id)
        if not previous:
            raise ValueError("Research run not found.")
        context = (
            "Use this previous completed research as context for a new research "
            "follow-up. Fetch new/current sources as needed.\n\n"
            f"Previous query:\n{previous['query']}\n\n"
            f"Previous report:\n{previous.get('report') or ''}\n\n"
            f"Follow-up request:\n{query}"
        )
        return self.create_run(context, settings or {})

    def _merged_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        saved_settings = self.storage.get_settings()
        graph_settings = saved_settings.get("graph", {})
        if isinstance(graph_settings, dict):
            return {**graph_settings, **settings}
        return dict(settings)

    def _runnable_config(self, settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "configurable": {
                **settings,
                "apiKeys": self.secrets.get_api_keys(),
            }
        }

    def _extract_sources(self, raw_notes: list[str]) -> list[dict[str, str | None]]:
        sources: list[dict[str, str | None]] = []
        for raw_note in raw_notes:
            for match in SOURCE_RE.finditer(raw_note):
                body = self._clean_source_snippet(match.group("body"))
                sources.append(
                    {
                        "title": match.group("title").strip(),
                        "url": match.group("url").strip(),
                        "snippet": body[:1000] if body else None,
                    }
                )
        return sources

    def _clean_source_snippet(self, content: str) -> str:
        lines = []
        for line in content.strip().splitlines():
            stripped = line.strip()
            if stripped and set(stripped) == {"-"}:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _last_message_content(self, messages: list[Any]) -> str | None:
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        return str(content).strip() if content else None

    def _report_chat_prompt(self, run: dict[str, Any]) -> str:
        notes = "\n\n".join(
            f"[{note['kind']}]\n{note['content']}" for note in run.get("notes", [])
        )
        sources = "\n".join(
            f"- {source.get('title') or source['url']}: {source['url']}"
            for source in run.get("sources", [])
        )
        prior_chat = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in run.get("messages", [])
            if msg.get("mode") == "report_chat"
        )
        return (
            "You answer follow-up questions about a completed research report. "
            "Use only the saved report, notes, sources, and prior chat below. "
            "If the user asks for new or current information that is not in the "
            "saved context, say that they should use Research Follow-up.\n\n"
            f"Original query:\n{run['query']}\n\n"
            f"Final report:\n{run.get('report') or ''}\n\n"
            f"Notes:\n{notes or 'No notes saved.'}\n\n"
            f"Sources:\n{sources or 'No sources saved.'}\n\n"
            f"Prior report chat:\n{prior_chat or 'No prior report chat.'}"
        )
