"""API request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    """Request to start a new deep research run."""

    query: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """Request for lightweight post-report chat."""

    message: str = Field(min_length=1)
    model: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class ResearchFollowupRequest(BaseModel):
    """Request to start a full research follow-up run."""

    query: str = Field(min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingsUpdateRequest(BaseModel):
    """Request to update local backend settings."""

    settings: dict[str, Any] = Field(default_factory=dict)


class ApiKeysUpdateRequest(BaseModel):
    """Request to save provider API keys."""

    keys: dict[str, str] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Saved message."""

    id: int
    role: str
    content: str
    mode: str
    created_at: str


class NoteResponse(BaseModel):
    """Saved research note."""

    id: int
    kind: str
    content: str
    created_at: str


class SourceResponse(BaseModel):
    """Saved source record."""

    id: int
    title: str | None = None
    url: str
    snippet: str | None = None
    created_at: str


class RunEventResponse(BaseModel):
    """Saved run status/progress event."""

    id: int
    run_id: str
    event_type: str
    message: str
    payload: dict[str, Any]
    created_at: str


class RunSummaryResponse(BaseModel):
    """Research run summary."""

    id: str
    query: str
    status: str
    settings: dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: str | None = None
    error: str | None = None
    report: str | None = None


class RunDetailResponse(RunSummaryResponse):
    """Research run detail."""

    messages: list[MessageResponse] = Field(default_factory=list)
    notes: list[NoteResponse] = Field(default_factory=list)
    sources: list[SourceResponse] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Post-report chat answer."""

    run_id: str
    answer: str


class SettingsResponse(BaseModel):
    """Local settings and API key availability."""

    settings: dict[str, Any]
    api_keys: list[dict[str, Any]]


class ApiKeysUpdateResponse(BaseModel):
    """Saved API key refs."""

    saved: list[dict[str, Any]]
