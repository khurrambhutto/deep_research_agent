"""Provider API key handling for local backend execution."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:
    import keyring
    from keyring.errors import KeyringError
except ImportError:  # pragma: no cover - dependency is installed in app envs.
    keyring = None

    class KeyringError(Exception):
        """Fallback error when keyring is unavailable."""

from open_deep_research.storage import ResearchStorage


KEYRING_SERVICE = "open_deep_research"
SUPPORTED_KEY_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
}

load_dotenv()


class SecretStore:
    """Store API keys in OS keyring and keep references in SQLite."""

    def __init__(self, storage: ResearchStorage) -> None:
        self.storage = storage

    def save_api_keys(self, keys: dict[str, str]) -> list[dict[str, Any]]:
        saved: list[dict[str, Any]] = []
        for key_name, value in keys.items():
            normalized = key_name.upper()
            if normalized not in SUPPORTED_KEY_NAMES or not value:
                continue
            if keyring is None:
                raise RuntimeError(
                    "The keyring package is not installed. Install backend "
                    "dependencies or use environment variables for development."
                )
            try:
                keyring.set_password(KEYRING_SERVICE, normalized, value)
            except KeyringError as exc:
                raise RuntimeError(
                    "Unable to save API key in OS keyring. Configure a local "
                    "keyring backend or use environment variables for development."
                ) from exc
            self.storage.save_api_key_ref(normalized, KEYRING_SERVICE, normalized)
            saved.append({"provider": normalized, "stored": True})
        return saved

    def get_api_keys(self) -> dict[str, str]:
        """Load saved keys, falling back to environment variables for dev use."""
        keys: dict[str, str] = {}
        for ref in self.storage.list_api_key_refs():
            provider = ref["provider"]
            if keyring is None:
                value = None
            else:
                try:
                    value = keyring.get_password(
                        ref["keyring_service"], ref["keyring_username"]
                    )
                except KeyringError:
                    value = None
            if value:
                keys[provider] = value

        for provider in SUPPORTED_KEY_NAMES:
            if provider not in keys and os.getenv(provider):
                keys[provider] = os.environ[provider]
        return keys

    def list_available_keys(self) -> list[dict[str, Any]]:
        available = []
        saved = {ref["provider"]: ref for ref in self.storage.list_api_key_refs()}
        for provider in sorted(SUPPORTED_KEY_NAMES):
            available.append(
                {
                    "provider": provider,
                    "stored": provider in saved,
                    "from_env": bool(os.getenv(provider)),
                    "updated_at": saved.get(provider, {}).get("updated_at"),
                }
            )
        return available
