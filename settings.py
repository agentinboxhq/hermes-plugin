"""Where the plugin finds its connection: this Hermes profile's env first, then ``platforms.agentinbox`` in config.yaml."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

PLATFORM_NAME = "agentinbox"

# The host this plugin starts, as Agent Inbox names it (the API's ListenHostId) and as people read it.
HOST_ID = "hermes"
HOST_NAME = "Hermes"

DEFAULT_API_URL = "https://www.agentinbox.md/api"

# The account's first agent, as the API names it. Requests leave it out, as the CLI does.
DEFAULT_AGENT = "human-relay"

ENV_SECRET_KEY = "AGENTINBOX_SECRET_KEY"
ENV_AGENT = "AGENTINBOX_AGENT"
ENV_API_URL = "AGENTINBOX_API_URL"
# Hermes's home channel on Agent Inbox, when this profile names one: a conversation, or `operator`.
ENV_HOME_CHANNEL = "AGENTINBOX_HOME_CHANNEL"

# The chat that stands for the operator, in a new conversation on their channel: Agent Inbox's home channel, unless
# this profile picked another (`/sethome` in a conversation, or AGENTINBOX_HOME_CHANNEL).
OPERATOR_HOME = "operator"

SETUP_COMMAND = "hermes agentinbox setup <agent> --code <code>"


def secret(name: str, default: str = "") -> str:
    """This profile's value: Hermes keeps each profile's `.env` apart, so never plain `os.environ` inside Hermes."""
    try:
        from gateway.platforms._shared import get_scoped_secret  # type: ignore

        value = get_scoped_secret(name, default)
    except Exception:  # noqa: BLE001 — outside Hermes (tests, scripts)
        value = os.getenv(name, default)
    return str(value or "").strip()


@dataclass(frozen=True)
class Settings:
    secret_key: str = ""
    agent: str = DEFAULT_AGENT
    api_url: str = DEFAULT_API_URL

    @property
    def configured(self) -> bool:
        return bool(self.secret_key)

    @property
    def connection(self) -> str:
        """The connection as a fingerprint, never the key itself. What the operator allowed, and where the wait for
        messages got to, belong to one connection: setting up anew starts both over."""
        digest = hashlib.sha256(self.secret_key.encode("utf-8")).hexdigest()[:16]
        return f"{self.api_url}|{self.agent}|{digest}"


def read_settings(extra: Optional[Mapping[str, Any]] = None) -> Settings:
    """The key only ever comes from the env (`hermes agentinbox setup` writes it there); the agent and the API may
    also come from config.yaml."""
    extra = dict(extra or {})
    agent = secret(ENV_AGENT) or str(extra.get("agent") or "").strip() or DEFAULT_AGENT
    api_url = secret(ENV_API_URL) or str(extra.get("api_url") or "").strip() or DEFAULT_API_URL
    return Settings(secret_key=secret(ENV_SECRET_KEY), agent=agent, api_url=api_url.rstrip("/"))


# The name that gives contacts this plugin's own tools, in `contacts_can_use`.
AGENT_INBOX_TOOLSET = "agentinbox"


def contacts_can_use(extra: Optional[Mapping[str, Any]] = None) -> Tuple[str, ...]:
    """What the operator lets contacts use in a contact's chat, besides talking with them, reading that conversation
    and resolving it: `platforms.agentinbox.contacts_can_use` in config.yaml. Hermes's toolsets by name (`web`,
    `terminal`, `file`, …), MCP servers by name, and `agentinbox` for this plugin's tools. Nothing by default. A list,
    or names split by commas. A contact the operator gave full access isn't held to it in their own direct chat:
    they have everything there."""
    given = dict(extra or {}).get("contacts_can_use")
    names = given.split(",") if isinstance(given, str) else given if isinstance(given, (list, tuple)) else []
    return tuple(dict.fromkeys(str(name).strip() for name in names if str(name or "").strip()))


def state_dir() -> Path:
    """This profile's folder for the plugin's own state."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home()) / PLATFORM_NAME
    except Exception:  # noqa: BLE001 — outside Hermes
        return Path.home() / ".hermes" / PLATFORM_NAME
