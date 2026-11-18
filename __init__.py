"""Agent Inbox for Hermes Agent: messages to your agent on Telegram, Slack or email reach Hermes's gateway within
seconds, and Hermes answers in the same conversation. Its tools ask the operator, write to people, and read the
conversations and contacts."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Optional

from . import words
from .settings import (ENV_HOME_CHANNEL, ENV_SECRET_KEY, OPERATOR_HOME, PLATFORM_NAME, SETUP_COMMAND, read_settings,
                       secret)

logger = logging.getLogger(__name__)


def check_requirements() -> bool:
    """Passive probe: the key is there, and httpx (which Hermes itself ships) imports. Never installs anything."""
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return bool(secret(ENV_SECRET_KEY))


def _validate_config(config: Any) -> bool:
    return read_settings(getattr(config, "extra", None) or {}).configured


def _env_enablement() -> Optional[dict]:
    """An env-only setup (what `hermes agentinbox setup` writes) shows in `hermes gateway status` as it is."""
    settings = read_settings()
    if not settings.configured:
        return None
    return {"agent": settings.agent, "api_url": settings.api_url}


def _home_is_the_operator(config: Any) -> None:
    """Agent Inbox's home channel is the operator, unless this profile picked one (`/sethome` in a conversation, or
    AGENTINBOX_HOME_CHANNEL). Hermes's scheduled jobs and its messages for Agent Inbox then reach the operator on
    their own channel, and a new conversation doesn't start with Hermes asking for a home channel. Being the home
    only by default, the operator isn't told each time the gateway restarts."""
    from gateway.config import HomeChannel, Platform, PlatformConfig  # type: ignore

    platform = Platform(PLATFORM_NAME)
    platform_config = config.platforms.setdefault(platform, PlatformConfig())
    if platform_config.home_channel is not None:
        return
    chat_id = secret(ENV_HOME_CHANNEL) or OPERATOR_HOME
    platform_config.home_channel = HomeChannel(platform=platform, chat_id=chat_id, name=words.HOME_NAME)
    if chat_id == OPERATOR_HOME:
        platform_config.gateway_restart_notification = False


def _build_adapter(config: Any) -> Any:
    from .adapter import AgentInboxAdapter

    return AgentInboxAdapter(config)


def register(ctx: Any) -> None:
    """Plugin entry point, called by Hermes's plugin system."""
    kwargs: dict = dict(
        name=PLATFORM_NAME,
        label="Agent Inbox",
        adapter_factory=_build_adapter,
        check_fn=check_requirements,
        validate_config=_validate_config,
        is_connected=_validate_config,
        required_env=[ENV_SECRET_KEY],
        install_hint=f"Run: {SETUP_COMMAND}",
        env_enablement_fn=_env_enablement,
        max_message_length=words.CARD_TEXT_MAX,
        pii_safe=True,
        emoji="📥",
        allow_update_command=False,
        platform_hint=words.PLATFORM_HINT,
    )
    try:
        from gateway.platform_registry import PlatformEntry  # type: ignore

        known = {item.name for item in dataclasses.fields(PlatformEntry)}
    except Exception:  # noqa: BLE001
        known = set()
    # Only Hermes's answer reaches the conversation, not its tool progress. Older Hermes refuses fields it doesn't know.
    if "display_tier" in known:
        kwargs["display_tier"] = "minimal"
    if "env_overrides" in known:
        kwargs["env_overrides"] = (_home_is_the_operator,)
    if "cron_deliver_env_var" in known:
        kwargs["cron_deliver_env_var"] = ENV_HOME_CHANNEL
    ctx.register_platform(**kwargs)

    from .tools import register_tools

    register_tools(ctx)

    if hasattr(ctx, "register_cli_command"):
        from .cli import handle_cli, setup_argparse

        ctx.register_cli_command(
            name=PLATFORM_NAME,
            help="Agent Inbox: `setup` connects this profile with a one-time code, `status` shows where it listens",
            setup_fn=setup_argparse,
            handler_fn=handle_cli,
        )
