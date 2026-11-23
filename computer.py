"""This computer as the operator knows it, the way `npx @agentinbox/cli listen` names it."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import Mapping, Optional

# The API takes a computer name of 1–64 characters.
COMPUTER_NAME_MAX = 64

# What hosts put in the environment of the commands their agent runs (the CLI's `detectHost` markers): any of them
# in the environment this process started with means an agent runs this, not a person at a terminal. Hermes's own
# CLI adds `AI_AGENT` and `HERMES_AGENT` to its environment as it starts, for the commands it runs in turn, so in
# `hermes agentinbox setup` the current environment always has them: only the one it started with tells.
AGENT_MARKERS = (
    ("CODEX_THREAD_ID", None), ("CODEX_CI", "1"), ("CODEX_SANDBOX", None),
    ("CLAUDE_CODE_CHILD_SESSION", "1"), ("OPENCODE", "1"), ("OPENCODE_SESSION_ID", None),
    ("CURSOR_AGENT", None), ("CURSOR_SANDBOX", None),
    ("HERMES_AGENT", None), ("HERMES_SESSION_ID", None), ("OPENCLAW_SHELL", "exec"),
)


def computer_name() -> str:
    """Its name on macOS ("Ada’s MacBook Pro"), its host name elsewhere."""
    if sys.platform == "darwin":
        try:
            ran = subprocess.run(["scutil", "--get", "ComputerName"], capture_output=True, text=True, timeout=5)
            name = ran.stdout.strip() if ran.returncode == 0 else ""
            if name:
                return name[:COMPUTER_NAME_MAX]
        except (OSError, subprocess.SubprocessError):
            pass
    name = socket.gethostname()
    if name.lower().endswith(".local"):
        name = name[: -len(".local")]
    return (name or "this computer")[:COMPUTER_NAME_MAX]


def started_environment() -> Optional[Mapping[str, str]]:
    """The environment this process started with, as whoever ran it gave it. None when it can't be read."""
    try:
        import psutil  # type: ignore  # one of Hermes's own dependencies

        started = psutil.Process().environ()
    except Exception:  # noqa: BLE001
        return None
    # A process title written over it (setproctitle, on Linux) leaves no PATH: it then tells nothing.
    return started if started.get("PATH") else None


def agent_runs_this(env: Optional[Mapping[str, str]] = None) -> bool:
    """An agent's command, not a person typing: the operator is then asked on their channel instead. Without the
    environment this process started with, the current one decides, which inside Hermes always looks like an
    agent's: the operator is asked on their channel, the safe side."""
    if env is None:
        env = started_environment() or os.environ
    if str(env.get("AI_AGENT") or "").strip():
        return True
    for name, value in AGENT_MARKERS:
        found = str(env.get(name) or "").strip()
        if found and (value is None or found == value):
            return True
    return False


def person_at_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty() and not agent_runs_this()
