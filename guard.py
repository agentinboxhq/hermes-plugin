"""What in a tool call reaches what stays closed in a contact's chat, whatever the operator lets contacts use: the
keys this computer keeps (the CLI's `~/.agentinbox`, any `.env`) and Hermes's own files (its home, with this profile's
`.env`, its sessions, memories and this plugin's state). What else a contact's chat may use is the operator's list
(`contacts_can_use`), which the hook goes by. Neither holds in the direct chat of a contact the operator gave full
access: everything is open there, as in the operator's own.

Commands and paths are matched as written: a script set up to get past it can (code that builds the path, say). What it
stops is Hermes doing what a contact asks for in the plain ways; the docs say a contact who may run commands may run
any."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional

MCP_PREFIX = "mcp__"

# Hermes's tools that run commands or code, and the arguments that carry them.
COMMAND_TOOLS = {"terminal": ("command",), "process_manage": ("data",), "execute_code": ("code",)}

# Hermes's tools that read or write files, and the arguments that name the files.
FILE_TOOLS = {"read_file": ("path",), "write_file": ("path",), "patch": ("path",), "search_files": ("path",)}

# What stays closed: the CLI's folder with its saved key, any `.env`, and Hermes's home.
_CLOSED = re.compile(
    r"\.agentinbox\b"
    r"|(?:^|[\s'\"=:/\\~]|\$HOME)\.env\b"
    r"|(?:^|[\s'\"=:/\\~]|\$HOME)\.hermes\b",
    re.IGNORECASE)


def mcp_parts(tool_name: str) -> Optional[tuple]:
    """`mcp__<server>__<tool>` as (server, tool)."""
    if not tool_name.startswith(MCP_PREFIX):
        return None
    server, _, tool = tool_name[len(MCP_PREFIX):].rpartition("__")
    return (server, tool) if server and tool else None


def _strings(args: Mapping[str, Any], keys: Iterable[str]) -> List[str]:
    return [args[key] for key in keys if isinstance(args.get(key), str)]


def closed_use(tool_name: str, args: Mapping[str, Any], hermes_home: str = "") -> bool:
    """Whether this call reaches what stays closed in a contact's chat: a key, or Hermes's own files (`hermes_home`,
    wherever this profile keeps them)."""
    args = args if isinstance(args, Mapping) else {}
    keys = COMMAND_TOOLS.get(tool_name) or FILE_TOOLS.get(tool_name) or ()
    texts = _strings(args, keys)
    home = hermes_home.rstrip("/\\")
    return any(_CLOSED.search(text) or (home and home in text) for text in texts)
