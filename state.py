"""What the plugin keeps between gateway runs, in this profile's folder: which computer it is, what the operator allowed,
where the wait for messages got to, and the level each contact's conversation has."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# How many handled and own ids are kept: far more than one wait ever brings back.
IDS_KEPT = 500

# How many of Hermes's own questions wait for their answers at once: the oldest go first.
WAITING_KEPT = 100

# How many conversations' levels are kept: one for each contact's conversation that had full access at some point.
LEVELS_KEPT = 500

# The API's rule for a computer id: 16–64 of `[A-Za-z0-9_-]`.
_COMPUTER_ID = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

# A conversation's sessions after its first: once it is resolved, its messages come with a later round, which starts
# a fresh Hermes session (a thread of its own in the session key).
ROUND_THREAD = "round-"

# The level a contact's session runs at: everything the operator's own chat has, or a contact's limits.
FULL = "full"
LIMITED = "limited"


def new_computer_id() -> str:
    return f"cmp_{secrets.token_urlsafe(18)}"


def new_level_token() -> str:
    """What tells one level's session from another's, in a conversation's threads: new at every change."""
    return secrets.token_hex(4)


def session_thread(round_number: int, level: str = LIMITED, since: str = "") -> Optional[str]:
    """The thread Hermes keys a conversation's session by: its round, once the conversation was resolved (`round-2`),
    and its level, once its contact had full access or lost it (`full-1a2b3c4d`, `limited-5e6f7a8b`). A new thread is
    a fresh session. None for a conversation's first session, as before."""
    parts = [f"{ROUND_THREAD}{round_number}"] if round_number else []
    if since:
        parts.append(f"{level}-{since}")
    return ".".join(parts) or None


def thread_level(thread_id: Optional[str]) -> Tuple[str, str]:
    """A session's level, and the token of its level, as its thread says: limited for a thread without one."""
    last = str(thread_id or "").rsplit(".", 1)[-1]
    for level in (FULL, LIMITED):
        if last.startswith(f"{level}-") and len(last) > len(level) + 1:
            return level, last[len(level) + 1:]
    return LIMITED, ""


def round_thread(thread_id: Optional[str]) -> Optional[str]:
    """A session's thread without its level: the conversation's first, limited, session in the same round."""
    first = str(thread_id or "").split(".", 1)[0]
    return first if first.startswith(ROUND_THREAD) else None


def round_of(thread_id: Optional[str]) -> int:
    """The round a session's thread is in: 0 for a conversation's first."""
    first = round_thread(thread_id)
    try:
        return int(first[len(ROUND_THREAD):]) if first else 0
    except ValueError:
        return 0


@dataclass
class State:
    # This gateway as a computer the lease names, kept for good: the API knows it again after a restart or a new key.
    computer_id: str = ""
    # The connection everything below belongs to (`Settings.connection`): connecting anew starts it over.
    connection: str = ""
    # When the operator allowed listening here. Nothing is taken or started without it.
    allowed_at: Optional[str] = None
    # The operator agreed to move listening here from another computer: the first lease takes it over.
    take_over: bool = False
    # The Allow question still waiting for its answer: `id`, `expiresAt`, and its `title`, since the same question is
    # only asked anew when it would read otherwise now.
    asked: Optional[Dict[str, Any]] = None
    # Where the next wait for messages goes on from.
    cursor: Optional[str] = None
    # The latest handled message ids, so none is handled twice.
    handled: List[str] = field(default_factory=list)
    # What the plugin sent lately (its questions and the replies it posted): an answer to one of its own questions
    # never starts Hermes.
    own: List[str] = field(default_factory=list)
    # The questions Hermes asked with its tools that still wait for an answer, by id: whom it asked, what, and the
    # chat whose session the answer goes back to.
    waiting: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # The level of each contact's direct conversation whose contact had full access at some point, by conversation:
    # `level` (full or limited), `since` (the token of its sessions' threads at that level), `thread` (the session its
    # latest message went to), and the contact, as `person` and `name`. A change of level starts a fresh session.
    levels: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def remember(self, name: str, item: str) -> None:
        items: List[str] = getattr(self, name)
        if item not in items:
            items.append(item)
            del items[:-IDS_KEPT]

    def wait_for(self, question: Dict[str, Any]) -> None:
        self.waiting[str(question["id"])] = question
        for stale in list(self.waiting)[:-WAITING_KEPT]:
            del self.waiting[stale]

    def set_level(self, conversation: str, record: Dict[str, Any]) -> None:
        """A conversation's level, as its latest message set it. Replaced where it stands, so a tool call checked at
        the same moment on another thread never finds it missing; past the limit, the ones kept longest go first, and
        their conversations start over at a contact's first, limited, session."""
        self.levels[conversation] = record
        for stale in list(self.levels)[:-LEVELS_KEPT]:
            del self.levels[stale]


class StateStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self, connection: str) -> State:
        """The state for `connection`. For another, a fresh one that keeps only which computer this is."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        computer_id = raw.get("computer_id")
        if not isinstance(computer_id, str) or not _COMPUTER_ID.match(computer_id):
            computer_id = new_computer_id()
        if raw.get("connection") != connection:
            return State(computer_id=computer_id, connection=connection)
        known = {item.name for item in fields(State)}
        state = State(**{key: value for key, value in raw.items() if key in known})
        state.computer_id = computer_id
        state.handled = [str(item) for item in state.handled if item][-IDS_KEPT:]
        state.own = [str(item) for item in state.own if item][-IDS_KEPT:]
        waiting = state.waiting if isinstance(state.waiting, dict) else {}
        state.waiting = {str(key): value for key, value in waiting.items() if isinstance(value, dict)}
        levels = state.levels if isinstance(state.levels, dict) else {}
        state.levels = {str(key): value for key, value in levels.items()
                        if isinstance(value, dict) and value.get("level") in (FULL, LIMITED)}
        return state

    def save(self, state: State) -> None:
        """Written aside and moved into place, so a crash never leaves half a file; only this user can read it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".state-", dir=str(self.path.parent))
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(asdict(state), file, indent=2)
                file.write("\n")
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


class MemoryStore(StateStore):
    """For tests: the same rules, kept in memory."""

    def __init__(self, initial: Optional[State] = None):
        super().__init__(Path("/nonexistent"))
        self.current: Optional[State] = initial

    def load(self, connection: str) -> State:
        if self.current is None or self.current.connection != connection:
            computer_id = self.current.computer_id if self.current else new_computer_id()
            return State(computer_id=computer_id, connection=connection)
        return State(**asdict(self.current))

    def save(self, state: State) -> None:
        self.current = State(**asdict(state))
