"""Hermes's tools for Agent Inbox: ask the operator, write to someone, and read the conversations and contacts. They act
as the agent with the key the plugin holds; the model never sees it.

In the gateway, the tools work with the adapter that listens: it knows the operator, whose message started each chat,
and which questions still wait, so an answer comes back into the chat that asked. Elsewhere (`hermes chat`, a one-shot
run) they call Agent Inbox on their own, and an answer reaches Hermes in the operator's conversation once the gateway
runs.

A contact's chat is for talking with them: a `pre_tool_call` hook lets through reading and resolving that same
conversation, asking the person and keeping a plan, and only what the operator lets contacts use besides
(`contacts_can_use`); keys and Hermes's own files stay closed whatever it says. Agent Inbox's actions belong to the
operator, so these tools refuse there unless the operator lets contacts use `agentinbox`.

A contact the operator gave full access is the exception, in their own direct chat: a session of theirs started at
full access has everything the operator's own chat has, while it is their conversation's current one. A message at
another level starts a fresh session, and the hook judges each call by the session it runs in, so a message that says
the switch is off takes full access away at once, from a run already going on too.

From another app's chat, writing to anyone but the operator asks the operator first: the hook hands the call to
Hermes's own approval gate. The tool sends nothing the hook didn't hand to the gate."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from . import guard, words
from .api import AgentInboxApi, ApiError
from .settings import (AGENT_INBOX_TOOLSET, ENV_SECRET_KEY, HOST_NAME, OPERATOR_HOME, PLATFORM_NAME, contacts_can_use,
                       read_settings, secret)

logger = logging.getLogger(__name__)

TOOLSET = "agentinbox"

ASK_OPERATOR = "agentinbox_ask_operator"
ASK_HERE = "agentinbox_ask_here"
SEND = "agentinbox_send"
CONVERSATIONS = "agentinbox_conversations"
READ = "agentinbox_read"
CONTACTS = "agentinbox_contacts"
RESOLVE = "agentinbox_resolve"

# This plugin's tools; in a contact's chat, only asking in that conversation, reading it and resolving it stay on.
PLUGIN_TOOLS = (ASK_OPERATOR, ASK_HERE, SEND, CONVERSATIONS, READ, CONTACTS, RESOLVE)

# What a card asks for, in the words of the CLI and the MCP server: `tell` nothing, `ask` words, `approve` Approve or
# Deny, `choose` one of a few options.
KINDS = ("tell", "ask", "approve", "choose")

# How long agentinbox_ask_here waits for the answer within its call: well inside the 7 minutes Hermes gives a tool
# call by default. A later answer comes into the chat as a message.
ASK_HERE_WAIT_SECONDS = 5 * 60

# Hermes's toolsets a contact's chat always has: asking the person something, and keeping a plan.
CONTACT_TOOLSETS = ("clarify", "todo")

# How long a send the hook handed to the approval gate may still run: longer than any approval waits.
CLEARANCE_SECONDS = 30 * 60

# The longest one tool call waits for Agent Inbox.
CALL_TIMEOUT_SECONDS = 60.0

# Sessions on the operator's own computer: whoever is at it owns this Hermes.
LOCAL_SURFACES = ("cli", "tui", "desktop", "acp")

# What `to` may say for the operator.
OPERATOR_WORDS = ("operator", "my operator", "the operator", "your operator", "me", "owner")

# How many contacts a name is looked up among: the API's page, three times.
CONTACT_PAGES = 3

# An option's label on a card, at most (the API's limit).
OPTION_MAX = 100

# The adapters running in this process, by connection: a gateway that serves several profiles has one each.
_adapters: Dict[str, Any] = {}
_lock = threading.Lock()

# What Hermes gave the plugin at registration, for its documented session helpers.
_context: Any = None

# The sends the hook handed to Hermes's approval gate, by call, with when: such a send runs only once the gate allowed
# it, so finding it here tells the tool the operator was asked.
_cleared: Dict[Tuple[str, str, str, str], float] = {}


class ToolError(Exception):
    """A refusal or a failure, as the model reads it in the tool's result."""


def bind(adapter: Any) -> None:
    """The adapter listens now: tools in this profile's sessions go through it."""
    with _lock:
        _adapters[adapter.settings.connection] = adapter


def unbind(adapter: Any) -> None:
    with _lock:
        if _adapters.get(adapter.settings.connection) is adapter:
            del _adapters[adapter.settings.connection]


def _bound() -> Optional[Any]:
    connection = read_settings().connection
    with _lock:
        return _adapters.get(connection)


def configured() -> bool:
    """The tools show only to a profile that `hermes agentinbox setup` connected."""
    return bool(secret(ENV_SECRET_KEY))


# --- where a tool runs ---


def _session_value(name: str) -> str:
    """The calling session's chat, as Hermes's own tools read it. No documented helper says which chat a tool runs
    for; without this one, the tools take it for an unknown chat, and ask the operator more, never less."""
    try:
        from gateway.session_context import get_session_env  # type: ignore

        return str(get_session_env(name, "") or "")
    except Exception:  # noqa: BLE001 — outside Hermes
        return os.getenv(name, "")


def _surface() -> str:
    if _context is not None and hasattr(_context, "session_surface"):
        try:
            return str(_context.session_surface() or "").lower()
        except Exception:  # noqa: BLE001
            pass
    return _session_value("HERMES_SESSION_PLATFORM").lower()


def _in_cron() -> bool:
    if _context is not None and hasattr(_context, "current_cron_execution"):
        try:
            if _context.current_cron_execution() is not None:
                return True
        except Exception:  # noqa: BLE001
            pass
    return _session_value("HERMES_CRON_SESSION") == "1"


@dataclass(frozen=True)
class Here:
    """The session a tool runs for: its surface (`agentinbox`, `telegram`, `cli`, …) and, for an Agent Inbox chat,
    which conversation and round it is."""

    surface: str = ""
    cron: bool = False
    session_key: str = ""
    chat_id: str = ""
    chat_type: str = "dm"
    thread_id: str = ""
    user_id: str = ""
    user_name: str = ""
    chat_name: str = ""
    # The call, as Hermes names it to both the hook and the tool.
    session_id: str = ""
    task_id: str = ""

    @property
    def in_agent_inbox(self) -> bool:
        return self.surface == PLATFORM_NAME and bool(self.chat_id)

    @property
    def direct(self) -> bool:
        return self.chat_type == "dm"

    def chat(self, starter: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """The chat an answer goes back to, as the adapter builds its session's source again."""
        return {"chat_id": self.chat_id, "chat_type": self.chat_type, "thread_id": self.thread_id or None,
                "user_id": self.user_id or None, "user_name": self.user_name or None,
                "chat_name": self.chat_name or self.chat_id, "starter": _starter_kept(starter)}


def here() -> Here:
    return Here(surface=_surface(), cron=_in_cron(), session_key=_session_value("HERMES_SESSION_KEY"),
                chat_id=_session_value("HERMES_SESSION_CHAT_ID"),
                chat_type=_session_value("HERMES_SESSION_CHAT_TYPE") or "dm",
                thread_id=_session_value("HERMES_SESSION_THREAD_ID"),
                user_id=_session_value("HERMES_SESSION_USER_ID"),
                user_name=_session_value("HERMES_SESSION_USER_NAME"),
                chat_name=_session_value("HERMES_SESSION_CHAT_NAME"))


def _starter_kept(message: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """What the plugin keeps of the message that started a chat: enough to say whose run it is after a restart. The
    level its session runs at is kept with the conversation, and in its thread."""
    if not message:
        return None
    sender = message.get("from") or {}
    return {"from": {"id": sender.get("id"), "name": sender.get("name"), "isOperator": bool(sender.get("isOperator"))},
            "channel": message.get("channel"), "text": words.clean(message.get("text"))[:200],
            "isDirectMessage": message.get("isDirectMessage") is not False}


def _starter(adapter: Optional[Any], place: Here) -> Optional[Mapping[str, Any]]:
    if adapter is None or not place.in_agent_inbox:
        return None
    return adapter.starter_of(place.chat_id)


def _operator_of(adapter: Optional[Any]) -> str:
    return str(getattr(getattr(adapter, "listener", None), "operator_id", None) or "")


def _latest_in(adapter: Optional[Any], chat_id: str) -> Optional[Mapping[str, Any]]:
    """The latest message in an Agent Inbox chat, commands too, as the listening adapter keeps it: who wrote it, and
    whether it is a group. None outside the gateway, or before any came."""
    return adapter.writer_of(chat_id) if adapter is not None and chat_id else None


def _operators_own(adapter: Optional[Any], chat_id: str, chat_type: str, user_id: str) -> bool:
    """Whether an Agent Inbox chat is the operator's own direct chat: never a group, where a contact may be too, and
    never a chat it can't tell."""
    if chat_id == OPERATOR_HOME:
        return True
    if not chat_id or chat_type != "dm":
        return False
    latest = _latest_in(adapter, chat_id)
    if latest is not None and latest.get("isDirectMessage") is False:
        return False
    operator = _operator_of(adapter)
    if operator and user_id:
        return user_id == operator
    sender = (latest or {}).get("from") or {}
    return bool(sender.get("isOperator")) and (not user_id or str(sender.get("id") or "") == user_id)


def _full_access(adapter: Optional[Any], chat_id: str, chat_type: str, user_id: str, thread_id: str) -> bool:
    """Whether a session runs at the full access of the contact whose direct chat it is: started at full access, as
    its thread says, and still their conversation's current session. Their next message without it starts a fresh
    session and takes it away from this one at once. Never a group, nor a chat it can't tell."""
    if adapter is None or not chat_id or chat_type != "dm":
        return False
    full_access = adapter.full_access_session(chat_id, thread_id)
    return full_access is not None and (not user_id or str(full_access.get("person") or "") == user_id)


def _unlimited(adapter: Optional[Any], chat_id: str, chat_type: str, user_id: str, thread_id: str = "") -> bool:
    """Whether an Agent Inbox session has everything Hermes has, with nothing kept back for contacts: the operator's
    own direct chat, or a session at the full access of a contact the operator gave it."""
    return _operators_own(adapter, chat_id, chat_type, user_id) or \
        _full_access(adapter, chat_id, chat_type, user_id, thread_id)


def _job_origin() -> Optional[Mapping[str, Any]]:
    """The chat the scheduled job running now was set up from, as Hermes keeps it; None when it has none."""
    if _context is None or not hasattr(_context, "current_cron_execution"):
        return None
    job_id = getattr(_context.current_cron_execution(), "job_id", None)
    if not job_id:
        return None
    from cron.jobs import get_job  # type: ignore

    origin = (get_job(str(job_id)) or {}).get("origin")
    return origin if isinstance(origin, Mapping) else None


def _for_a_contact(adapter: Optional[Any], place: Here) -> bool:
    """Whether this call is for a contact, where Agent Inbox is off: an Agent Inbox chat other than the operator's own
    direct chat and the current session at a contact's full access (a group counts, even when the operator writes in
    it), or a scheduled job set up from one. A job set up from a contact's chat counts even when they had full access:
    it carries no message of theirs, so nothing says the switch is still on. Hermes's other apps, the terminal and the
    operator's own jobs aren't."""
    if place.cron:
        origin = _job_origin()
        if not origin or str(origin.get("platform") or "").lower() != PLATFORM_NAME:
            return False
        return not _operators_own(adapter, str(origin.get("chat_id") or ""), "dm", str(origin.get("user_id") or ""))
    agent_inbox = place.surface == PLATFORM_NAME or (
        not place.surface and _latest_in(adapter, place.chat_id) is not None)
    if not agent_inbox:
        return False
    return not _unlimited(adapter, place.chat_id, place.chat_type, place.user_id, place.thread_id)


def _refuse_for_a_contact(adapter: Optional[Any], place: Here) -> None:
    """The tools refuse on their own as well, should a call reach them without the hook. A check that fails refuses
    too."""
    try:
        for_a_contact = _for_a_contact(adapter, place)
    except Exception:  # noqa: BLE001
        logger.exception("[Agent Inbox] Couldn't tell whose chat this is, so the tool refused.")
        raise ToolError(words.CHECK_FAILED_MESSAGE) from None
    if for_a_contact and AGENT_INBOX_TOOLSET not in _contacts_can_use(adapter):
        raise ToolError(words.contact_blocked(job=place.cron))


def _operator_asks_first(adapter: Optional[Any], place: Here) -> bool:
    """Writing to a contact from another app's chat asks the operator first, through Hermes's own approvals there. The
    operator's own Agent Inbox chat, their terminal and their own jobs write without asking, and so does the chat of
    a contact with full access; any other contact's Agent Inbox chat can't write at all."""
    if place.cron or place.surface == PLATFORM_NAME:
        return False
    return bool(place.surface) and place.surface not in LOCAL_SURFACES


# --- running the work ---

Work = Callable[[AgentInboxApi, Optional[Any]], Awaitable[Any]]


def _run(work: Work) -> Any:
    """Runs `work` on the listening adapter's loop, with its client and what it knows; outside the gateway, with a
    client of its own for this one call."""
    adapter = _bound()
    loop = getattr(adapter, "loop", None) if adapter is not None else None
    if loop is not None and loop.is_running():
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            raise ToolError("Agent Inbox's tools can't run on the gateway's own event loop.")
        return asyncio.run_coroutine_threadsafe(work(adapter.api, adapter), loop).result(CALL_TIMEOUT_SECONDS)
    settings = read_settings()
    if not settings.configured:
        raise ToolError(words.NOT_SET_UP_MESSAGE)

    async def alone() -> Any:
        api = AgentInboxApi(settings.api_url, settings.secret_key, settings.agent)
        try:
            return await work(api, None)
        finally:
            await api.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, alone()).result(CALL_TIMEOUT_SECONDS + 5)


def _error(message: str) -> str:
    try:
        from tools.registry import tool_error  # type: ignore

        return tool_error(message)
    except Exception:  # noqa: BLE001 — outside Hermes
        return json.dumps({"error": message}, ensure_ascii=False)


def _failure(exc: ApiError) -> str:
    if exc.key_rejected:
        return words.KEY_CHANGED_MESSAGE
    if exc.outage:
        return f"Agent Inbox couldn't be reached ({exc}). Try again in a moment."
    return str(exc)


def _here_for(context: Mapping[str, Any]) -> Here:
    return replace(here(), session_id=str(context.get("session_id") or ""), task_id=str(context.get("task_id") or ""))


def _tool(fn: Callable[[Dict[str, Any], Here], str]) -> Callable[..., str]:
    def handler(args: Dict[str, Any], **context: Any) -> str:
        try:
            return fn(args if isinstance(args, dict) else {}, _here_for(context))
        except ToolError as exc:
            return _error(str(exc))
        except ApiError as exc:
            return _error(_failure(exc))
        except concurrent.futures.TimeoutError:
            return _error("Agent Inbox didn't answer in time. Try again in a moment.")

    handler.__name__ = fn.__name__
    return handler


# --- reading arguments ---


def _text(value: Any, name: str) -> str:
    text = words.clean(value).strip()
    if not text:
        raise ToolError(f"`{name}` is required.")
    return text


def _options(value: Any) -> List[str]:
    """A plain list, or the shapes small models send for one: a comma-separated string, or objects with a label."""
    if isinstance(value, str):
        value = [part for line in value.splitlines() for part in line.split(",")]
    found: List[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, Mapping):
            item = item.get("label") or item.get("text") or item.get("item") or next(iter(item.values()), "")
        label = words.clean(item).strip()
        if label:
            found.append(label[:OPTION_MAX])
    return found


def _limit(value: Any, default: int, most: int) -> int:
    try:
        return max(1, min(most, int(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class Answer:
    """What the person may answer with: `tell` none, `ask` words, `approve` Approve or Deny, `choose` one option."""

    kind: str
    options: Dict[str, str] = field(default_factory=dict)  # option id -> its label
    approve_label: str = ""
    deny_label: str = ""


def _answer(args: Mapping[str, Any], *, default: str) -> Answer:
    """`kind`, `options`, `approve_label` and `deny_label`, as the CLI and the MCP server take them. Options without a
    kind, or with `ask` or `tell`, make it `choose`; labels without a kind make it `approve`."""
    options = _options(args.get("options"))
    labels = [words.clean(args.get(name)).strip()[:OPTION_MAX] for name in ("approve_label", "deny_label")]
    kind = str(args.get("kind") or "").strip().lower()
    if not kind:
        kind = "choose" if options else "approve" if any(labels) else default
    elif options and kind in ("tell", "ask"):
        kind = "choose"
    if kind not in KINDS:
        raise ToolError(f"`kind` is one of: {', '.join(KINDS)}.")
    if any(labels) and kind != "approve":
        raise ToolError("`approve_label` and `deny_label` go with `kind: approve`.")
    if kind == "choose":
        if not 2 <= len(options) <= 10:
            raise ToolError("`kind: choose` takes 2 to 10 `options`.")
        return Answer("choose", {f"opt_{index}": label for index, label in enumerate(options, start=1)})
    return Answer(kind, approve_label=labels[0], deny_label=labels[1])


def _card(text: str, answer: Answer, body: Optional[str] = None) -> Dict[str, Any]:
    card: Dict[str, Any] = {"title": words.clip_card(text)}
    if body:
        card["body"] = words.clip_card(body)
    if answer.kind == "approve":
        card.update(approveLabel=answer.approve_label or "Approve", denyLabel=answer.deny_label or "Deny")
    if answer.kind == "choose":
        card["options"] = [{"id": key, "label": label} for key, label in answer.options.items()]
    return card


# --- people ---


def _name_of(contact: Mapping[str, Any]) -> str:
    name = " ".join(words.clean(part).strip() for part in (contact.get("firstName"), contact.get("lastName"))
                    if words.clean(part).strip())
    return name or words.clean(contact.get("email")) or words.clean(contact.get("id"))


async def _operator(api: AgentInboxApi, adapter: Optional[Any]) -> str:
    operator = getattr(getattr(adapter, "listener", None), "operator_id", None) or await api.find_operator()
    if not operator:
        raise ToolError(words.NO_OPERATOR_MESSAGE)
    return str(operator)


async def _contacts(api: AgentInboxApi) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    after: Optional[str] = None
    for _ in range(CONTACT_PAGES):
        page, after = await api.list_contacts(limit=100, after=after)
        found.extend(page)
        if not after:
            break
    return found


async def _recipient(api: AgentInboxApi, adapter: Optional[Any], to: str) -> Dict[str, Any]:
    """`to` as the person it names: the operator, or one contact by id, email or name."""
    operator = await _operator(api, adapter)
    wanted = to.strip()
    if wanted.lower() in OPERATOR_WORDS or wanted == operator:
        return {"id": operator, "name": "your operator", "isOperator": True}
    contacts = await _contacts(api)
    lowered = wanted.lower()
    matches = [each for each in contacts if str(each.get("id") or "") == wanted]
    matches = matches or [each for each in contacts if lowered in (
        words.clean(each.get("email")).strip().lower(), _name_of(each).lower())]
    matches = matches or [each for each in contacts
                          if words.clean(each.get("firstName")).strip().lower() == lowered]
    if len(matches) > 1:
        listed = "; ".join(f"{_name_of(each)} ({each.get('id')})" for each in matches[:10])
        raise ToolError(f"Several contacts match “{wanted}”: {listed}. Write to one of them by their id.")
    if not matches:
        raise ToolError(f"No contact matches “{wanted}”. {CONTACTS} lists them; your operator adds new ones on "
                        "agentinbox.md.")
    contact = matches[0]
    is_operator = bool(contact.get("isOperator")) or contact.get("id") == operator
    return {"id": str(contact.get("id")), "name": "your operator" if is_operator else _name_of(contact),
            "isOperator": is_operator}


# --- asking and sending ---


def _expect(adapter: Optional[Any], sent: Mapping[str, Any], *, question: str, to: Mapping[str, Any],
            answer: Answer, place: Here) -> str:
    """Keeps the question until its answer comes, and says where the answer will reach Hermes."""
    threads = sent.get("threads") if isinstance(sent.get("threads"), list) else []
    conversation = str((threads[0] or {}).get("id") or "") if threads and isinstance(threads[0], dict) else ""
    name = "your operator" if to["isOperator"] else str(to["name"])
    if adapter is None:
        return f"Asked {name}. The answer reaches {HOST_NAME} in that Agent Inbox conversation while its gateway runs."
    chat = place.chat(_starter(adapter, place)) if place.in_agent_inbox else None
    adapter.expect_answer({
        "id": str(sent.get("id") or ""), "question": question, "to": to["id"], "toOperator": bool(to["isOperator"]),
        "toName": "Your operator" if to["isOperator"] else name, "kind": answer.kind, "options": answer.options,
        "conversation": conversation, "expiresAt": sent.get("expiresAt"), "chat": chat,
    })
    if chat:
        return (f"Asked {name}. Their answer comes to you in this chat when they give it. Tell the person here you'll "
                "get back to them, then end your turn.")
    return f"Asked {name}. The answer comes to {HOST_NAME} in that Agent Inbox conversation."


def ask_operator(args: Dict[str, Any], place: Here) -> str:
    question = _text(args.get("question"), "question")
    answer = _answer(args, default="ask")
    if answer.kind == "tell":
        raise ToolError(f"A question needs an answer; to tell your operator something, use {SEND}.")
    adapter = _bound()
    _refuse_for_a_contact(adapter, place)
    if place.in_agent_inbox and _operators_own(adapter, place.chat_id, place.chat_type, place.user_id):
        raise ToolError(f"You're talking with your operator in this chat: ask them here with {ASK_HERE} instead.")

    async def work(api: AgentInboxApi, adapter: Optional[Any]) -> str:
        operator = await _operator(api, adapter)
        body = words.ask_card_body(words.run_starter(_starter(adapter, place)))
        sent = await api.create_interaction({"kind": answer.kind, "card": _card(question, answer, body),
                                             "to": operator, "from": HOST_NAME,
                                             "ttlSeconds": words.TOOL_QUESTION_TTL_SECONDS})
        return _expect(adapter, sent, question=question, to={"id": operator, "name": "your operator",
                                                              "isOperator": True}, answer=answer, place=place)

    return _run(work)


def _asked_here(adapter: Any, place: Here) -> Dict[str, Any]:
    """Who answers a question in this conversation, as its answer names them. Only what the operator writes, or a
    contact in their own session at full access, is more than a request and data."""
    if _operators_own(adapter, place.chat_id, place.chat_type, place.user_id):
        return {"to": _operator_of(adapter), "toOperator": True, "toName": "Your operator"}
    if not place.direct:
        return {"to": "", "toOperator": False, "toName": "Someone in this conversation"}
    return {"to": place.user_id, "toOperator": False, "toName": place.user_name or "The contact",
            "trusted": _full_access(adapter, place.chat_id, place.chat_type, place.user_id, place.thread_id)}


def ask_here(args: Dict[str, Any], place: Here) -> str:
    """Asks the people in this Agent Inbox conversation, in it, and waits for their answer, as `reply <id> ask`,
    `approve` or `choose` does in the CLI. Past the wait, the question stays open, and its answer comes to the chat as a
    message."""
    question = _text(args.get("question"), "question")
    answer = _answer(args, default="ask")
    if answer.kind == "tell":
        raise ToolError("A question needs an answer; anything else, say in your answer here.")
    not_here = (f"{ASK_HERE} asks in the Agent Inbox conversation you're in, and this isn't one. To ask your operator, "
                f"use {ASK_OPERATOR}.")
    bound = _bound()
    if bound is None or not place.in_agent_inbox or place.chat_id == OPERATOR_HOME:
        raise ToolError(not_here)
    asked = _asked_here(bound, place)

    async def post(api: AgentInboxApi, adapter: Optional[Any]) -> Tuple[str, concurrent.futures.Future]:
        if adapter is None:
            raise ToolError(not_here)
        since = adapter.latest_id_in(place.chat_id)
        sent = await api.create_interaction({"kind": answer.kind, "card": _card(question, answer),
                                             "thread": place.chat_id, "ttlSeconds": words.TOOL_QUESTION_TTL_SECONDS})
        question_id = str(sent.get("id") or "")
        # Its answer comes to the chat, also after a restart, unless this call still waits for it.
        adapter.expect_answer({"id": question_id, "question": question, **asked, "kind": answer.kind,
                               "options": answer.options, "conversation": place.chat_id,
                               "expiresAt": sent.get("expiresAt"), "chat": place.chat(_starter(adapter, place))})
        return question_id, adapter.wait_here(question_id, place.chat_id, since)

    async def stop_waiting(api: AgentInboxApi, adapter: Optional[Any]) -> Optional[tuple]:
        return adapter.stop_waiting_here(question_id) if adapter is not None else None

    question_id, waiting = _run(post)
    try:
        outcome = waiting.result(timeout=ASK_HERE_WAIT_SECONDS)
    except (concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
        # An answer that came just now is still this call's.
        outcome = _run(stop_waiting)
    return words.NO_ANSWER_YET if outcome is None else words.here_answer(*outcome)


def _talking_with(place: Here, to: Mapping[str, Any]) -> bool:
    return place.in_agent_inbox and place.direct and to["id"] == place.user_id


def _clearance(place: Here, args: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (place.session_id or place.session_key, place.task_id, words.clean(args.get("to")).strip().lower(),
            " ".join(words.clean(args.get("text")).split()))


def _clear(place: Here, args: Mapping[str, Any]) -> None:
    now = time.monotonic()
    with _lock:
        for key, at in list(_cleared.items()):
            if now - at > CLEARANCE_SECONDS:
                del _cleared[key]
        _cleared[_clearance(place, args)] = now


def _was_cleared(place: Here, args: Mapping[str, Any]) -> bool:
    with _lock:
        at = _cleared.pop(_clearance(place, args), None)
    return at is not None and time.monotonic() - at <= CLEARANCE_SECONDS


def _contacts_can_use(adapter: Optional[Any]) -> tuple:
    """What the operator lets contacts use, from the platform's settings the gateway gave the adapter."""
    return contacts_can_use(getattr(getattr(adapter, "config", None), "extra", None) or {})


def _toolset_tools(name: str) -> List[str]:
    """The tools of one of Hermes's toolsets; none outside Hermes, or for a name that isn't one."""
    try:
        from toolsets import resolve_toolset  # type: ignore

        return list(resolve_toolset(name) or [])
    except Exception:  # noqa: BLE001
        return []


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return str(get_hermes_home())
    except Exception:  # noqa: BLE001 — outside Hermes
        return ""


def _server_name(name: str) -> str:
    """An MCP server's name as Hermes writes it in its tools' names."""
    bare = name[4:] if name.startswith("mcp-") else name
    return "".join(char if char.isalnum() else "_" for char in bare)


def _contact_may_use(tool_name: str, can_use: tuple) -> bool:
    """Whether a contact's chat has this tool of Hermes's: one of the toolsets it always has, or of those the operator
    lets contacts use, or a tool of an MCP server they name."""
    parts = guard.mcp_parts(tool_name)
    if parts is not None:
        return _server_name(parts[0]) in {_server_name(name) for name in can_use}
    return any(tool_name in _toolset_tools(name) for name in (*CONTACT_TOOLSETS, *can_use))


def _this_conversation(args: Mapping[str, Any], place: Here, *, by_default: bool) -> bool:
    """Whether a reading or resolving call names the conversation it runs in (or none, where that means this one)."""
    named = words.clean(args.get("conversation")).strip()
    return place.in_agent_inbox and (named == place.chat_id or (by_default and not named))


def _allowed_for_a_contact(tool_name: str, args: Mapping[str, Any], place: Here) -> bool:
    """In a contact's chat, only asking in that same conversation, reading it and resolving it."""
    if tool_name == ASK_HERE:
        return place.in_agent_inbox
    if tool_name == READ:
        return _this_conversation(args, place, by_default=False)
    if tool_name == RESOLVE:
        return _this_conversation(args, place, by_default=True)
    return False


def before_tool_call(tool_name: str = "", args: Any = None, **context: Any) -> Optional[Dict[str, Any]]:
    """Hermes's `pre_tool_call` hook, for every tool call. In a contact's chat it lets through only asking in that
    conversation, reading it and resolving it, a plan, and what the operator lets contacts use; keys and Hermes's
    own files stay closed there. A session at the full access of a contact isn't held back while it is their
    conversation's current one. From another app's chat, before agentinbox_send writes to
    anyone but the operator, it hands the call to Hermes's approval gate. Anything else goes on as it is. Hermes runs a
    tool whose hook failed, so a failure here blocks it instead."""
    try:
        return _before_tool_call(str(tool_name or ""), args if isinstance(args, dict) else {}, context)
    except Exception:  # noqa: BLE001
        logger.exception("[Agent Inbox] Couldn't check %s, so it is blocked.", tool_name)
        return {"action": "block", "message": words.CHECK_FAILED_MESSAGE}


def _before_tool_call(tool_name: str, args: Dict[str, Any], context: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    place = _here_for(context)
    adapter = _bound()
    if _for_a_contact(adapter, place):
        return _for_contacts_chat(tool_name, args, place, adapter)
    if tool_name == SEND:
        return _ask_before_writing(args, place, adapter)
    return None


def _for_contacts_chat(tool_name: str, args: Dict[str, Any], place: Here,
                       adapter: Optional[Any]) -> Optional[Dict[str, Any]]:
    """In a contact's chat: reading and resolving it, asking the person, a plan, and what the operator lets contacts
    use. Keys and Hermes's own files stay closed whatever the list says."""
    can_use = _contacts_can_use(adapter)
    where = place.chat_id or "a scheduled job"
    if tool_name in PLUGIN_TOOLS:
        if _allowed_for_a_contact(tool_name, args, place) or AGENT_INBOX_TOOLSET in can_use:
            return None
        logger.info("[Agent Inbox] Blocked %s for a contact in %s.", tool_name, where)
        return {"action": "block", "message": words.contact_blocked(job=place.cron)}
    if not _contact_may_use(tool_name, can_use):
        logger.info("[Agent Inbox] Blocked %s for a contact in %s: contacts can't use it.", tool_name, where)
        return {"action": "block", "message": words.contact_cannot_use(tool_name, job=place.cron)}
    if guard.closed_use(tool_name, args, _hermes_home()):
        logger.info("[Agent Inbox] Blocked %s for a contact in %s: it reaches a key or Hermes's own files.",
                    tool_name, where)
        return {"action": "block", "message": words.CLOSED_FOR_A_CONTACT}
    return None


def _ask_before_writing(args: Mapping[str, Any], place: Here, adapter: Optional[Any]) -> Optional[Dict[str, Any]]:
    """From another app's chat, writing to anyone but the operator goes to Hermes's approval gate, which asks there
    as it asks before commands, and runs the tool only once allowed."""
    if not _operator_asks_first(adapter, place):
        return None
    try:
        text = _text(args.get("text"), "text")
        to = _run(lambda api, adapter: _recipient(api, adapter, _text(args.get("to"), "to")))
    except (ToolError, ApiError, concurrent.futures.TimeoutError):
        return None  # the tool says what's wrong, and sends nothing
    if to["isOperator"] or _talking_with(place, to):
        return None
    _clear(place, args)
    shown = " ".join(text.split())
    return {"action": "approve",
            "message": f"write to {to['name']}: “{shown if len(shown) <= 200 else shown[:199] + '…'}”",
            # "Allow for this session" lets Hermes write to this person again in the session without asking.
            "rule_key": f"{SEND}:{to['id']}"}


def send(args: Dict[str, Any], place: Here) -> str:
    to_words = _text(args.get("to"), "to")
    text = _text(args.get("text"), "text")
    answer = _answer(args, default="tell")
    adapter = _bound()
    _refuse_for_a_contact(adapter, place)
    to = _run(lambda api, adapter: _recipient(api, adapter, to_words))
    if _talking_with(place, to):
        raise ToolError(f"You're talking with {to['name']} in this chat: your answer already goes to them, and "
                        f"{ASK_HERE} asks them something. Use {SEND} only to write to someone else.")
    if not to["isOperator"] and _operator_asks_first(adapter, place) and not _was_cleared(place, args):
        # The hook didn't hand this call to Hermes's approval gate, so the operator wasn't asked.
        raise ToolError("Your operator couldn't be asked to allow it, so nothing was sent.")

    async def work(api: AgentInboxApi, adapter: Optional[Any]) -> str:
        payload: Dict[str, Any] = {"kind": answer.kind, "card": _card(text, answer), "to": to["id"]}
        if answer.kind != "tell":
            payload["ttlSeconds"] = words.TOOL_QUESTION_TTL_SECONDS
        sent = await api.create_interaction(payload)
        if to["id"] in (sent.get("failedTo") or []):
            raise ToolError(f"It couldn't be delivered to {to['name']}: no channel of theirs took it.")
        if answer.kind == "tell":
            return f"Sent to {to['name']}."
        return _expect(adapter, sent, question=text, to=to, answer=answer, place=place)

    return _run(work)


# --- reading ---


def _ago(stamp: Any, now: Optional[datetime] = None) -> str:
    """"just now", "5 min ago", "3 h ago", "2 d ago"."""
    try:
        at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return ""
    seconds = ((now or datetime.now(timezone.utc)) - at).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"


def _one_line(text: Any, most: int) -> str:
    flat = " ".join(words.clean(text).split())
    return flat if len(flat) <= most else f"{flat[: most - 1].rstrip()}…"


def _people(thread: Mapping[str, Any]) -> str:
    people = [words.clean(person.get("name") or person.get("id")) + (" (stranger)" if person.get("kind") == "stranger"
                                                                     else "")
              for person in thread.get("people") or [] if isinstance(person, Mapping)]
    return ", ".join(person for person in people if person) or "nobody yet"


def _conversation_line(thread: Mapping[str, Any]) -> str:
    where = words.clean(thread.get("channel")) + ("" if thread.get("isDirectMessage", True) else ", a group")
    parts = [words.clean(thread.get("id")), _people(thread), where, words.clean(thread.get("status")) or "open"]
    unread = int(thread.get("unreadCount") or 0)
    if unread:
        parts.append(f"{unread} unread")
    last = thread.get("lastMessage") if isinstance(thread.get("lastMessage"), Mapping) else None
    line = " · ".join(part for part in parts if part)
    if last:
        who = "you" if last.get("from") == "agent" else ("Agent Inbox" if last.get("from") == "system" else "them")
        line += f" · {_ago(last.get('at'))}, {who}: “{_one_line(last.get('text'), 160)}”"
    return line


def conversations(args: Dict[str, Any], place: Here) -> str:
    _refuse_for_a_contact(_bound(), place)
    status = str(args.get("status") or "all").strip().lower()
    if status not in ("open", "resolved", "all"):
        raise ToolError("`status` is one of: open, resolved, all.")
    limit = _limit(args.get("limit"), 20, 50)
    after = words.clean(args.get("after")).strip() or None

    async def work(api: AgentInboxApi, _adapter: Optional[Any]) -> str:
        rows, more = await api.list_conversations(filter="unread" if args.get("unread_only") else "all",
                                                  status=status, limit=limit, after=after)
        if not rows:
            return "No more conversations." if after else "No conversations yet."
        lines = [_conversation_line(row) for row in rows]
        if more:
            lines.append(f"\nMore: call {CONVERSATIONS} with after: \"{more}\".")
        return "\n".join(lines)

    return _run(work)


def _message_line(message: Mapping[str, Any]) -> str:
    sender = message.get("from")
    if sender == "agent":
        who = "You"
    elif sender == "system":
        who = "Agent Inbox"
    else:
        who = words.clean(message.get("senderName")).strip() or "Someone"
        if message.get("senderKind") == "stranger":
            who += " (a stranger)"
    text = words.clean(message.get("text")).strip()
    files = [item for item in message.get("attachments") or [] if isinstance(item, Mapping)]
    if files:
        text += f" (with {'a file' if len(files) == 1 else f'{len(files)} files'})"
    interaction = message.get("interaction") if isinstance(message.get("interaction"), Mapping) else None
    if interaction and interaction.get("status"):
        text += f" [{words.clean(interaction.get('kind'))}: {words.clean(interaction.get('status'))}]"
    shown = text if len(text) <= 2000 else f"{text[:1999]}…"
    return f"[{_ago(message.get('at'))}] {who}: {shown}"


def read(args: Dict[str, Any], place: Here) -> str:
    conversation = _text(args.get("conversation"), "conversation")
    if not _this_conversation(args, place, by_default=False):
        _refuse_for_a_contact(_bound(), place)
    limit = _limit(args.get("limit"), 20, 100)
    before = words.clean(args.get("before")).strip() or None

    async def work(api: AgentInboxApi, _adapter: Optional[Any]) -> str:
        view = await api.get_conversation(conversation, limit=limit, before=before)
        thread = view.get("thread") if isinstance(view.get("thread"), Mapping) else {}
        messages = [item for item in view.get("messages") or [] if isinstance(item, Mapping)]
        where = words.clean(thread.get("channel")) + ("" if thread.get("isDirectMessage", True) else ", a group")
        head = " · ".join(part for part in (words.clean(thread.get("id")) or conversation, _people(thread), where,
                                            words.clean(thread.get("status"))) if part)
        lines = [head, ""] + ([_message_line(message) for message in messages] or ["No messages."])
        if view.get("hasMore") and messages:
            lines.append(f"\nOlder: call {READ} with before: \"{words.clean(messages[0].get('id'))}\".")
        return "\n".join(lines)

    return _run(work)


def resolve(args: Dict[str, Any], place: Here) -> str:
    named = words.clean(args.get("conversation")).strip()
    here_chat = place.chat_id if place.in_agent_inbox else ""
    target = named or here_chat
    if not target:
        raise ToolError(f"Name the conversation to resolve: its id, as {CONVERSATIONS} lists it.")
    adapter = _bound()
    if target == here_chat:
        async def mark(api: AgentInboxApi, listening: Optional[Any]) -> None:
            if listening is None:
                await api.resolve_conversation(here_chat)
            else:
                # Any message into it would open it again, your answer too: it is resolved once you're done here.
                await listening.resolve_after_reply(here_chat)

        _run(mark)
        return ("This conversation is resolved once your answer here goes out. Its next message, from anyone, opens it "
                "again, in a fresh session.")
    _refuse_for_a_contact(adapter, place)
    if _operator_asks_first(adapter, place):
        raise ToolError("Another conversation is resolved only from your operator's own Agent Inbox chat, the "
                        "terminal, or a scheduled job.")
    _run(lambda api, _adapter: api.resolve_conversation(target))
    return f"Resolved {target}. Its next message, from anyone, opens it again."


def _contact_line(contact: Mapping[str, Any], operator: Optional[str]) -> str:
    channels = [words.clean(channel.get("via")) + (" (default)" if channel.get("isDefault") else "")
                for channel in contact.get("channels") or [] if isinstance(channel, Mapping) and channel.get("via")]
    parts = [words.clean(contact.get("id")), _name_of(contact), words.clean(contact.get("email")),
             ", ".join(channels) or "no channel yet"]
    is_operator = bool(contact.get("isOperator")) or (operator is not None and contact.get("id") == operator)
    parts.append("your operator" if is_operator else f"listen to: {'on' if contact.get('listenTo') else 'off'}")
    return " · ".join(part for part in parts if part)


def contacts(args: Dict[str, Any], place: Here) -> str:
    _refuse_for_a_contact(_bound(), place)
    limit = _limit(args.get("limit"), 50, 100)
    after = words.clean(args.get("after")).strip() or None

    async def work(api: AgentInboxApi, adapter: Optional[Any]) -> str:
        rows, more = await api.list_contacts(limit=limit, after=after)
        if not rows:
            return "No more contacts." if after else "No contacts yet: your operator adds them on agentinbox.md."
        operator = getattr(getattr(adapter, "listener", None), "operator_id", None)
        lines = [_contact_line(row, operator) for row in rows]
        if more:
            lines.append(f"\nMore: call {CONTACTS} with after: \"{more}\".")
        return "\n".join(lines)

    return _run(work)


# --- what Hermes sees ---


def _schema(name: str, description: str, properties: Dict[str, Any],
            required: Optional[List[str]] = None) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return {"name": name, "description": description, "parameters": parameters}


def _asking(kinds: List[str], first: str) -> Dict[str, Any]:
    """`kind`, `options` and the Approve and Deny labels, as the CLI and the MCP server name them."""
    said = {"tell": "tell: just a message", "ask": "ask: they answer in words",
            "approve": "approve: Approve and Deny buttons", "choose": "choose: one of `options`"}
    return {
        "kind": {"type": "string", "enum": kinds,
                 "description": ". ".join(said[kind] + (" (default)" if kind == first else "") for kind in kinds) + "."},
        "options": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
                    "description": "With `kind: choose`: 2 to 10 short options, as buttons to pick one from."},
        "approve_label": {"type": "string", "description": "With `kind: approve`: the Approve button's words, e.g. "
                                                           "“Ship it”. Default Approve."},
        "deny_label": {"type": "string", "description": "With `kind: approve`: the Deny button's words, e.g. “Hold”. "
                                                        "Default Deny."},
    }


_ASK_DESCRIPTION = (
    "Ask your operator, the person you work for, through Agent Inbox, and get their answer back. Use it from a "
    "scheduled job, the terminal, another app's chat or the conversation of a contact with full access, before you "
    "do something only your operator can decide. It returns at once; when your operator answers, even hours later, "
    "the answer comes to you and you go on. Not in your operator's own Agent Inbox conversation: ask them there with "
    f"{ASK_HERE}. In any other contact's, Agent Inbox is off."
)

_ASK_HERE_DESCRIPTION = (
    "Ask the people in the Agent Inbox conversation you're in something, there, and wait for the answer: words "
    "(kind ask), Approve or Deny (kind approve), or one of a few options (kind choose). It waits up to 5 minutes; "
    "without an answer by then, the question stays open, and the answer comes to you in this chat when they give it. "
    f"Only in an Agent Inbox conversation; to ask your operator from anywhere else, use {ASK_OPERATOR}."
)

_SEND_DESCRIPTION = (
    "Write to your operator or to one of their contacts, in a new conversation on their usual channel. Your answer "
    f"in a chat already goes to the person there, and {ASK_HERE} asks them something, so use this only for someone "
    "else. With `kind`, it asks a question, and the answer comes back to you in the chat you asked from. Not in a "
    "contact's Agent Inbox conversation, where Agent Inbox is off, unless your operator gave them full access. From "
    "another app's chat, writing to anyone but your operator asks your operator first."
)

_RESOLVE_DESCRIPTION = (
    "Mark an Agent Inbox conversation finished, once what it was about is done and nothing is left to answer, as a "
    "support desk closes a ticket. Nothing is sent. The conversation you're in is marked once your answer here goes "
    "out; its next message, from anyone, opens it again and starts a fresh session. Another conversation, by its "
    "id, only in your operator's own chat or that of a contact with full access, at the terminal, or in a scheduled "
    "job."
)

_NOT_FOR_CONTACTS = (" Not in a contact's Agent Inbox conversation, where Agent Inbox is off, unless your operator "
                     "gave them full access.")

TOOLS: List[Tuple[Dict[str, Any], Callable[[Dict[str, Any], Here], str], str]] = [
    (_schema(ASK_OPERATOR, _ASK_DESCRIPTION, {
        "question": {"type": "string", "description": "What to ask, as your operator should read it: say who wants "
                                                      "what, e.g. “Ben asks for a 10% discount on his order. OK?”"},
        **_asking(["ask", "approve", "choose"], "ask"),
    }, ["question"]), ask_operator, "🙋"),
    (_schema(ASK_HERE, _ASK_HERE_DESCRIPTION, {
        "question": {"type": "string", "description": "What to ask, as the people here should read it, e.g. "
                                                      "“Deploy to staging?”"},
        **_asking(["ask", "approve", "choose"], "ask"),
    }, ["question"]), ask_here, "❓"),
    (_schema(SEND, _SEND_DESCRIPTION, {
        "to": {"type": "string", "description": "“operator”, or a contact's id, email or name, as "
                                                f"{CONTACTS} lists them."},
        "text": {"type": "string", "description": "The message, as they should read it."},
        **_asking(["tell", "ask", "approve", "choose"], "tell"),
    }, ["to", "text"]), send, "✉️"),
    (_schema(CONVERSATIONS, "List the agent's conversations, latest first: the id, who is in it, the channel, and the "
                            f"last message. {READ} shows one in full." + _NOT_FOR_CONTACTS, {
        "unread_only": {"type": "boolean", "description": "Only conversations with messages nobody read yet."},
        "status": {"type": "string", "enum": ["open", "resolved", "all"], "description": "Default all."},
        "limit": {"type": "integer", "description": "How many, 1 to 50 (default 20)."},
        "after": {"type": "string", "description": "The cursor from a previous call, for the next page."},
    }), conversations, "📥"),
    (_schema(READ, "Read one conversation's latest messages, oldest first, with who wrote each. It doesn't mark the "
                   "conversation read. In a contact's Agent Inbox conversation, only that same conversation, unless "
                   "your operator gave them full access.", {
        "conversation": {"type": "string", "description": f"The conversation's id, as {CONVERSATIONS} lists it."},
        "limit": {"type": "integer", "description": "How many messages, 1 to 100 (default 20)."},
        "before": {"type": "string", "description": "A message id, for the messages before it."},
    }, ["conversation"]), read, "📖"),
    (_schema(CONTACTS, f"List your operator's contacts: the id to use as `to` in {SEND}, the name, email and "
                       "channels, and whether you listen to them (whether their messages reach you)."
                       + _NOT_FOR_CONTACTS, {
        "limit": {"type": "integer", "description": "How many, 1 to 100 (default 50)."},
        "after": {"type": "string", "description": "The cursor from a previous call, for the next page."},
    }), contacts, "👥"),
    (_schema(RESOLVE, _RESOLVE_DESCRIPTION, {
        "conversation": {"type": "string", "description": f"Another conversation's id, as {CONVERSATIONS} lists it. "
                                                          "Leave it out for the conversation you're in."},
    }), resolve, "✅"),
]


def register_tools(ctx: Any) -> None:
    global _context
    if not hasattr(ctx, "register_tool") or not hasattr(ctx, "register_hook"):
        # Without the hook, nothing would keep Agent Inbox off in a contact's chat.
        return
    _context = ctx
    for schema, fn, emoji in TOOLS:
        ctx.register_tool(schema["name"], TOOLSET, schema, _tool(fn), check_fn=configured,
                          requires_env=[ENV_SECRET_KEY], description=schema["description"], emoji=emoji)
    ctx.register_hook("pre_tool_call", before_tool_call)
