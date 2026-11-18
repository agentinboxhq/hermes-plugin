"""Agent Inbox as a channel of Hermes's gateway. Each conversation is a chat, and a fresh one once it is resolved;
Hermes's answer is the reply in it; Hermes's approval prompts go to the operator as Allow and Deny buttons, and its
questions to the person in the conversation. A question Hermes asks with its tools gets its answer back in the chat
that asked, as a message there.

A contact the operator gave full access stands in for the operator in their own direct chat: Hermes has everything
there, their commands work, and the approval prompts of their runs go to them, in that conversation. The switch counts
message by message, as each of their messages says, and each level has sessions of its own: a message at another level
than the conversation's session starts a fresh one, so nothing from one level carries into the other."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

from . import tools as agent_tools
from . import words
from .api import AgentInboxApi, ApiError
from .computer import computer_name
from .listener import Listener, StopForGood, Timing
from .settings import HOST_NAME, OPERATOR_HOME, PLATFORM_NAME, contacts_can_use, read_settings, state_dir
from .state import FULL, LIMITED, StateStore, new_level_token, round_thread, session_thread, thread_level

logger = logging.getLogger(__name__)

# The id of the extra button on an approval card: Hermes's own name for allowing until the session ends. Hermes's
# "always" is never offered: it writes Hermes's settings, which the operator does at Hermes itself.
ALLOW_FOR_SESSION = "session"

# How long a question to a person waits for the answer.
QUESTION_TTL_SECONDS = 10 * 60

# How long a command Hermes confirms first (/new, /undo, /reload-mcp) waits: Hermes drops it after five minutes.
CONFIRM_TTL_SECONDS = 5 * 60

# How often an open card's answer is looked up, when it didn't come with the messages.
ANSWER_POLL_SECONDS = 5.0

# How often the answers to Hermes's own questions are looked up: a contact the agent doesn't listen to answers only
# where the plugin looks.
WAITING_POLL_SECONDS = 20.0

# What a contact may still type as a command in their own conversation: stopping the run there.
CONTACT_COMMANDS = ("/stop",)

# When Hermes resolves the conversation it works in, how long it must be quiet there first: any message into a
# conversation opens it again, Hermes's own answer too, and Hermes refreshes its typing every 2 seconds while it works.
RESOLVE_QUIET_SECONDS = 10.0


@dataclass
class Pending:
    """A card waiting for its answer: an approval Hermes waits on, or a question its `clarify` tool asked."""

    kind: str  # "approval", "clarify" or "confirm"
    session_key: str
    chat_id: str
    clarify_id: str = ""
    confirm_id: str = ""
    options: Dict[str, str] = field(default_factory=dict)  # option id -> its label
    # Whom an approval or a confirmation asks, as Hermes reads them in a sentence.
    asked: str = "your operator"


@dataclass(frozen=True)
class Approver:
    """Who allows what Hermes's run in a chat asks first: the operator, on their own channel; or a contact the operator
    gave full access, in their own conversation."""

    id: str
    # As Hermes reads it in a sentence: "your operator", or the contact's name.
    who: str = "your operator"
    # The contact's own conversation, where their cards go. None for the operator.
    thread: Optional[str] = None

    @property
    def address(self) -> Dict[str, str]:
        return {"to": self.id, **({"thread": self.thread} if self.thread else {})}


def _may_name_mcp_server(name: str) -> bool:
    """Whether a name the operator lets contacts use may be an MCP server's: anything that isn't one of Hermes's own
    toolsets, nor this plugin's."""
    if name == agent_tools.TOOLSET or name.startswith("mcp-"):
        return name.startswith("mcp-")
    try:
        from toolsets import TOOLSETS  # type: ignore

        return name not in TOOLSETS
    except Exception:  # noqa: BLE001 — outside Hermes
        return True


def _slash_confirm_resolver() -> Optional[Callable[..., Awaitable[Optional[str]]]]:
    """Hermes's registry of the commands waiting for a confirmation (/new, /undo, /reload-mcp), which its own button
    adapters answer through. None when this Hermes has none: its text form asks instead."""
    try:
        from tools import slash_confirm  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    return getattr(slash_confirm, "resolve", None)


def _approval_timeout() -> int:
    try:
        from gateway.platforms.base import approval_timeout_seconds  # type: ignore

        return int(approval_timeout_seconds())
    except Exception:  # noqa: BLE001 — older or newer Hermes: its documented default
        return 300


def _resolve_approval(session_key: str, choice: str, reason: Optional[str] = None) -> None:
    from tools.approval import resolve_gateway_approval  # type: ignore

    resolve_gateway_approval(session_key, choice, reason=reason)


def _resolve_clarify(clarify_id: str, response: str) -> None:
    from tools.clarify_gateway import resolve_gateway_clarify  # type: ignore

    resolve_gateway_clarify(clarify_id, response)


def _conversation_chat(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Where an answer goes when Hermes asked outside an Agent Inbox chat (a terminal, a scheduled job): the question's
    own conversation, the person asked in it."""
    conversation = str(question.get("conversation") or "")
    if not conversation:
        return None
    asked = {"id": question.get("to"), "name": question.get("toName"), "isOperator": bool(question.get("toOperator"))}
    return {"chat_id": conversation, "chat_type": "dm", "thread_id": None, "user_id": question.get("to") or None,
            "user_name": None, "chat_name": conversation, "starter": {"from": asked, "channel": "", "text": ""}}


def _multi_select(clarify_id: str) -> bool:
    try:
        from tools.clarify_gateway import get_pending  # type: ignore

        return bool(getattr(get_pending(clarify_id), "multi_select", False))
    except Exception:  # noqa: BLE001
        return False


class AgentInboxAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = words.CARD_TEXT_MAX

    def __init__(self, config: PlatformConfig, *, api: Optional[AgentInboxApi] = None,
                 store: Optional[StateStore] = None, timing: Optional[Timing] = None,
                 answer_poll_seconds: float = ANSWER_POLL_SECONDS,
                 waiting_poll_seconds: float = WAITING_POLL_SECONDS,
                 resolve_quiet_seconds: float = RESOLVE_QUIET_SECONDS):
        super().__init__(config, Platform(PLATFORM_NAME))
        # Hermes's own notices (no home channel, a helper that failed, a sign-in) go through `send_private_notice`,
        # which keeps them out of a contact's conversation.
        if not isinstance(config.extra, dict):
            config.extra = {}
        config.extra["notice_delivery"] = "private"
        self.settings = read_settings(config.extra or {})
        self.api = api or AgentInboxApi(self.settings.api_url, self.settings.secret_key, self.settings.agent)
        self.computer_name = computer_name()
        self.listener = Listener(api=self.api, store=store or StateStore(state_dir() / "state.json"),
                                 settings=self.settings, computer_name=self.computer_name,
                                 on_message=self._on_message, on_answer=self._on_answer, timing=timing)
        # The loop the adapter runs on, where Hermes's tools hand their calls.
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._answer_poll_seconds = answer_poll_seconds
        self._waiting_poll_seconds = waiting_poll_seconds
        self._waiting_polled_at = 0.0
        self._task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, Pending] = {}
        # Each conversation's latest message, which the cards of its run name: "For Ben's message on Telegram".
        self._latest: Dict[str, Dict[str, Any]] = {}
        # Each conversation's latest message, commands too: whose chat it is goes by it.
        self._writers: Dict[str, Dict[str, Any]] = {}
        # The conversations Hermes resolved while working in them, each resolved once Hermes is quiet there.
        self._resolve_quiet_seconds = resolve_quiet_seconds
        self._resolving: Dict[str, Optional[asyncio.Task]] = {}

    @property
    def authorization_is_upstream(self) -> bool:
        """Agent Inbox decides who may start Hermes: the operator, and the contacts the agent listens to ("Listen to"
        on agentinbox.md). A stranger's message never comes here."""
        return True

    def toolsets_for_source(self, source: Any) -> Optional[List[str]]:
        """The toolsets a turn in this chat has, as the gateway asks each adapter at every turn: as configured in the
        operator's own direct chat, and in the session of a contact the operator gave full access while it is their
        conversation's current one; in any other contact's chat or a group, this plugin's (which there only read and
        resolve it), asking the person, a plan, and only what the operator lets contacts use (`contacts_can_use`),
        with no MCP server unless it names one. Hermes adds a few of its own whatever this says, so the
        `pre_tool_call` hook is what holds."""
        chat_id = str(getattr(source, "chat_id", "") or "")
        chat_type = str(getattr(source, "chat_type", "") or "dm")
        user_id = str(getattr(source, "user_id", "") or "")
        thread_id = str(getattr(source, "thread_id", "") or "")
        if agent_tools._unlimited(self, chat_id, chat_type, user_id, thread_id):
            return None
        can_use = list(contacts_can_use(self.config.extra))
        toolsets = [agent_tools.TOOLSET, *agent_tools.CONTACT_TOOLSETS, *can_use]
        return toolsets if any(_may_name_mcp_server(name) for name in can_use) else [*toolsets, "no_mcp"]

    # --- lifecycle ---

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.settings.configured:
            logger.error("[Agent Inbox] %s", words.NOT_SET_UP_MESSAGE)
            self._set_fatal_error("agentinbox_not_set_up", words.NOT_SET_UP_MESSAGE, retryable=False)
            return False
        # One gateway per agent on this computer, whichever Hermes profile runs it.
        identity = f"{self.settings.api_url}|{self.settings.agent}"
        if not self._acquire_platform_lock(PLATFORM_NAME, identity, f"Agent Inbox agent {self.settings.agent}"):
            return False
        self._running = True
        self.loop = asyncio.get_running_loop()
        agent_tools.bind(self)
        self._task = asyncio.create_task(self._run())
        if self.listener.state.waiting:
            self._start_polling()
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        agent_tools.unbind(self)
        self._running = False
        # Hermes resolved these and is stopping: resolve them now, or they'd stay open.
        for conversation in list(self._resolving):
            await self._resolve_now(conversation)
        for task in (self._task, self._poll_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._task = self._poll_task = None
        for interaction_id in list(self._pending):
            self._pending.pop(interaction_id, None)
            await self._withdraw(interaction_id)
        await self.listener.close()
        await self.api.close()
        self._release_platform_lock()
        self._mark_disconnected()

    async def _run(self) -> None:
        try:
            await self.listener.run()
        except asyncio.CancelledError:
            raise
        except StopForGood as stop:
            logger.warning("[Agent Inbox] %s", stop.message)
            self._set_fatal_error(f"agentinbox_{stop.reason}", stop.message, retryable=False)
            await self._notify_fatal_error()
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Agent Inbox] Listening stopped.")
            self._set_fatal_error("agentinbox_error", f"Listening stopped: {exc}", retryable=True)
            await self._notify_fatal_error()

    # --- inbound ---

    async def _on_message(self, message: Dict[str, Any]) -> None:
        conversation = str(message["conversationId"])
        sender = message.get("from") or {}
        text = words.clean(message.get("text")).strip()
        first = text.split(maxsplit=1)[0].lower() if text else ""
        full_access = words.full_access(message)
        # The operator owns this Hermes: their commands work here as in any chat of theirs, and so do those of a
        # contact the operator gave full access, in their own direct chat. Any other contact may only stop the run in
        # their own conversation; anything else they type is a message, and can't answer a prompt.
        control = bool(sender.get("isOperator")) or full_access or first in CONTACT_COMMANDS
        command = control and text.startswith("/")
        self._writers[conversation] = message
        if not command:
            self._latest[conversation] = message
            # Someone wrote before Hermes's resolve went through: the conversation goes on.
            self._keep_open(conversation)
        group = message.get("isDirectMessage") is False
        source = self.build_source(
            chat_id=conversation,
            chat_name=words.clean(sender.get("name")) or conversation,
            chat_type="group" if group else "dm",
            user_id=str(sender.get("id") or "") or None,
            user_name=words.clean(sender.get("name")) or None,
            thread_id=await self._session_thread(conversation, message),
            message_id=str(message["id"]),
        )
        event = MessageEvent(
            text=text if command else words.message_for_host(message),
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(message["id"]),
            allow_gateway_control=control,
        )
        await self.handle_message(event)

    async def _on_answer(self, message: Dict[str, Any]) -> bool:
        """An answer to one of the plugin's open cards resolves what Hermes waits on, and starts nothing. One to a
        question Hermes asked with its tools goes to the chat that asked."""
        answers = str(message.get("answersMessageId") or "")
        answer = message.get("answer") or {}
        status = str(answer.get("status") or "answered")
        text = answer.get("text") or message.get("text")
        pending = self._pending.pop(answers, None)
        if pending is not None:
            await self._settle(pending, status, answer.get("optionId"), text)
            return True
        if answers in self.listener.state.waiting:
            await self._answered(answers, status, answer.get("optionId"), text)
            return True
        return False

    async def _settle(self, pending: Pending, status: str, option_id: Optional[str], text: Optional[str]) -> None:
        """A card's answer, for what waits on it."""
        if pending.kind != "confirm":
            self._resolve(pending, status, option_id, text)
            return
        # Expired or withdrawn: Hermes drops the command by itself. Words instead of a button don't run it either.
        if status not in ("approved", "denied", "answered"):
            return
        resolve = _slash_confirm_resolver()
        if resolve is None:
            return
        try:
            said = await resolve(pending.session_key, pending.confirm_id, "once" if status == "approved" else "cancel")
        except Exception:  # noqa: BLE001
            logger.exception("[Agent Inbox] Answering Hermes's confirmation failed.")
            return
        # What the command says when it's done, such as "/new cancelled"; the card itself shows the answer.
        if said:
            await self.send(pending.chat_id, said)

    def _resolve(self, pending: Pending, status: str, option_id: Optional[str], text: Optional[str]) -> None:
        if pending.kind == "approval":
            who = words.sentence_start(pending.asked)
            if status == "approved":
                _resolve_approval(pending.session_key, "session" if option_id == ALLOW_FOR_SESSION else "once")
            elif status == "denied":
                _resolve_approval(pending.session_key, "deny", f"{who} said no.")
            elif status == "answered" and text:
                # Words instead of a button: not an Allow, but what they wrote may say what to do instead.
                _resolve_approval(pending.session_key, "deny", (
                    f"{who} answered in words instead of allowing it: “{words.clip_card(words.clean(text))}”. "
                    "It isn't allowed; follow what they wrote if it says what to do instead."))
            # Expired or withdrawn: Hermes's own wait ends with the same timeout, as a Deny.
            return
        if status in ("answered", "approved"):
            response = pending.options.get(str(option_id or "")) or words.clean(text)
            if response:
                _resolve_clarify(pending.clarify_id, response)

    # --- outbound ---

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        if (metadata or {}).get("is_approval_prompt"):
            # The text form of an approval, which would go to whoever is in the conversation. Approvals go to the
            # operator as cards only.
            logger.warning("[Agent Inbox] Kept a text approval prompt out of %s.", chat_id)
            return SendResult(success=True)
        text = (content or "").strip()
        if not text:
            return SendResult(success=True)
        payload: Dict[str, Any] = {"kind": "tell", "card": {"title": words.clip_card(text)}}
        try:
            if chat_id == OPERATOR_HOME:
                # Agent Inbox's home channel: a scheduled job's result, or Hermes's message for Agent Inbox, goes to
                # the operator in a new conversation.
                operator = self.listener.operator_id or await self.api.find_operator()
                if not operator:
                    logger.error("[Agent Inbox] %s", words.NO_OPERATOR_TO_WRITE_TO)
                    return SendResult(success=False, error=words.NO_OPERATOR_TO_WRITE_TO)
                payload["to"] = str(operator)
            else:
                payload["thread"] = chat_id
            sent = await self.api.create_interaction(payload)
        except ApiError as exc:
            logger.error("[Agent Inbox] Couldn't write in %s: %s", chat_id, exc)
            if chat_id in self._resolving:
                # Its answer never got there: resolving now would leave the conversation resolved without one.
                self._keep_open(chat_id)
                logger.warning("[Agent Inbox] Left %s open, as Hermes's reply there didn't go out.", chat_id)
            return SendResult(success=False, error=str(exc), retryable=exc.outage)
        self._put_off_resolving(chat_id)
        return SendResult(success=True, message_id=str(sent.get("id") or ""))

    async def send_private_notice(self, chat_id: str, user_id: Optional[str], content: str,
                                  reply_to: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Hermes's own notices are for the operator: they show in the operator's own conversation, and in the session
        of a contact the operator gave full access, who stands in for them there; never in any other contact's, nor in
        one the operator shares with others."""
        latest = self.writer_of(chat_id) or {}
        sender = latest.get("from") or {}
        operator = self.listener.operator_id
        by_operator = bool(sender.get("isOperator")) or bool(operator and user_id and str(user_id) == str(operator))
        full_access = self.full_access_session(chat_id, (metadata or {}).get("thread_id"))
        by_full_access = full_access is not None and (not user_id or str(full_access.get("person")) == str(user_id))
        if chat_id == OPERATOR_HOME or (latest.get("isDirectMessage") is not False and (by_operator or by_full_access)):
            return await self.send(chat_id, content, reply_to=reply_to, metadata=metadata)
        logger.info("[Agent Inbox] Kept a Hermes notice out of %s: %s", chat_id, (content or "")[:200])
        return SendResult(success=True)

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        self._put_off_resolving(chat_id)
        if not self.listener.holding or chat_id == OPERATOR_HOME:
            return
        try:
            await self.api.show_typing(self.listener.computer_id, chat_id)
        except ApiError:
            pass  # typing is a courtesy: the reply still goes out

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if chat_id == OPERATOR_HOME:
            return {"name": words.HOME_NAME, "type": "dm", "chat_id": chat_id}
        latest = self._latest.get(chat_id) or {}
        kind = "group" if latest.get("isDirectMessage") is False else "dm"
        return {"name": chat_id, "type": kind, "chat_id": chat_id}

    async def _send_exec_approval_prompt(self, prompt: Any) -> SendResult:
        """Hermes wants to run a command its approval settings flag. The operator decides, on their own channel,
        whoever's message started the run; in the session of a contact the operator gave full access, that contact
        decides, in their conversation. Hermes names the session by its thread."""
        timeout = _approval_timeout()
        approver = self._approver(prompt.chat_id, (getattr(prompt, "metadata", None) or {}).get("thread_id"))
        if approver is None:
            _resolve_approval(prompt.session_key, "deny", "Your operator couldn't be asked, so it isn't allowed.")
            return SendResult(success=True)
        # A contact's own card is about their own message; the operator's says whose message the run is for.
        starter = "you" if approver.thread else words.run_starter(self._latest.get(prompt.chat_id))
        card: Optional[Tuple[str, str]] = words.permission_card(
            command=prompt.command, reason=prompt.description, computer_name=self.computer_name, starter=starter,
            timeout_seconds=timeout, smart_denied=bool(prompt.smart_denied))
        if card is None:
            _resolve_approval(prompt.session_key, "deny", words.too_long_to_ask(approver.who))
            return SendResult(success=True)
        title, body = card
        buttons: Dict[str, Any] = {"approveLabel": "Allow", "denyLabel": "Deny"}
        if ALLOW_FOR_SESSION in prompt.choices:
            buttons["extraActions"] = [{"id": ALLOW_FOR_SESSION, "label": "Allow for this session"}]
        try:
            asked = await self.api.create_interaction({
                "kind": "approve",
                "card": {"title": title, "body": body, **buttons},
                **approver.address,
                "from": HOST_NAME,
                # A late tap can't allow what Hermes already took as a Deny: the card expires with its wait.
                "ttlSeconds": max(60, timeout),
            })
        except ApiError as exc:
            # Never the text prompt instead: it would go to whoever is in the conversation, a contact too.
            logger.error("[Agent Inbox] Couldn't ask %s (%s): denied.", approver.who, exc)
            _resolve_approval(prompt.session_key, "deny",
                              f"{words.sentence_start(approver.who)} couldn't be asked, so it isn't allowed.")
            return SendResult(success=True)
        self._watch(str(asked["id"]), Pending(kind="approval", session_key=prompt.session_key, chat_id=prompt.chat_id,
                                              asked=approver.who))
        logger.info("[Agent Inbox] Asked %s to allow `%s`.", approver.who, words.clean(prompt.command)[:200])
        return SendResult(success=True, message_id=str(asked["id"]))

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
                           session_key: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """A question from Hermes to the person whose message started the run, in that conversation."""
        card = words.question_card(question, [str(choice) for choice in choices or []], _multi_select(clarify_id))
        options = {f"opt_{index}": label for index, label in enumerate(card.get("options") or [], start=1)}
        if card["kind"] == "choose":
            content: Dict[str, Any] = {"title": card["title"],
                                       "options": [{"id": key, "label": label} for key, label in options.items()]}
        else:
            content = {"title": card["title"], **({"body": card["body"]} if card.get("body") else {})}
        try:
            asked = await self.api.create_interaction(
                {"kind": card["kind"], "card": content, "thread": chat_id, "ttlSeconds": QUESTION_TTL_SECONDS})
        except ApiError as exc:
            logger.error("[Agent Inbox] Couldn't ask in %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc), retryable=exc.outage)
        self._watch(str(asked["id"]), Pending(kind="clarify", session_key=session_key, chat_id=chat_id,
                                              clarify_id=clarify_id, options=options))
        return SendResult(success=True, message_id=str(asked["id"]))

    async def send_slash_confirm(self, chat_id: str, title: str, message: str, session_key: str, confirm_id: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """A command Hermes confirms first, as it throws something away (`/new`, `/undo`, `/reload-mcp`): whoever typed
        it answers with buttons, as for an approval: the operator, on their own channel, or a contact the operator gave
        full access, in their own conversation. Only they type commands here. Hermes's text form would ask for a typed
        `/approve`, and words like "Approve once" would reach Hermes as a message instead. "Always" is never offered:
        it writes Hermes's settings."""
        approver = self._approver(chat_id, (metadata or {}).get("thread_id"))
        if _slash_confirm_resolver() is None or approver is None:
            return SendResult(success=False, error="Not supported")
        heading, body = words.confirm_card(title, message)
        try:
            asked = await self.api.create_interaction({
                "kind": "approve",
                "card": {"title": heading, **({"body": body} if body else {}), "approveLabel": "Approve",
                         "denyLabel": "Cancel"},
                **approver.address,
                "from": HOST_NAME,
                "ttlSeconds": CONFIRM_TTL_SECONDS,
            })
        except ApiError as exc:
            logger.error("[Agent Inbox] Couldn't ask %s to confirm %s: %s", approver.who, title, exc)
            return SendResult(success=False, error=str(exc), retryable=exc.outage)
        self._watch(str(asked["id"]), Pending(kind="confirm", session_key=session_key, chat_id=chat_id,
                                              confirm_id=confirm_id, asked=approver.who))
        return SendResult(success=True, message_id=str(asked["id"]))

    async def interrupt_session_activity(self, session_key: str, chat_id: str, metadata: Any = None) -> None:
        """`/stop` in a conversation: its open cards come off people's phones."""
        for interaction_id, pending in list(self._pending.items()):
            if pending.session_key == session_key:
                self._pending.pop(interaction_id, None)
                await self._withdraw(interaction_id)
        await super().interrupt_session_activity(session_key, chat_id, metadata)

    # --- Hermes's tools ---

    def starter_of(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """The message that started this chat's run, as its cards name it; None before any came here."""
        return self._latest.get(chat_id)

    def writer_of(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """The latest message in this chat, commands too: who wrote it, and whether it is a group. After a restart,
        until someone writes there again, the message the chat's run started with. None before any came here."""
        return self._writers.get(chat_id) or self._latest.get(chat_id)

    def _approver(self, chat_id: str, thread_id: Optional[str]) -> Optional[Approver]:
        """Who allows what Hermes's run in a session asks first: the contact the operator gave full access, in the
        session of theirs at full access that is still their conversation's current one; for any other run, the
        operator. None while the operator isn't known."""
        full_access = self.full_access_session(chat_id, thread_id)
        if full_access is not None:
            return Approver(id=str(full_access["person"]), who=str(full_access.get("name") or "") or "the contact",
                            thread=chat_id)
        operator = self.listener.operator_id
        return Approver(id=str(operator)) if operator else None

    # --- levels ---

    def level_of(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """The level a contact's direct conversation has now, as their latest message set it, also after a restart.
        None for the operator's own conversations, a group, and a contact's conversation that never had full access:
        those run at the level they always had."""
        return self.listener.state.levels.get(chat_id)

    def full_access_session(self, chat_id: str, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """The level of the contact whose full access a session of this chat runs at: only a session started at full
        access that is still their conversation's current one. None for any other: a message without it took it away,
        at once, also from a run going on in it."""
        level, since = thread_level(thread_id)
        current = self.level_of(chat_id)
        if level != FULL or current is None or current.get("level") != FULL or current.get("since") != since or \
                not current.get("person"):
            return None
        return current

    async def _session_thread(self, conversation: str, message: Dict[str, Any]) -> Optional[str]:
        """The thread of the session a message goes to. The operator's own messages and a group's go on as before. A
        contact's direct conversation keeps the level its session started at: a message at another one, as when the
        operator turned Full access on or off, starts a fresh session first, and the approval cards still waiting in
        the old one are withdrawn. Nothing carries from one level into the other: the talk, what the session read,
        its /yolo and its "Allow for this session"."""
        round_number = int(message.get("round") or 0)
        sender = message.get("from") or {}
        if sender.get("isOperator") or message.get("isDirectMessage") is False:
            return session_thread(round_number)
        level = FULL if words.full_access(message) else LIMITED
        current = self.level_of(conversation)
        if current is None and level == LIMITED:
            return session_thread(round_number)  # never had full access: its sessions go on as before
        current = dict(current or {"level": LIMITED, "since": ""})
        changed = current.get("level") != level
        if changed:
            current = {"level": level, "since": new_level_token()}
        name = words.clean(sender.get("name")).strip()
        current.update(thread=session_thread(round_number, level, current["since"]),
                       person=str(sender.get("id") or ""), name=name)
        self.listener.state.set_level(conversation, current)
        if changed:
            self.listener.save()
            logger.info("[Agent Inbox] %s is at %s access in %s now: a fresh session.", sender.get("id"), level,
                        conversation)
            await self._withdraw_cards(conversation, (
                f"{name or 'The contact'}'s full access was turned {'on' if level == FULL else 'off'}, so their "
                "conversation moved to a fresh session: it isn't allowed in this one."))
        return current["thread"]

    def _chat_for_answer(self, chat: Dict[str, Any]) -> Dict[str, Any]:
        """The session an answer to Hermes's own question goes to: the one that asked, at its level. When that one ran
        at full access and the switch changed since, the conversation's current session instead, so the answer never
        keeps anyone at full access, nor brings them what that session read."""
        chat_id = str(chat.get("chat_id") or "")
        thread = chat.get("thread_id")
        if thread_level(thread)[0] != FULL or self.full_access_session(chat_id, thread) is not None:
            return chat
        current = self.level_of(chat_id) or {}
        return {**chat, "thread_id": current.get("thread") or round_thread(thread)}

    def expect_answer(self, question: Dict[str, Any]) -> None:
        """A question Hermes asked with its tools: its answer comes back to the chat that asked, also after a restart.
        On the adapter's loop."""
        self.listener.state.wait_for(question)
        self.listener.save()
        self._start_polling()

    async def resolve_after_reply(self, chat_id: str) -> None:
        """Hermes resolved the conversation it works in. Any message into it would open it again, Hermes's answer too,
        so it is resolved once Hermes has been quiet there for a moment: no typing, no reply. A message from someone
        there first keeps it open. On the adapter's loop."""
        self._resolving[chat_id] = None
        self._put_off_resolving(chat_id)

    def _put_off_resolving(self, chat_id: str) -> None:
        if chat_id not in self._resolving:
            return
        waiting = self._resolving.get(chat_id)
        if waiting is not None and not waiting.done():
            waiting.cancel()
        self._resolving[chat_id] = asyncio.create_task(self._resolve_when_quiet(chat_id))

    def _keep_open(self, chat_id: str) -> None:
        waiting = self._resolving.pop(chat_id, None)
        if waiting is not None and not waiting.done():
            waiting.cancel()

    async def _resolve_when_quiet(self, chat_id: str) -> None:
        await asyncio.sleep(self._resolve_quiet_seconds)
        await self._resolve_now(chat_id)

    async def _resolve_now(self, chat_id: str) -> None:
        if chat_id not in self._resolving:
            return
        waiting = self._resolving.pop(chat_id)
        if waiting is not None and waiting is not asyncio.current_task() and not waiting.done():
            waiting.cancel()
        try:
            await self.api.resolve_conversation(chat_id)
            logger.info("[Agent Inbox] Resolved %s, as Hermes asked.", chat_id)
        except ApiError as exc:
            logger.warning("[Agent Inbox] Couldn't resolve %s: %s", chat_id, exc)

    async def _answered(self, question_id: str, status: str, option_id: Optional[str], text: Optional[str]) -> None:
        question = self.listener.state.waiting.pop(question_id, None)
        if question is None:
            return
        # Its answer reached Hermes: another one, or the same coming with the messages later, starts nothing.
        self.listener.state.remember("own", question_id)
        self.listener.save()
        picked = (question.get("options") or {}).get(str(option_id or "")) if option_id else None
        note = None if status == "canceled" else words.answer_note(question, status, picked=picked, text=text)
        chat = question.get("chat") or _conversation_chat(question)
        if note is None or chat is None:
            return
        await self._tell_hermes(self._chat_for_answer(chat), note, f"{question_id}:answer")

    async def _tell_hermes(self, chat: Dict[str, Any], text: str, message_id: str) -> None:
        """A message to Hermes in one of its Agent Inbox chats, from the plugin itself."""
        chat_id = str(chat["chat_id"])
        starter = chat.get("starter")
        if starter and chat_id not in self._latest:
            # After a restart: whose chat this is stays known, for what Hermes may do next in it.
            self._latest[chat_id] = starter
        source = self.build_source(
            chat_id=chat_id,
            chat_name=str(chat.get("chat_name") or chat_id),
            chat_type=str(chat.get("chat_type") or "dm"),
            user_id=chat.get("user_id") or None,
            user_name=chat.get("user_name") or None,
            thread_id=chat.get("thread_id") or None,
            message_id=message_id,
        )
        await self.handle_message(MessageEvent(text=text, message_type=MessageType.TEXT, source=source,
                                               message_id=message_id, allow_gateway_control=False))

    # --- open cards ---

    def _watch(self, interaction_id: str, pending: Pending) -> None:
        self._pending[interaction_id] = pending
        self.listener.remember_own(interaction_id)
        self._start_polling()

    async def _withdraw_cards(self, conversation: str, reason: str) -> None:
        """A conversation moved to a fresh session as its contact's level changed: the approval and confirmation
        cards still waiting in it are withdrawn, whoever they ask, and count as a Deny, so a tap allows nothing in the
        session left behind. Hermes's questions to the person stay: answering one allows nothing."""
        for interaction_id, pending in list(self._pending.items()):
            if pending.kind == "clarify" or pending.chat_id != conversation:
                continue
            self._pending.pop(interaction_id, None)
            await self._withdraw(interaction_id)
            try:
                if pending.kind == "approval":
                    _resolve_approval(pending.session_key, "deny", reason)
                else:
                    await self._settle(pending, "denied", None, None)
            except Exception:  # noqa: BLE001 — their message still reaches Hermes; the wait ends as a Deny anyway
                logger.exception("[Agent Inbox] Couldn't tell Hermes a card in %s was withdrawn.", conversation)
            logger.info("[Agent Inbox] Withdrew a card waiting in %s: its session was left for a fresh one.",
                        conversation)

    def _start_polling(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_answers())

    async def _poll_answers(self) -> None:
        """Answers usually come with the messages; this finds the ones that don't: a card's, and the answer of a
        contact the agent doesn't listen to."""
        while self._pending or self.listener.state.waiting:
            await asyncio.sleep(self._answer_poll_seconds)
            for interaction_id in list(self._pending):
                try:
                    asked = await self.api.get_interaction(interaction_id)
                except ApiError:
                    continue
                status = str(asked.get("status") or "pending")
                if status == "pending":
                    continue
                pending = self._pending.pop(interaction_id, None)
                if pending is not None:
                    response = asked.get("response") or {}
                    await self._settle(pending, status, response.get("optionId"), response.get("text"))
            if self.listener.state.waiting and \
                    time.monotonic() - self._waiting_polled_at >= self._waiting_poll_seconds:
                self._waiting_polled_at = time.monotonic()
                await self._poll_waiting()

    async def _poll_waiting(self) -> None:
        for question_id in list(self.listener.state.waiting):
            try:
                asked = await self.api.get_interaction(question_id)
            except ApiError as exc:
                if exc.status == 404:
                    # Gone, as with another key: nothing will answer it.
                    self.listener.state.waiting.pop(question_id, None)
                    self.listener.save()
                continue
            status = str(asked.get("status") or "pending")
            if status != "pending":
                response = asked.get("response") or {}
                await self._answered(question_id, status, response.get("optionId"), response.get("text"))

    async def _withdraw(self, interaction_id: str) -> None:
        try:
            await self.api.cancel_interaction(interaction_id)
        except ApiError:
            pass
