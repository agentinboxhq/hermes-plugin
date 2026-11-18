"""The listen loop, with the rules of `npx @agentinbox/cli listen`: nothing starts before the operator allows it, one
computer listens at a time (the lease), and only the operator's messages and those of the contacts the agent listens
to come here. Hermes's gateway runs the sessions; this hands it each message."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from . import words
from .api import AGENT_PAUSED, INVALID_CURSOR, LISTENING_ELSEWHERE, NOT_LISTENING, AgentInboxApi, ApiError
from .settings import HOST_ID, HOST_NAME, Settings
from .state import State, StateStore

logger = logging.getLogger(__name__)

# How long the operator has to allow listening here.
ALLOW_TTL_SECONDS = 60 * 60

# Hosts as people read them, for where the agent listens now.
HOST_NAMES = {
    "claude-code": "Claude Code",
    "codex": "Codex",
    "opencode": "OpenCode",
    "cursor": "Cursor",
    "hermes": "Hermes",
    "openclaw": "OpenClaw",
}

OnMessage = Callable[[Dict[str, Any]], Awaitable[None]]
OnAnswer = Callable[[Dict[str, Any]], Awaitable[bool]]


@dataclass
class Timing:
    # How long the server holds one wait for messages, in seconds.
    wait_seconds: int = 25
    # How often a paused agent is tried again: a pause is the operator's choice, not an outage.
    paused_retry: float = 180.0
    # How long to wait after one, two, three… failures in a row.
    retry: Sequence[float] = (5.0, 15.0, 30.0, 60.0)
    # How often the answer to the Allow question is looked up.
    allow_poll: float = 2.0


class StopForGood(Exception):
    """Trying again won't help until something changes: a new key, the operator's answer, a restart to take it back."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason
        self.message = message


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _seconds_until(stamp: Any, clock: Callable[[], float]) -> Optional[float]:
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return at.timestamp() - clock()


class Listener:
    def __init__(self, *, api: AgentInboxApi, store: StateStore, settings: Settings, computer_name: str,
                 on_message: OnMessage, on_answer: OnAnswer, timing: Optional[Timing] = None,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep):
        self.api = api
        self.store = store
        self.settings = settings
        self.computer_name = computer_name
        self.on_message = on_message
        self.on_answer = on_answer
        self.timing = timing or Timing()
        self.clock = clock
        self.sleep = sleep
        self.state: State = store.load(settings.connection)
        self.operator_id: Optional[str] = None
        # This gateway holds the lease now: only then may it wait for messages, show typing, or let go of it.
        self.holding = False
        self._renew_at = 0.0
        self._failures = 0
        self._told_paused = False

    @property
    def computer_id(self) -> str:
        return self.state.computer_id

    def save(self) -> None:
        self.store.save(self.state)

    def remember_own(self, interaction_id: str) -> None:
        """One of the plugin's own questions: its answer never starts Hermes, also after a restart."""
        self.state.remember("own", interaction_id)
        self.save()

    async def run(self) -> None:
        """Listens until cancelled. Raises `StopForGood` when it can't go on."""
        self.save()  # keeps the computer id from the first start on
        self.operator_id = await self._keep_trying(self.api.find_operator)
        if not self.operator_id:
            raise StopForGood("no_operator", words.NO_OPERATOR_MESSAGE)
        await self._ensure_allowed()
        await self._listen()

    async def close(self) -> None:
        """Lets go of the lease, so agentinbox.md shows nobody listening until the gateway is back."""
        if not self.holding:
            return
        self.holding = False
        try:
            await self.api.release_lease(self.computer_id)
        except ApiError as exc:
            logger.info("[Agent Inbox] Couldn't let go of listening (%s); it lapses by itself.", exc)

    # --- turning it on ---

    async def _ensure_allowed(self) -> None:
        if self.state.allowed_at:
            return

        status = await self._keep_trying(self.api.get_lease)
        listening = status.get("listening")
        elsewhere = None if self._is_mine(listening) else listening
        title, body = words.allow_card(computer_name=self.computer_name, elsewhere=elsewhere,
                                       computer_id=self.computer_id, host_names=HOST_NAMES)
        asked = self.state.asked
        # The same request waits on the same question; one that reads otherwise now asks anew, since the operator
        # allows what they read.
        if not asked or asked.get("title") != title or (_seconds_until(asked.get("expiresAt"), self.clock) or 0) <= 0:
            interaction = await self._keep_trying(lambda: self.api.create_interaction({
                "kind": "approve",
                "card": {"title": title, "body": body, "approveLabel": "Allow", "denyLabel": "Deny"},
                "to": self.operator_id,
                "from": HOST_NAME,
                "ttlSeconds": ALLOW_TTL_SECONDS,
            }))
            asked = {"id": str(interaction["id"]), "expiresAt": interaction.get("expiresAt"), "title": title}
            self.state.asked = asked
            self.state.remember("own", asked["id"])
            self.save()
            logger.info("[Agent Inbox] Asked your operator to allow listening on %s.", self.computer_name)

        outcome = await self._wait_for_allow(asked)
        self.state.asked = None
        if outcome == "approved":
            self.state.allowed_at = now_iso()
            self.state.take_over = bool(elsewhere)
            self.save()
            logger.info("[Agent Inbox] Your operator allowed listening on %s.", self.computer_name)
            return
        self.save()
        if outcome == "denied":
            raise StopForGood("denied", words.DENIED_MESSAGE)
        raise StopForGood("unanswered", words.UNANSWERED_MESSAGE)

    async def _wait_for_allow(self, asked: Mapping[str, Any]) -> str:
        while True:
            interaction = await self._keep_trying(lambda: self.api.get_interaction(str(asked["id"])))
            status = str(interaction.get("status") or "")
            if status == "approved":
                return "approved"
            # Words instead of a button aren't an Allow.
            if status in ("denied", "answered"):
                return "denied"
            left = _seconds_until(interaction.get("expiresAt") or asked.get("expiresAt"), self.clock)
            if status not in ("", "pending") or (left is not None and left < -60):
                return "unanswered"
            await self.sleep(self.timing.allow_poll)

    # --- listening ---

    async def _listen(self) -> None:
        while True:
            try:
                if self.clock() >= self._renew_at:
                    await self._take_lease()
                wait = int(max(1, min(self.timing.wait_seconds, self._renew_at - self.clock())))
                messages, after = await self.api.wait_for_messages(
                    computer_id=self.computer_id, after=self.state.cursor, wait=wait)
                self._failures = 0
            except ApiError as exc:
                await self._on_error(exc)
                continue

            for message in messages:
                await self._handle(message)
            if messages or (after and after != self.state.cursor):
                self.state.cursor = after or self.state.cursor
                self.save()

    async def _take_lease(self) -> None:
        try:
            lease = await self.api.take_lease(computer_id=self.computer_id, computer_name=self.computer_name,
                                              take_over=self.state.take_over)
        except ApiError as exc:
            if exc.code == LISTENING_ELSEWHERE:
                raise self._moved(exc.listening) from exc
            raise
        if not self.holding:
            logger.info("[Agent Inbox] Listening on %s for %s.", self.computer_name, HOST_NAME)
        self.holding = True
        self._told_paused = False
        if self.state.take_over:
            self.state.take_over = False
            self.save()
        renew = lease.get("renewInSeconds") if isinstance(lease, dict) else None
        self._renew_at = self.clock() + max(5.0, float(renew or 60))

    async def _on_error(self, exc: ApiError) -> None:
        if exc.key_rejected:
            self.holding = False
            raise StopForGood("key_changed", words.KEY_CHANGED_MESSAGE) from exc
        if exc.code == AGENT_PAUSED:
            if not self._told_paused:
                self._told_paused = True
                logger.info("[Agent Inbox] %s", words.PAUSED_MESSAGE)
            self._renew_at = 0.0
            await self.sleep(self.timing.paused_retry)
            return
        if exc.code in (NOT_LISTENING, LISTENING_ELSEWHERE):
            holder = exc.listening
            if holder and not self._is_mine(holder):
                raise self._moved(holder) from exc
            # The lease lapsed, as when the computer slept: take it again.
            self.holding = False
            self._renew_at = 0.0
            return
        if exc.code == INVALID_CURSOR:
            # Not a cursor this API handed out (another deployment's, say): waiting goes on from now.
            self.state.cursor = None
            self.save()
            return

        wait = self.timing.retry[min(self._failures, len(self.timing.retry) - 1)]
        self._failures += 1
        logger.warning("[Agent Inbox] %s; trying again in %s.", exc, words.span_words(wait))
        await self.sleep(wait)

    async def _handle(self, message: Dict[str, Any]) -> None:
        message_id = str(message.get("id") or "")
        if not message_id or message_id in self.state.handled or not message.get("conversationId"):
            return
        self.state.remember("handled", message_id)
        answers = str(message.get("answersMessageId") or "")
        try:
            if answers and await self.on_answer(message):
                return
            if answers and answers in self.state.own:
                return  # an answer to a question of the plugin's that nothing waits for any more
            await self.on_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — one message must not stop the others
            logger.exception("[Agent Inbox] Couldn't hand message %s to %s.", message_id, HOST_NAME)

    # --- helpers ---

    def _moved(self, holder: Any) -> StopForGood:
        """Another computer, or another host here, took the agent over. The Allow went with it: the next start asks
        the operator again, with a card that says it moves the agent back here."""
        self.holding = False
        self.state.allowed_at = None
        self.state.take_over = False
        self.save()
        return StopForGood("listening_elsewhere", words.moved_message(holder, self.computer_id, HOST_NAMES))

    def _is_mine(self, listening: Any) -> bool:
        return isinstance(listening, Mapping) and listening.get("computerId") == self.computer_id and \
            listening.get("host") == HOST_ID

    async def _keep_trying(self, call: Callable[[], Awaitable[Any]]) -> Any:
        """A call before listening starts: tried again while Agent Inbox can't be reached or the agent is paused."""
        failures = 0
        while True:
            try:
                return await call()
            except ApiError as exc:
                if exc.key_rejected:
                    raise StopForGood("key_changed", words.KEY_CHANGED_MESSAGE) from exc
                if exc.code == AGENT_PAUSED:
                    if not self._told_paused:
                        self._told_paused = True
                        logger.info("[Agent Inbox] %s", words.PAUSED_MESSAGE)
                    await self.sleep(self.timing.paused_retry)
                    continue
                if not exc.outage:
                    raise
                wait = self.timing.retry[min(failures, len(self.timing.retry) - 1)]
                failures += 1
                logger.warning("[Agent Inbox] %s; trying again in %s.", exc, words.span_words(wait))
                await self.sleep(wait)
