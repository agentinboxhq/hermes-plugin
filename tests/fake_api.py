"""A fake of Agent Inbox's API, as far as the plugin calls it, behind httpx's MockTransport."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import httpx

OPERATOR = "op_ada"


def stamp(seconds_from_now: float = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)).isoformat().replace("+00:00", "Z")


def message(id: str, *, conversation: str = "conv_1", text: str = "hello", operator: bool = True,
            name: str = "Ada", sender: str = OPERATOR, channel: str = "telegram", round: int = 0,
            direct: bool = True, answers: Optional[str] = None, answer: Optional[dict] = None,
            full_access: bool = False) -> Dict[str, Any]:
    found: Dict[str, Any] = {"id": id, "conversationId": conversation, "channel": channel, "text": text,
                             "from": {"id": sender, "name": name, "isOperator": operator}, "at": stamp(),
                             "round": round, "isDirectMessage": direct}
    if full_access:
        # As the API says it: only on a contact's messages, and only while the operator's switch is on.
        found["from"]["fullAccess"] = True
    if answers:
        found["answersMessageId"] = answers
        found["answer"] = answer or {"status": "answered", "text": text}
    return found


class FakeApi:
    def __init__(self) -> None:
        self.holder: Optional[Dict[str, Any]] = None
        self.paused = False
        self.operator: Optional[str] = OPERATOR
        self.queue: List[Dict[str, Any]] = []
        self.interactions: Dict[str, Dict[str, Any]] = {}
        self.sent: List[Dict[str, Any]] = []
        self.typing: List[str] = []
        self.calls: List[str] = []
        # Who each call said it is: the plugin's `Agent-Inbox-Via` and `Agent-Inbox-Host` headers.
        self.callers: List[tuple] = []
        self.failures: Dict[str, List[httpx.Response]] = {}
        self.codes: Dict[str, Dict[str, Any]] = {}
        self.released = 0
        self.new_messages = asyncio.Event()
        # What the tools read: the contacts (pages of `contacts_page`), the conversations, and each one's messages.
        self.contacts: List[Dict[str, Any]] = []
        self.contacts_page = 100
        self.conversations: List[Dict[str, Any]] = []
        self.conversations_next: Optional[str] = None
        self.views: Dict[str, Dict[str, Any]] = {}
        self.reads: List[Dict[str, str]] = []
        # Recipients whose channels take nothing: a send to them comes back with them in `failedTo`.
        self.unreachable: List[str] = []
        # The conversations resolved, in order, with what each call asked as.
        self.resolved: List[str] = []
        self.resolved_as: List[Dict[str, str]] = []
        self._next = 0

    # --- what a test does ---

    def deliver(self, *messages: Dict[str, Any]) -> None:
        self.queue.extend(messages)
        self.new_messages.set()

    def answer(self, interaction_id: str, status: str, *, option_id: Optional[str] = None,
               text: Optional[str] = None) -> None:
        interaction = self.interactions[interaction_id]
        interaction["status"] = status
        interaction["response"] = {"type": "option" if option_id else "text", "optionId": option_id, "text": text,
                                   "respondedAt": stamp()}

    def fail(self, route: str, status: int, body: Optional[dict] = None, times: int = 1) -> None:
        """The next `times` calls of `route` ("POST /v1/listen/lease") answer with this error."""
        self.failures.setdefault(route, []).extend(httpx.Response(status, json=body or {}) for _ in range(times))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def last(self, kind: str) -> Dict[str, Any]:
        return [each for each in self.sent if each.get("kind") == kind][-1]

    # --- the API ---

    async def handle(self, request: httpx.Request) -> httpx.Response:
        # The site's `/api` (https://www.agentinbox.md/api/v1/…) or the API itself.
        path = request.url.path
        path = path[path.find("/v1/"):] if "/v1/" in path else path
        route = f"{request.method} {path}"
        self.calls.append(route)
        self.callers.append((request.headers.get("agent-inbox-via"), request.headers.get("agent-inbox-host")))
        if self.failures.get(route):
            return self.failures[route].pop(0)
        body = json.loads(request.content) if request.content else {}
        params = dict(request.url.params)

        if route == "GET /v1/operator":
            return httpx.Response(200, json={"data": {"subscriberId": self.operator}}) if self.operator else \
                httpx.Response(404, json={"message": "No operator"})
        if route == "POST /v1/connect":
            found = self.codes.get(body.get("code"))
            if not found or found["agent"]["handle"] != body.get("agent"):
                return httpx.Response(400, json={"message": "That code doesn't work. Get a new one on agentinbox.md."})
            return httpx.Response(200, json={"data": found})
        if route == "GET /v1/listen/lease":
            return httpx.Response(200, json={"data": {"listening": self.holder, "paused": self.paused}})
        if route == "POST /v1/listen/lease":
            return self._take(body)
        if route == "POST /v1/listen/lease/release":
            mine = self._is_holder(body)
            if mine:
                self.holder = None
                self.released += 1
            return httpx.Response(200, json={"data": {"released": mine}})
        if route == "POST /v1/listen/typing":
            self.typing.append(body["conversationId"])
            return httpx.Response(200, json={"data": {"typing": True}})
        if route == "GET /v1/listen/messages":
            return await self._wait(params)
        if route == "POST /v1/inbox/messages":
            self._next += 1
            to = body.get("to")
            conversation = body.get("thread") or f"conv_with_{to}"
            interaction = {"id": f"msg_{self._next}", "kind": body.get("kind"), "status": "pending",
                           "expiresAt": stamp(body.get("ttlSeconds") or 3600), "payload": body,
                           "threads": [{"id": conversation, "unreadBefore": 0}]}
            if to in self.unreachable:
                interaction["failedTo"] = [to]
            self.interactions[interaction["id"]] = interaction
            self.sent.append({**body, "id": interaction["id"]})
            return httpx.Response(201, json={"data": interaction})
        if route == "GET /v1/contacts":
            start = int(params.get("after") or 0)
            size = min(int(params.get("limit") or 50), self.contacts_page)
            page = self.contacts[start:start + size]
            more = str(start + size) if start + size < len(self.contacts) else None
            return httpx.Response(200, json={"data": page, "next": more})
        if route == "GET /v1/inbox":
            self.reads.append(params)
            return httpx.Response(200, json={"data": self.conversations, "next": self.conversations_next})
        if request.method == "POST" and path.startswith("/v1/inbox/messages/") and path.endswith("/cancel"):
            interaction = self.interactions[unquote(path.split("/")[4])]
            interaction["status"] = "canceled"
            return httpx.Response(200, json={"data": interaction})
        if request.method == "GET" and path.startswith("/v1/inbox/messages/"):
            interaction = self.interactions.get(unquote(path.split("/")[4]))
            return httpx.Response(200, json={"data": interaction}) if interaction else \
                httpx.Response(404, json={"message": "Not found"})
        if request.method == "POST" and path.startswith("/v1/inbox/") and path.endswith("/resolve"):
            conversation = unquote(path.split("/")[3])
            self.resolved.append(conversation)
            self.resolved_as.append(params)
            return httpx.Response(200, json={"data": {"id": conversation, "status": "resolved"}})
        if request.method == "GET" and path.startswith("/v1/inbox/"):
            self.reads.append(params)
            view = self.views.get(unquote(path.split("/")[3]))
            return httpx.Response(200, json={"data": view}) if view else \
                httpx.Response(404, json={"message": "Conversation not found"})
        return httpx.Response(404, json={"message": f"No route {route}"})

    def _is_holder(self, body: Dict[str, Any]) -> bool:
        return bool(self.holder) and self.holder["computerId"] == body.get("computerId") and \
            self.holder["host"] == body.get("host")

    def _take(self, body: Dict[str, Any]) -> httpx.Response:
        if self.paused:
            return httpx.Response(403, json={"code": "agent_paused", "message": "The agent is paused."})
        if self.holder and not self._is_holder(body) and not body.get("takeOver"):
            return httpx.Response(409, json={"code": "listening_elsewhere", "message": "Listening elsewhere",
                                             "listening": self.holder})
        self.holder = {"computerId": body["computerId"], "computerName": body["computerName"],
                       "host": body["host"], "since": stamp(), "renewedAt": stamp(), "expiresAt": stamp(90)}
        return httpx.Response(200, json={"data": {"listening": self.holder, "renewInSeconds": 60}})

    async def _wait(self, params: Dict[str, str]) -> httpx.Response:
        if not self._is_holder({"computerId": params.get("computerId"), "host": params.get("host")}):
            return httpx.Response(409, json={"code": "not_listening", "message": "Not listening",
                                             "listening": self.holder})
        after = params.get("after")
        start = 0
        if after:
            start = next((index + 1 for index, each in enumerate(self.queue) if each["id"] == after), 0)
        if start >= len(self.queue):
            # Holds the wait a moment, as the server does, so a test's loop doesn't spin.
            self.new_messages.clear()
            try:
                await asyncio.wait_for(self.new_messages.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
        page = self.queue[start:start + int(params.get("limit") or 20)]
        next_cursor = page[-1]["id"] if page else (after or "")
        return httpx.Response(200, json={"data": page, "next": next_cursor})
