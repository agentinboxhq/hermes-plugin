"""Agent Inbox's API, as far as the plugin needs it: the calls `npx @agentinbox/cli listen` makes, connecting, and
what Hermes's tools read and send."""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

from .settings import DEFAULT_AGENT, HOST_ID

PLUGIN_VERSION = "0.1.0"

# The longest the server holds one wait for messages, in seconds (the API's own limit).
MAX_WAIT_SECONDS = 25

# Who calls, as the API records it: `plugin`, Agent Inbox's plugin in a host's gateway, which holds the agent's key
# as the CLI does (`cli`), where an app that signed in is `mcp`. The host header names Hermes, so agentinbox.md
# shows "Hermes · Plugin" and the lease is Hermes's.
VIA_HEADER = "Agent-Inbox-Via"
VIA = "plugin"
HOST_HEADER = "Agent-Inbox-Host"

# The codes the API puts on the errors the listener acts on.
LISTENING_ELSEWHERE = "listening_elsewhere"
NOT_LISTENING = "not_listening"
AGENT_PAUSED = "agent_paused"
INVALID_CURSOR = "invalid_listen_cursor"


class ApiError(Exception):
    """A request that failed: `status` 0 when Agent Inbox wasn't reached at all."""

    def __init__(self, message: str, status: int, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def code(self) -> Optional[str]:
        code = self.body.get("code") if isinstance(self.body, dict) else None
        return code if isinstance(code, str) else None

    @property
    def listening(self) -> Optional[Dict[str, Any]]:
        """The lease's holder that a `listening_elsewhere` or `not_listening` error names, if it names one."""
        listening = self.body.get("listening") if isinstance(self.body, dict) else None
        return listening if isinstance(listening, dict) and "computerId" in listening else None

    @property
    def key_rejected(self) -> bool:
        """The key was regenerated on agentinbox.md (or never worked): no retry will help."""
        return self.status == 401

    @property
    def outage(self) -> bool:
        """Agent Inbox wasn't reached, or was busy for a moment: the same call may go through a little later. A limit
        with a code of its own, such as the day's messages, lasts too long to wait for."""
        if self.status == 429:
            return self.code is None
        return self.status in (0, 408) or self.status >= 500


def unwrap(body: Any) -> Any:
    """Novu's `{ data: ... }` envelope, which some endpoints leave out."""
    if isinstance(body, dict) and body.get("data") is not None:
        return body["data"]
    return body


def _message_of(body: Any) -> Optional[str]:
    message = body.get("message") if isinstance(body, dict) else None
    if isinstance(message, list):
        return "; ".join(str(part) for part in message)
    return message if isinstance(message, str) else None


def is_loopback(url: str) -> bool:
    """A local dev API: its proxies present self-signed certificates, and its traffic never leaves the machine."""
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class AgentInboxApi:
    def __init__(self, api_url: str, secret_key: str = "", agent: str = DEFAULT_AGENT, *, transport: Any = None):
        import httpx

        self.api_url = api_url.rstrip("/")
        self.agent = agent or DEFAULT_AGENT
        headers = {
            VIA_HEADER: VIA,
            HOST_HEADER: HOST_ID,
            "User-Agent": f"agentinbox-hermes/{PLUGIN_VERSION}",
        }
        if secret_key:
            headers["Authorization"] = f"ApiKey {secret_key}"
        self._client = httpx.AsyncClient(
            base_url=self.api_url, headers=headers, timeout=60.0, transport=transport,
            verify=not is_loopback(self.api_url))

    async def close(self) -> None:
        await self._client.aclose()

    def _agent(self) -> Dict[str, str]:
        """The default agent stays out of every request, as the CLI leaves it out."""
        return {} if self.agent == DEFAULT_AGENT else {"agentIdentifier": self.agent}

    async def _request(self, method: str, path: str, *, json: Any = None, params: Optional[Dict[str, Any]] = None,
                       timeout: Optional[float] = None, raw: bool = False) -> Any:
        """`raw` keeps the envelope, for a page whose `next` sits beside its `data`."""
        import httpx

        kwargs: Dict[str, Any] = {"json": json, "params": params}
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            res = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(f"{method} {path} timed out", 408) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Couldn't reach Agent Inbox at {self.api_url}: {exc}", 0) from exc
        try:
            body: Any = res.json() if res.content else None
        except ValueError:
            body = res.text
        if res.status_code >= 400:
            raise ApiError(_message_of(body) or f"{method} {path} answered {res.status_code}", res.status_code, body)
        return body if raw else unwrap(body)

    async def _page(self, path: str, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """One page of a list: its items, and the cursor for the next page, None after the last."""
        body = await self._request("GET", path, params={**{k: v for k, v in params.items() if v is not None},
                                                        **self._agent()}, raw=True)
        if isinstance(body, list):
            return [item for item in body if isinstance(item, dict)], None
        body = body if isinstance(body, dict) else {}
        data = body.get("data") if isinstance(body.get("data"), list) else []
        return [item for item in data if isinstance(item, dict)], (str(body.get("next") or "") or None)

    # --- connecting ---

    async def connect_with_code(self, agent: str, code: str) -> Dict[str, Any]:
        """Trades the one-time code from agentinbox.md for the agent's key. It works once."""
        connected = await self._request("POST", "/v1/connect", json={"agent": agent, "code": code})
        if not isinstance(connected, dict) or not connected.get("apiKey"):
            raise ApiError("Agent Inbox didn't connect. Try again with a new code.", 500, connected)
        return connected

    async def find_operator(self) -> Optional[str]:
        """The operator's contact id: the person whom permission questions and notices go to."""
        try:
            data = await self._request("GET", "/v1/operator")
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(data, dict):
            return None
        return str(data.get("subscriberId") or "") or None

    # --- listening ---

    async def take_lease(self, *, computer_id: str, computer_name: str, take_over: bool = False) -> Dict[str, Any]:
        """Takes the lease, renews it, or with `take_over` takes it from another computer."""
        body: Dict[str, Any] = {"computerId": computer_id, "computerName": computer_name, "host": HOST_ID,
                                **self._agent()}
        if take_over:
            body["takeOver"] = True
        return await self._request("POST", "/v1/listen/lease", json=body)

    async def get_lease(self) -> Dict[str, Any]:
        """Where the agent is listening now (`listening`, or None), and whether its operator `paused` it."""
        status = await self._request("GET", "/v1/listen/lease", params=self._agent()) or {}
        return {"listening": status.get("listening") or None, "paused": status.get("paused") is True}

    async def release_lease(self, computer_id: str) -> bool:
        """Lets go of the lease: False when this computer didn't hold it, and another holder's is never touched."""
        data = await self._request("POST", "/v1/listen/lease/release",
                                   json={"computerId": computer_id, "host": HOST_ID, **self._agent()})
        return bool((data or {}).get("released"))

    async def show_typing(self, computer_id: str, conversation_id: str) -> bool:
        """"Typing…" in the conversation for a few seconds; False where the chat has none, as in email."""
        data = await self._request("POST", "/v1/listen/typing", json={
            "computerId": computer_id, "host": HOST_ID, "conversationId": conversation_id, **self._agent()})
        return bool((data or {}).get("typing"))

    async def wait_for_messages(self, *, computer_id: str, after: Optional[str], wait: int,
                                limit: int = 20) -> Tuple[List[Dict[str, Any]], str]:
        """Up to `wait` seconds for messages that may start Hermes: the operator's, and those of the contacts the agent
        listens to. Only the lease's holder may wait. Oldest first, with where to go on from next time."""
        wait = max(0, min(wait, MAX_WAIT_SECONDS))
        params: Dict[str, Any] = {"computerId": computer_id, "host": HOST_ID, "wait": wait, "limit": limit,
                                  **self._agent()}
        if after:
            params["after"] = after
        import httpx

        res_timeout = httpx.Timeout(10.0, read=wait + 30.0)
        try:
            res = await self._client.get("/v1/listen/messages", params=params, timeout=res_timeout)
        except httpx.TimeoutException as exc:
            raise ApiError("Waiting for messages timed out", 408) from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"Couldn't reach Agent Inbox at {self.api_url}: {exc}", 0) from exc
        try:
            body: Any = res.json() if res.content else {}
        except ValueError:
            body = res.text
        if res.status_code >= 400:
            raise ApiError(_message_of(body) or f"Waiting for messages answered {res.status_code}",
                           res.status_code, body)
        body = body if isinstance(body, dict) else {}
        data = body.get("data") if isinstance(body.get("data"), list) else []
        return [item for item in data if isinstance(item, dict)], str(body.get("next") or after or "")

    # --- messages from the agent ---

    async def create_interaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sends a message from the agent: `tell`, `ask`, `approve` or `choose`, into a conversation (`thread`) or to a
        contact (`to`)."""
        return await self._request("POST", "/v1/inbox/messages", json={**payload, **self._agent()})

    async def show_card(self) -> Dict[str, Any]:
        """Shows the agent's card to its operator: what the agent says, and what its card holds. The first showing is
        kept on the agent, which is how agentinbox.md's onboarding sees it say hi."""
        found = await self._request("POST", "/v1/agent/card", json=self._agent())
        return found if isinstance(found, dict) else {}

    async def get_interaction(self, interaction_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v1/inbox/messages/{quote(interaction_id, safe='')}")

    async def cancel_interaction(self, interaction_id: str) -> Dict[str, Any]:
        """Withdraws a question still open: it can't be answered any more, and shows as canceled."""
        return await self._request("POST", f"/v1/inbox/messages/{quote(interaction_id, safe='')}/cancel", json={})

    # --- what Hermes's tools read ---

    async def list_conversations(self, *, filter: str = "all", status: str = "all", limit: int = 20,
                                 after: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """The agent's conversations, latest activity first. Strangers' stay out, as in the CLI's `inbox list`."""
        return await self._page("/v1/inbox", {"filter": filter, "status": status, "senders": "contacts",
                                              "limit": limit, "after": after})

    async def get_conversation(self, conversation_id: str, *, limit: int = 20,
                               before: Optional[str] = None) -> Dict[str, Any]:
        """A conversation and its latest messages, oldest first; `hasMore` when older ones are left. Reading it
        doesn't mark it read."""
        params = {"limit": limit, **({"before": before} if before else {}), **self._agent()}
        found = await self._request("GET", f"/v1/inbox/{quote(conversation_id, safe='')}", params=params)
        return found if isinstance(found, dict) else {}

    async def resolve_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Marks a conversation finished. Nothing is sent; the next message in it, from anyone, opens it again, and
        that message starts the conversation's next round."""
        found = await self._request("POST", f"/v1/inbox/{quote(conversation_id, safe='')}/resolve", json={},
                                    params=self._agent())
        return found if isinstance(found, dict) else {}

    async def list_contacts(self, *, limit: int = 50,
                            after: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """The operator's contacts, the operator among them."""
        return await self._page("/v1/contacts", {"limit": limit, "after": after})
