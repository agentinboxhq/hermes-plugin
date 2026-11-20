"""`hermes agentinbox setup` connects this Hermes profile to an agent with a one-time code from agentinbox.md, and
`hermes agentinbox status` shows where the agent listens."""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any, Callable, Optional

from .api import AgentInboxApi, ApiError
from .computer import computer_name, person_at_terminal
from .listener import HOST_NAMES, now_iso
from .settings import (DEFAULT_API_URL, ENV_AGENT, ENV_API_URL, ENV_SECRET_KEY, HOST_NAME, SETUP_COMMAND, Settings,
                       read_settings, state_dir)
from .state import StateStore

# The API's rule for an agent's handle: what its address starts with, before the @.
_HANDLE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_HANDLE_MAX = 32

CANCELED_MESSAGE = "Nothing changed. The code still works for the rest of its 10 minutes."


class SetupError(Exception):
    pass


def _out() -> tuple:
    """Hermes's own output helpers, with plain ones when this runs outside Hermes."""
    try:
        from hermes_cli.cli_output import print_error, print_info, print_success, print_warning  # type: ignore

        return print_info, print_success, print_warning, print_error
    except Exception:  # noqa: BLE001
        return print, print, print, lambda text: print(text, file=sys.stderr)


def _ask_yes_no(question: str, default: bool) -> bool:
    try:
        from hermes_cli.setup import prompt_yes_no  # type: ignore

        return bool(prompt_yes_no(question, default=default))
    except ImportError:
        answer = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not answer else answer in ("y", "yes")


def read_agent(value: Any) -> str:
    """`atlas`, `Atlas` and `atlas@agentinbox.md` all name the agent `atlas`."""
    handle = str(value or "").strip().lower().split("@")[0]
    if not _HANDLE.match(handle) or len(handle) > _HANDLE_MAX:
        raise SetupError("Name the agent as its address starts, before the @: hermes agentinbox setup atlas "
                         "--code BCDF-GHJK")
    return handle


def read_code(value: Any) -> str:
    """The code as agentinbox.md shows it, e.g. `BCDF-GHJK`. Case, spaces and the dash don't matter."""
    letters = re.sub(r"[\s-]", "", str(value or "").upper())
    if not re.fullmatch(r"[A-Z]{8}", letters):
        raise SetupError("Pass the one-time code from agentinbox.md: eight letters, like --code BCDF-GHJK.")
    return f"{letters[:4]}-{letters[4:]}"


def _save_env(name: str, value: Optional[str]) -> None:
    from hermes_cli.config import remove_env_value, save_env_value  # type: ignore

    if value:
        save_env_value(name, value)
    else:
        remove_env_value(name)


async def _connect(api_url: str, agent: str, code: str) -> tuple:
    public = AgentInboxApi(api_url)
    try:
        connected = await public.connect_with_code(agent, code)
    except ApiError as exc:
        if exc.status == 404 and not exc.code:
            raise SetupError("This API can't connect with a one-time code.") from exc
        raise SetupError(str(exc)) from exc
    finally:
        await public.close()

    details = connected.get("agent") or {}
    settings = Settings(secret_key=str(connected["apiKey"]), agent=str(details.get("identifier") or ""),
                        api_url=api_url)
    if not settings.agent:
        raise SetupError("Agent Inbox didn't say which agent the code is for. Try again with a new code.")
    api = AgentInboxApi(settings.api_url, settings.secret_key, settings.agent)
    try:
        operator = await api.find_operator()
        lease = await api.get_lease()
    except ApiError:
        operator, lease = None, {"listening": None, "paused": False}
    finally:
        await api.close()
    return settings, details, operator, lease


def run_setup(args: Any, *, ask: Callable[[str, bool], bool] = _ask_yes_no,
              at_terminal: Callable[[], bool] = person_at_terminal,
              save_env: Callable[[str, Optional[str]], None] = _save_env,
              store: Optional[StateStore] = None) -> int:
    info, success, warning, error = _out()
    try:
        agent = read_agent(args.agent)
        code = read_code(args.code)
        current = read_settings()
        if current.configured and not args.yes:
            already = "This Hermes profile is connected to Agent Inbox already."
            if not at_terminal():
                raise SetupError(f"{already} Run the same command with --yes to connect it to {agent} instead: the "
                                 "code still works.")
            if not ask(f"{already} Connect it to {agent} instead?", False):
                info(CANCELED_MESSAGE)
                return 1
        api_url = (args.api_url or current.api_url or DEFAULT_API_URL).rstrip("/")
        settings, details, operator, lease = asyncio.run(_connect(api_url, agent, code))
    except SetupError as exc:
        error(str(exc))
        return 1

    save_env(ENV_SECRET_KEY, settings.secret_key)
    save_env(ENV_AGENT, settings.agent)
    save_env(ENV_API_URL, None if settings.api_url == DEFAULT_API_URL else settings.api_url)

    handle = str(details.get("handle") or agent)
    name = str(details.get("name") or "")
    named = f" ({name})" if name and name.lower() != handle else ""
    success(f"Connected this Hermes profile to {handle}{named}.")
    if not operator:
        warning("The account has no operator contact yet: finish setting it up on agentinbox.md first.")

    # What the operator allowed belongs to one connection: a new one starts over, and only this computer's id stays.
    store = store or StateStore(state_dir() / "state.json")
    state = store.load(settings.connection)
    here = computer_name()
    listening = (lease or {}).get("listening")
    elsewhere = None
    if listening and not (listening.get("computerId") == state.computer_id and listening.get("host") == "hermes"):
        elsewhere = listening
    # Run by the operator at their own terminal, `--yes` is their yes to listening here: the commands from
    # agentinbox.md are pasted together, and a question would take the next one as its answer. An agent's run
    # never turns it on: the gateway then asks the operator when it starts.
    moves = ""
    if elsewhere:
        its = HOST_NAMES.get(str(elsewhere.get("host")), str(elsewhere.get("host")))
        moves = f" It listens on {elsewhere.get('computerName')} ({its}) now; this moves it here."
    if at_terminal() and args.yes:
        state.allowed_at = now_iso()
        state.take_over = bool(elsewhere)
        success(f"Listening for messages on {here} is on.{moves}")
    elif at_terminal():
        if ask(f"Listen for messages on {here}, so that one from you, or from a contact it listens to, starts "
               f"{HOST_NAME}?{moves}", True):
            state.allowed_at = now_iso()
            state.take_over = bool(elsewhere)
    store.save(state)

    if state.allowed_at:
        info("Restart the gateway to start listening: hermes gateway restart")
    else:
        info("Restart the gateway: hermes gateway restart. Your operator then gets an Allow and Deny message on their "
             "channel, and listening starts once they allow it.")
    return 0


def run_status(args: Any) -> int:
    info, success, warning, error = _out()
    settings = read_settings()
    if not settings.configured:
        warning(f"Agent Inbox isn't set up for this Hermes profile. Run: {SETUP_COMMAND}")
        return 1
    state = StateStore(state_dir() / "state.json").load(settings.connection)
    info(f"Agent: {settings.agent} at {settings.api_url}")
    info(f"This computer: {computer_name()}")
    info(f"Allowed here: {state.allowed_at}" if state.allowed_at
         else "Not allowed here yet: when the gateway starts, it asks your operator.")

    async def lease() -> dict:
        api = AgentInboxApi(settings.api_url, settings.secret_key, settings.agent)
        try:
            return await api.get_lease()
        finally:
            await api.close()

    try:
        status = asyncio.run(lease())
    except ApiError as exc:
        error(f"Couldn't ask Agent Inbox: {exc}")
        return 1
    if status.get("paused"):
        warning("The agent is paused on agentinbox.md: nothing starts Hermes until you resume it there.")
    listening = status.get("listening")
    if not listening:
        info("Listening: nowhere right now.")
    else:
        mine = listening.get("computerId") == state.computer_id and listening.get("host") == "hermes"
        host = HOST_NAMES.get(str(listening.get("host")), str(listening.get("host")))
        where = "this gateway" if mine else f"{listening.get('computerName')} ({host})"
        success(f"Listening on {where} since {listening.get('since')}.")
    return 0


def setup_argparse(subparser: Any) -> None:
    commands = subparser.add_subparsers(dest="agentinbox_command")
    setup = commands.add_parser("setup", help="Connect this Hermes profile to an agent with a one-time code from "
                                              "agentinbox.md, so its messages reach Hermes")
    setup.add_argument("agent", help="The agent, as its address starts: atlas for atlas@agentinbox.md")
    setup.add_argument("--code", required=True, help="The one-time code agentinbox.md shows, like BCDF-GHJK")
    setup.add_argument("--api-url", help=f"Agent Inbox's API (default {DEFAULT_API_URL})")
    setup.add_argument("--yes", "-y", action="store_true",
                       help="Ask nothing: replace a connection this profile has already, and at your own terminal, "
                            "listen here")
    commands.add_parser("status", help="Show the connection, and where the agent listens")


def handle_cli(args: Any) -> None:
    command = getattr(args, "agentinbox_command", None)
    if command == "setup":
        sys.exit(run_setup(args))
    if command == "status":
        sys.exit(run_status(args))
    print(f"Usage: {SETUP_COMMAND}\n       hermes agentinbox status")
    sys.exit(2)
