from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

import agentinbox_plugin.cli as cli
from agentinbox_plugin.api import AgentInboxApi
from agentinbox_plugin.settings import Settings
from agentinbox_plugin.state import MemoryStore

from conftest import REGISTERED
from fake_api import FakeApi

CONNECTED = {"apiKey": "sk_new", "environmentId": "env_1",
             "agent": {"identifier": "atlas-x1y2", "name": "Atlas", "handle": "atlas"}, "host": "hermes"}


@pytest.fixture
def fake(monkeypatch):
    for name in ("AGENTINBOX_SECRET_KEY", "AGENTINBOX_AGENT", "AGENTINBOX_API_URL"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeApi()
    fake.codes["BCDF-GHJK"] = CONNECTED
    monkeypatch.setattr(cli, "AgentInboxApi",
                        lambda url, key="", agent="human-relay": AgentInboxApi(url, key, agent,
                                                                              transport=fake.transport()))
    monkeypatch.setattr(cli, "computer_name", lambda: "MacBook")
    return fake


class Env:
    def __init__(self):
        self.values: Dict[str, Optional[str]] = {}

    def __call__(self, name: str, value: Optional[str]) -> None:
        self.values[name] = value


def args(**overrides):
    return SimpleNamespace(**{"agent": "atlas", "code": "bcdf ghjk", "api_url": None, "yes": False, **overrides})


def test_setup_trades_the_code_for_the_key_and_allows_at_the_terminal(fake):
    env, store, asked = Env(), MemoryStore(), []
    code = cli.run_setup(args(), ask=lambda question, default: asked.append(question) or True,
                         at_terminal=lambda: True, save_env=env, store=store)
    assert code == 0
    assert env.values == {"AGENTINBOX_SECRET_KEY": "sk_new", "AGENTINBOX_AGENT": "atlas-x1y2",
                          "AGENTINBOX_API_URL": None}
    assert asked == ["Listen for messages on MacBook, so that one from you, or from a contact it listens to, starts "
                     "Hermes?"]
    connection = Settings(secret_key="sk_new", agent="atlas-x1y2").connection
    assert store.current.connection == connection
    assert store.current.allowed_at


def test_an_agents_setup_leaves_the_allow_to_the_operators_channel(fake):
    store = MemoryStore()
    code = cli.run_setup(args(), ask=lambda *_: pytest.fail("nobody is at the terminal"),
                         at_terminal=lambda: False, save_env=Env(), store=store)
    assert code == 0
    assert store.current.allowed_at is None


def test_yes_at_the_operators_terminal_asks_nothing_and_listens_here(fake):
    # The commands from agentinbox.md are pasted together: a question would take the next one as its answer.
    fake.holder = {"computerId": "cmp_other_computer", "computerName": "Mac mini", "host": "codex"}
    store = MemoryStore()
    code = cli.run_setup(args(yes=True), ask=lambda *_: pytest.fail("--yes asks nothing"), at_terminal=lambda: True,
                         save_env=Env(), store=store)
    assert code == 0
    assert store.current.allowed_at
    assert store.current.take_over is True


def test_yes_in_an_agents_run_leaves_the_allow_to_the_operators_channel(fake):
    store = MemoryStore()
    cli.run_setup(args(yes=True), ask=lambda *_: pytest.fail("nobody is at the terminal"), at_terminal=lambda: False,
                  save_env=Env(), store=store)
    assert store.current.allowed_at is None


def test_moving_listening_here_says_so_and_takes_it_over(fake):
    fake.holder = {"computerId": "cmp_other_computer", "computerName": "Mac mini", "host": "codex"}
    store, asked = MemoryStore(), []
    cli.run_setup(args(), ask=lambda question, default: asked.append(question) or True, at_terminal=lambda: True,
                  save_env=Env(), store=store)
    assert asked[0].endswith("It listens on Mac mini (Codex) now; this moves it here.")
    assert store.current.take_over is True


def test_it_connects_as_hermess_plugin(fake):
    # agentinbox.md then lists the host as "Hermes · Plugin", not as the CLI.
    cli.run_setup(args(), ask=lambda *_: True, at_terminal=lambda: False, save_env=Env(), store=MemoryStore())
    assert fake.callers and set(fake.callers) == {("plugin", "hermes")}


def test_a_wrong_code_changes_nothing(fake):
    env = Env()
    assert cli.run_setup(args(code="ZZZZ-ZZZZ"), ask=lambda *_: True, at_terminal=lambda: True, save_env=env,
                         store=MemoryStore()) == 1
    assert env.values == {}


@pytest.mark.parametrize("agent,code", [("Atlas@agentinbox.md", "BCDF-GHJK"), ("atlas", "bcdfghjk")])
def test_the_agent_and_code_read_as_agentinbox_md_shows_them(agent, code):
    assert cli.read_agent(agent) == "atlas"
    assert cli.read_code(code) == "BCDF-GHJK"


@pytest.mark.parametrize("code", ["", "BCDF-GHJ", "BCDF-GHJK1", "12345678"])
def test_a_code_that_isnt_eight_letters_is_refused(code):
    with pytest.raises(cli.SetupError):
        cli.read_code(code)


def test_a_profile_connected_already_needs_yes_without_a_terminal(fake, monkeypatch):
    monkeypatch.setenv("AGENTINBOX_SECRET_KEY", "sk_old")
    env = Env()
    assert cli.run_setup(args(), ask=lambda *_: True, at_terminal=lambda: False, save_env=env,
                         store=MemoryStore()) == 1
    assert env.values == {}
    assert cli.run_setup(args(yes=True), ask=lambda *_: True, at_terminal=lambda: False, save_env=env,
                         store=MemoryStore()) == 0
    assert env.values["AGENTINBOX_SECRET_KEY"] == "sk_new"


def test_a_local_api_is_kept(fake):
    env = Env()
    cli.run_setup(args(api_url="http://localhost:4300/api/"), ask=lambda *_: True, at_terminal=lambda: False,
                  save_env=env, store=MemoryStore())
    assert env.values["AGENTINBOX_API_URL"] == "http://localhost:4300/api"


def test_the_plugin_registers_its_platform_and_commands():
    entry = REGISTERED.platforms[0]
    assert entry.name == "agentinbox"
    assert entry.label == "Agent Inbox"
    assert entry.max_message_length == 4000
    assert entry.display_tier == "minimal"
    assert entry.allow_update_command is False
    assert "Agent Inbox" in entry.platform_hint
    assert entry.cron_deliver_env_var == "AGENTINBOX_HOME_CHANNEL"
    assert len(entry.env_overrides) == 1
    assert REGISTERED.cli_commands[0]["name"] == "agentinbox"
