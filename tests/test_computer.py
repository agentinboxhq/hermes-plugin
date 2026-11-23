import sys
from types import SimpleNamespace

import pytest

import agentinbox_plugin.computer as computer

PERSONS_TERMINAL = {"PATH": "/usr/bin:/bin", "HOME": "/Users/ada", "TERM": "xterm-256color"}


def test_hermess_own_markers_dont_make_the_person_at_the_terminal_an_agent(monkeypatch):
    # Hermes's CLI adds both to its own environment as it starts, before `hermes agentinbox setup` runs.
    monkeypatch.setenv("AI_AGENT", "hermes-agent")
    monkeypatch.setenv("HERMES_AGENT", "true")
    monkeypatch.setattr(computer, "started_environment", lambda: dict(PERSONS_TERMINAL))

    assert not computer.agent_runs_this()


@pytest.mark.parametrize("markers", [
    {"AI_AGENT": "hermes-agent", "HERMES_AGENT": "true"},  # what Hermes's terminal tool exports to each command
    {"AI_AGENT": "pi"},
    {"CLAUDE_CODE_CHILD_SESSION": "1"},
    {"CODEX_THREAD_ID": "thread_1"},
    {"OPENCLAW_SHELL": "exec"},
])
def test_an_agents_command_is_known_by_the_environment_it_started_with(monkeypatch, markers):
    monkeypatch.setattr(computer, "started_environment", lambda: {**PERSONS_TERMINAL, **markers})

    assert computer.agent_runs_this()


def test_without_the_environment_it_started_with_hermes_counts_as_an_agent(monkeypatch):
    monkeypatch.setenv("AI_AGENT", "hermes-agent")
    monkeypatch.setattr(computer, "started_environment", lambda: None)

    assert computer.agent_runs_this()


def test_the_environment_it_started_with_leaves_out_what_was_set_since(monkeypatch):
    pytest.importorskip("psutil")
    monkeypatch.setenv("AGENTINBOX_SET_SINCE", "1")

    started = computer.started_environment()

    assert started is not None and started.get("PATH")
    assert "AGENTINBOX_SET_SINCE" not in started


def test_an_environment_written_over_by_a_process_title_tells_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: SimpleNamespace(
        environ=lambda: {"hermes": ""})))

    assert computer.started_environment() is None
