import asyncio
from typing import Any, Dict, List

import pytest

from agentinbox_plugin.api import AgentInboxApi
from agentinbox_plugin.listener import Listener, StopForGood, Timing
from agentinbox_plugin.settings import Settings
from agentinbox_plugin.state import MemoryStore, State

from fake_api import FakeApi, message

SETTINGS = Settings(secret_key="sk_test", agent="human-relay", api_url="https://api.test")
FAST = Timing(wait_seconds=1, paused_retry=0.01, retry=(0.01,), allow_poll=0.01)


class Harness:
    def __init__(self, fake: FakeApi, state: State = None, answered: List[str] = None):
        self.fake = fake
        self.store = MemoryStore(state)
        self.started: List[Dict[str, Any]] = []
        self.answers: List[Dict[str, Any]] = []
        self.answered = set(answered or [])
        self.api = AgentInboxApi(SETTINGS.api_url, SETTINGS.secret_key, SETTINGS.agent, transport=fake.transport())
        self.listener = Listener(api=self.api, store=self.store, settings=SETTINGS, computer_name="MacBook",
                                 on_message=self.on_message, on_answer=self.on_answer, timing=FAST)

    async def on_message(self, found: Dict[str, Any]) -> None:
        self.started.append(found)

    async def on_answer(self, found: Dict[str, Any]) -> bool:
        if found.get("answersMessageId") in self.answered:
            self.answers.append(found)
            return True
        return False

    async def run_until(self, done, timeout: float = 2.0):
        task = asyncio.create_task(self.listener.run())
        try:
            for _ in range(int(timeout / 0.01)):
                if task.done():
                    return task.result()
                if done():
                    return None
                await asyncio.sleep(0.01)
            raise AssertionError("timed out")
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


def allowed_state(**extra) -> State:
    return State(computer_id="cmp_test_computer_1", connection=SETTINGS.connection, allowed_at="2026-11-05T10:00:00Z",
                 **extra)


async def test_asks_the_operator_once_and_listens_after_allow():
    fake = FakeApi()
    harness = Harness(fake)
    task = asyncio.create_task(harness.listener.run())
    try:
        for _ in range(200):
            if fake.sent:
                break
            await asyncio.sleep(0.01)
        ask = fake.last("approve")
        assert ask["to"] == "op_ada"
        assert ask["card"]["title"] == "Listen for messages on MacBook, and start Hermes?"
        assert ask["card"]["approveLabel"] == "Allow"
        assert fake.holder is None  # nothing taken before the Allow

        fake.answer(ask["id"], "approved")
        fake.deliver(message("m1"))
        for _ in range(200):
            if harness.started:
                break
            await asyncio.sleep(0.01)
        assert [each["id"] for each in harness.started] == ["m1"]
        assert fake.holder["computerName"] == "MacBook"
        assert harness.store.current.allowed_at
        assert harness.store.current.asked is None
    finally:
        task.cancel()


async def test_a_deny_stops_for_good_and_takes_nothing():
    fake = FakeApi()
    harness = Harness(fake)
    task = asyncio.create_task(harness.listener.run())
    for _ in range(200):
        if fake.sent:
            break
        await asyncio.sleep(0.01)
    fake.answer(fake.last("approve")["id"], "denied")
    with pytest.raises(StopForGood) as stop:
        await asyncio.wait_for(task, 2)
    assert stop.value.reason == "denied"
    assert fake.holder is None
    assert harness.store.current.allowed_at is None


async def test_words_instead_of_allow_are_not_an_allow():
    fake = FakeApi()
    harness = Harness(fake)
    task = asyncio.create_task(harness.listener.run())
    for _ in range(200):
        if fake.sent:
            break
        await asyncio.sleep(0.01)
    fake.answer(fake.last("approve")["id"], "answered", text="sure, go ahead")
    with pytest.raises(StopForGood) as stop:
        await asyncio.wait_for(task, 2)
    assert stop.value.reason == "denied"


async def test_an_allow_that_moves_listening_here_takes_the_lease_over():
    fake = FakeApi()
    fake.holder = {"computerId": "cmp_other_computer", "computerName": "Mac mini", "host": "claude-code",
                   "project": "~/projects/app"}
    harness = Harness(fake)
    task = asyncio.create_task(harness.listener.run())
    try:
        for _ in range(200):
            if fake.sent:
                break
            await asyncio.sleep(0.01)
        ask = fake.last("approve")
        assert "It is listening on Mac mini now (Claude Code, in ~/projects/app); this moves it here." in \
            ask["card"]["body"]
        fake.answer(ask["id"], "approved")
        for _ in range(200):
            if fake.holder["computerName"] == "MacBook":
                break
            await asyncio.sleep(0.01)
        assert fake.holder["host"] == "hermes"
        assert harness.store.current.take_over is False  # only the first lease takes it over
    finally:
        task.cancel()


async def test_already_allowed_listens_without_asking_and_hands_each_message_once():
    fake = FakeApi()
    harness = Harness(fake, allowed_state())
    fake.deliver(message("m1"), message("m2", conversation="conv_2", operator=False, name="Ben", sender="sub_ben"))
    await harness.run_until(lambda: len(harness.started) == 2)
    assert not fake.sent
    assert harness.store.current.cursor == "m2"

    # The same messages again, as after a restart that lost the cursor: none is handled twice.
    harness.store.current.cursor = None
    again = Harness(fake, harness.store.current)
    await again.run_until(lambda: fake.calls.count("GET /v1/listen/messages") > 6)
    assert again.started == []


async def test_answers_go_to_what_waits_for_them_and_never_start_hermes():
    fake = FakeApi()
    harness = Harness(fake, allowed_state(own=["msg_old"]), answered=["msg_card"])
    fake.deliver(
        message("m1", answers="msg_card", answer={"status": "approved"}),
        message("m2", answers="msg_old", answer={"status": "approved"}),
        message("m3", answers="msg_agents_own_question", text="tomorrow"),
    )
    await harness.run_until(lambda: harness.store.current.cursor == "m3")
    assert [each["id"] for each in harness.answers] == ["m1"]
    # An answer to a question the agent asked itself (with the CLI or MCP) joins the run, as `listen` does.
    assert [each["id"] for each in harness.started] == ["m3"]


async def test_another_computer_taking_over_stops_it_for_good():
    fake = FakeApi()
    harness = Harness(fake, allowed_state())
    task = asyncio.create_task(harness.listener.run())
    for _ in range(200):
        if fake.holder:
            break
        await asyncio.sleep(0.01)
    fake.holder = {"computerId": "cmp_other_computer", "computerName": "Mac mini", "host": "codex"}
    with pytest.raises(StopForGood) as stop:
        await asyncio.wait_for(task, 2)
    assert stop.value.reason == "listening_elsewhere"
    assert stop.value.message == ("It is listening on Mac mini (Codex) now, so this gateway stopped. Restart the "
                                  "gateway to ask your operator to move it back here.")
    assert harness.listener.holding is False
    assert harness.store.current.allowed_at is None


async def test_a_restart_after_a_move_asks_the_operator_and_moves_it_back():
    fake = FakeApi()
    harness = Harness(fake, allowed_state())
    task = asyncio.create_task(harness.listener.run())
    for _ in range(200):
        if fake.holder:
            break
        await asyncio.sleep(0.01)
    fake.holder = {"computerId": "cmp_other_computer", "computerName": "Mac mini", "host": "codex"}
    with pytest.raises(StopForGood):
        await asyncio.wait_for(task, 2)

    # The gateway restarts on the state the stop saved.
    again = Harness(fake, harness.store.current)
    task = asyncio.create_task(again.listener.run())
    try:
        for _ in range(200):
            if fake.sent:
                break
            await asyncio.sleep(0.01)
        ask = fake.last("approve")
        assert "It is listening on Mac mini now (Codex); this moves it here." in ask["card"]["body"]
        assert fake.holder["computerName"] == "Mac mini"  # nothing taken before the Allow
        fake.answer(ask["id"], "approved")
        for _ in range(200):
            if fake.holder["computerName"] == "MacBook":
                break
            await asyncio.sleep(0.01)
        assert fake.holder["host"] == "hermes"
    finally:
        task.cancel()


async def test_a_lapsed_lease_is_taken_again():
    fake = FakeApi()
    harness = Harness(fake, allowed_state())
    task = asyncio.create_task(harness.listener.run())
    try:
        for _ in range(200):
            if fake.holder:
                break
            await asyncio.sleep(0.01)
        fake.holder = None  # it lapsed, as when the computer slept
        for _ in range(200):
            if fake.holder:
                break
            await asyncio.sleep(0.01)
        assert fake.holder["computerName"] == "MacBook"
        assert fake.calls.count("POST /v1/listen/lease") == 2
    finally:
        task.cancel()


async def test_a_changed_key_stops_for_good():
    fake = FakeApi()
    fake.fail("GET /v1/operator", 401, {"message": "Unauthorized"})
    harness = Harness(fake, allowed_state())
    with pytest.raises(StopForGood) as stop:
        await asyncio.wait_for(harness.listener.run(), 2)
    assert stop.value.reason == "key_changed"


async def test_an_outage_is_waited_out():
    fake = FakeApi()
    fake.fail("POST /v1/listen/lease", 503, times=2)
    harness = Harness(fake, allowed_state())
    fake.deliver(message("m1"))
    await harness.run_until(lambda: harness.started)
    assert fake.calls.count("POST /v1/listen/lease") == 3


async def test_a_paused_agent_is_tried_again_later():
    fake = FakeApi()
    fake.paused = True
    harness = Harness(fake, allowed_state())
    task = asyncio.create_task(harness.listener.run())
    try:
        for _ in range(200):
            if fake.calls.count("POST /v1/listen/lease") >= 2:
                break
            await asyncio.sleep(0.01)
        assert fake.holder is None
        fake.paused = False
        fake.deliver(message("m1"))
        for _ in range(200):
            if harness.started:
                break
            await asyncio.sleep(0.01)
        assert harness.started
    finally:
        task.cancel()


async def test_a_cursor_the_api_never_handed_out_starts_from_now():
    fake = FakeApi()
    fake.fail("GET /v1/listen/messages", 400, {"code": "invalid_listen_cursor", "message": "Bad cursor"})
    harness = Harness(fake, allowed_state(cursor="someone_elses"))
    fake.deliver(message("m1"))
    await harness.run_until(lambda: harness.started)
    assert harness.store.current.cursor == "m1"


async def test_no_operator_stops_for_good():
    fake = FakeApi()
    fake.operator = None
    harness = Harness(fake)
    with pytest.raises(StopForGood) as stop:
        await asyncio.wait_for(harness.listener.run(), 2)
    assert stop.value.reason == "no_operator"


async def test_close_lets_go_of_the_lease_it_holds():
    fake = FakeApi()
    harness = Harness(fake, allowed_state())
    await harness.run_until(lambda: fake.holder)
    await harness.listener.close()
    assert fake.holder is None
    assert fake.released == 1
