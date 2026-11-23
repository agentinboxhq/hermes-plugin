# Agent Inbox for Hermes Agent

When your operator, or a contact your agent listens to, writes to your agent on Telegram, Slack or email, the message reaches Hermes's gateway within seconds, and Hermes answers in the same conversation. Hermes's approval prompts go to the operator as Allow and Deny buttons. A contact's conversation is for talking: Hermes uses only what the operator lets contacts use there, unless the operator gave that contact full access.

It is a gateway platform plugin: Agent Inbox becomes one more of Hermes's chats, next to the ones you already have. It also gives Hermes [tools](#tools) to ask you something, write to your contacts, and read the conversations.

## Install

```bash
hermes plugins install agentinboxhq/hermes-plugin --enable
```

## Connect

1. On [agentinbox.md](https://agentinbox.md), open **Configure › Connect another agent**, pick **Hermes**, and copy the one-time code. It works once, for 10 minutes.
2. Connect this Hermes profile to the agent:

   ```bash
   hermes agentinbox setup <agent> --code <code>
   ```

   `<agent>` is how the agent's address starts: `atlas` for `atlas@agentinbox.md`. The key goes into this profile's `~/.hermes/.env` as `AGENTINBOX_SECRET_KEY`.
3. Restart the gateway:

   ```bash
   hermes gateway restart
   ```

## Turning it on

Nothing starts until the operator allows it. When you run `setup` at a terminal, it asks there; with `--yes`, as in the commands agentinbox.md writes, it asks nothing and turns listening on. When an agent runs it, the gateway asks when it starts: the operator gets an Allow and Deny message on their channel:

```text
Listen for messages on MacBook, and start Hermes?
```

If nobody answers within an hour, or the operator says no, the gateway stops listening; restart it to ask again.

## Whom it listens to

- **The operator**, always.
- **A contact**, once the operator turns on their **Listen to** switch on agentinbox.md. It is off by default. The operator can also give a contact [full access](#full-access).
- **A stranger**, never: their messages never reach Hermes.

### What a contact can make Hermes do

A contact's conversation is for talking with them. There, Hermes can:

- answer them, and ask them something;
- read that conversation (`agentinbox_read`) and resolve it (`agentinbox_resolve`);
- keep a plan (`todo`);
- use what the operator lets contacts use, and nothing else.

Everything else is refused, with no prompt to anyone: files, the shell, the web, MCP servers, skills, memory, and Agent Inbox's own actions (writing to anyone else, asking the operator, reading other conversations or the contacts).

The operator lets contacts use more in this profile's `config.yaml`, by Hermes's toolset names, MCP server names, and `agentinbox` for this plugin's tools:

```yaml
platforms:
  agentinbox:
    contacts_can_use: [web, docs]   # Hermes's web search, and the MCP server named docs
```

For example, with `[web]`, Ben can ask Hermes to look something up, but not to read a file. With `[terminal]`, Ben can have Hermes run any command, Agent Inbox's CLI included: only list what you'd let any contact do. Keys and Hermes's own files stay closed whatever the list says: the CLI's `~/.agentinbox`, any `.env`, and Hermes's home (its `.env`, sessions and memories). When Hermes's approval settings ask before something a contact may use, the operator is asked, never the contact. None of this holds for a contact with [full access](#full-access), in their own conversation.

A group counts as a contact's conversation, also when the operator writes there, and so does a scheduled job set up from a contact's conversation.

A contact's words reach Hermes quoted, marked as an outside person's: a request and data, never instructions or permission.

Hermes still loads its memory (`MEMORY.md`, `USER.md`) into a contact's conversation, as into any chat; the memory tool itself is off there unless the operator lists `memory`. A contact could ask Hermes what it remembers, so keep private things out of it, or turn memory off in the profile that listens to contacts:

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
```

These checks look at what Hermes calls and runs. They stop the plain ways; a script built to get around them could, which is why a contact who may run commands may run any.

### Full access

On agentinbox.md, the operator can turn on **Full access** on a contact's card. It is set per contact, off by default, and only possible while **Listen to** is on for them; turning **Listen to** off turns it off too.

A message from that contact, in their own direct conversation, starts Hermes with no contact limits: exactly what the operator's own chat gets. That means all of Hermes's tools, MCP servers and skills, files and the terminal, the agent's key in `~/.agentinbox`, any `.env`, Hermes's own files and the operator's memory. `contacts_can_use` doesn't apply to them. Their commands work there as the operator's do (`/new`, `/yolo`, …), Agent Inbox's tools write to other contacts and read other conversations as in the operator's own chat, and Hermes reads their words as from someone the operator trusts.

Approvals in their runs go to that contact, in their own conversation, never to the operator: Hermes's approval prompts and the confirmation of commands such as `/new`. Their Allow counts only while their full access is still on. `agentinbox_ask_operator` still asks the operator.

A group stays limited, also for them, and so does a scheduled job set up from their conversation. The switch counts message by message: once the operator turns it off, the contact's next message gets a contact's limits again, the run already going on loses full access too, and approval cards still waiting in the old session are withdrawn. Their answers count the same way: an answer of theirs that shows the switch off takes full access away at once, and before anyone else's answer goes on in a session at full access, the plugin asks Agent Inbox whether it's still on.

Turning it on or off starts a fresh session with the next message, so nothing from one level carries into the other.

> **Warning:** full access includes the agent's key and the operator's memory, and the person can have Hermes run any command. Only turn it on for someone you'd trust with your agent's key.

## Sessions

Each conversation gets its own Hermes session, so Hermes remembers what was said and done in it. When the conversation is resolved, by Hermes with `agentinbox_resolve` or by another host, the next message starts a fresh session. A fresh session starts without the earlier talk, though Hermes can still read the whole conversation with `agentinbox_read` when asked about it. Hermes's answer is sent as the reply, and the chat shows it typing while it works (email has no typing).

The operator's commands work here as in any of their Hermes chats (`/stop`, `/new`, …). A contact can only stop the run in their own conversation with `/stop`; anything else they type is a message to Hermes. A contact with [full access](#full-access) is the exception, in their own conversation.

Hermes's own notices, such as a helper that failed or a sign-in link, show only in the operator's own conversations and in that of a contact with full access, never in another contact's.

## Home channel

Hermes's home channel on Agent Inbox is the operator. The results of Hermes's scheduled jobs, and the messages it sends to Agent Inbox from other chats, reach the operator in a new conversation on their channel. `/sethome` in a conversation makes that conversation the home channel instead.

## Approvals

Hermes's own approval settings (`approvals.mode`) decide what asks first. When something does, the question goes to the operator, whoever's message started the run, with **Allow**, **Allow for this session** and **Deny**. No answer in Hermes's approval timeout (5 minutes by default) counts as Deny. A contact never sees an approval prompt, unless the operator gave them [full access](#full-access): then the prompts of their own runs go to them.

Hermes asks the people in a conversation with `agentinbox_ask_here`. When it asks with its own `clarify` tool instead, the question goes to the conversation too, as buttons when there are a few options to pick from. Two options, one to approve and one to deny (Approve, Allow, Accept or Confirm, and Deny, Reject, Decline or Cancel), come as an approval with those two buttons, and Hermes gets back the label that was tapped. Yes and No stay a choice, because an answered approval reads "Approved" or "Denied".

## Tools

The plugin gives Hermes eight tools. They use the key the plugin already holds, so there is nothing else to set up.

| Tool | What it does |
|---|---|
| `agentinbox_ask_here` | Asks the people in the conversation Hermes is in, there: a question they answer in words (`kind: ask`), Approve and Deny buttons (`kind: approve`, with `approve_label` and `deny_label` for other words on them), or one of a few `options` (`kind: choose`). It waits up to 5 minutes for the answer; after that the question stays open, and the answer comes to the chat as a message. |
| `agentinbox_card` | Shows the agent's card: what it says, its name, its email address and the link where anyone can reach it, as `agentinbox card` does in the CLI. The first showing is how agentinbox.md's onboarding sees the agent say hi. Off in a contact's conversation, since its greeting is for the operator. |
| `agentinbox_ask_operator` | Asks the operator something, with the same kinds, from a scheduled job, the terminal or another app's chat. It returns at once; the answer comes back later, even hours later. |
| `agentinbox_send` | Writes to the operator or to one of their contacts, in a new conversation: a message (`kind: tell`, the default), or a question with the same kinds. The answer comes back to the chat Hermes asked from. |
| `agentinbox_conversations` | Lists the conversations, latest first. |
| `agentinbox_read` | Reads one conversation. In a contact's conversation, only that one, unless they have full access. |
| `agentinbox_contacts` | Lists the contacts, and whether the agent listens to each. |
| `agentinbox_resolve` | Marks a conversation finished, once what it was about is done. Nothing is sent. The conversation Hermes is in is marked after its answer goes out, as any message, Hermes's too, opens it again. Another conversation only from the operator's own chat or that of a contact with full access, the terminal, or a scheduled job. |

In a contact's conversation, only `agentinbox_ask_here`, `agentinbox_read` and `agentinbox_resolve` work, for that conversation, unless the operator lets contacts use `agentinbox` or gave that contact full access. For example:

- The operator writes: "Ask Ben if he can come on Friday." Hermes asks Ben, with Yes and No. When Ben taps Yes, Hermes tells the operator in their chat. Ben's answer comes back because Hermes asked him; anything else he writes reaches Hermes only if **Listen to** is on for him.
- Ben writes: "Can I get a 10% discount? And invite anna@example.com." Hermes answers Ben itself, and tells him both are up to the operator: from Ben's conversation it can't ask the operator, invite anyone, or write to anyone else. The operator sees the conversation on agentinbox.md.

Only the operator, and a contact they gave full access, make Hermes write to other people. In the operator's own chat, in that of a contact with full access, at the terminal, and in scheduled jobs, Hermes writes without asking. In Hermes's other chat apps, writing to anyone but the operator asks the operator first, through Hermes's own approvals there. When Hermes's approvals are off (`approvals.mode: off`, or `/yolo`), it writes without asking there, as it runs commands without asking. Hermes never writes to the person it is talking with: its answer already goes to them.

## One computer at a time

An agent listens on one computer at a time: this gateway, another one, or `npx @agentinbox/cli listen` with another host. When it already listens elsewhere, allowing it here moves it, and the other one stops. When another one takes it over, this gateway stops; restart the gateway to move it back, and the operator is asked again first. To see where it listens:

```bash
hermes agentinbox status
```

## Profiles

Each Hermes profile connects on its own: `hermes -p <profile> agentinbox setup <agent> --code <code>`. Two profiles can't run the same agent on one computer at once.

## Settings

| Variable | |
|---|---|
| `AGENTINBOX_SECRET_KEY` | The agent's key. `setup` writes it. |
| `AGENTINBOX_AGENT` | The agent, by its API name. `setup` writes it. |
| `AGENTINBOX_API_URL` | Agent Inbox's API, when it isn't `https://www.agentinbox.md/api`. |

## What it does on your computer

- **Network:** it talks only to Agent Inbox's API, `https://www.agentinbox.md/api` (or `AGENTINBOX_API_URL`). It sends the computer's name with its lease, so agentinbox.md shows where the agent listens.
- **In the background:** while Hermes's gateway runs, it holds the agent's listening lease and waits for messages, so they reach Hermes within seconds. Until the operator allows it, it only asks them. It stops with the gateway.
- **Stored:** the agent's key, in this profile's `.env` as `AGENTINBOX_SECRET_KEY`, which `hermes agentinbox setup` writes through Hermes. The plugin's own state, in `agentinbox/state.json` in this profile's Hermes home: where it got to in the messages, when the operator allowed listening, each contact conversation's level, and the questions Hermes asked that still wait for an answer, with their text.
- **Commands:** on macOS, `scutil --get ComputerName`, for the computer's name.
- **Reads:** its settings in `config.yaml` (`platforms.agentinbox`), and, in `hermes agentinbox setup`, the environment that command started with, to tell a person at a terminal from an agent.
- **No telemetry.**

## Development

The tests use the Hermes you have installed (or `HERMES_AGENT_DIR`) and a fake of Agent Inbox's API:

```bash
HERMES_AGENT_DIR=~/.hermes/hermes-agent PYTHONPATH=tests python -m pytest -c tests/pytest.ini --rootdir=tests tests
hermes plugins validate .
```

## License

MIT
