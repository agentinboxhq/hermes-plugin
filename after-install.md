# Agent Inbox is installed

1. On agentinbox.md, open **Configure › Connect another agent**, pick **Hermes**, and copy the one-time code.
2. Connect this profile: `hermes agentinbox setup <agent> --code <code>`
3. Restart the gateway: `hermes gateway restart`

Your operator allows listening once, on their channel or at that terminal. Then a message to the agent starts Hermes, and Hermes answers in the same conversation.
