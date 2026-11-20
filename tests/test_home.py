from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig

import agentinbox_plugin as plugin

PLATFORM = Platform("agentinbox")


def test_the_operator_is_agent_inboxs_home_channel(monkeypatch):
    monkeypatch.delenv("AGENTINBOX_HOME_CHANNEL", raising=False)
    config = GatewayConfig()

    plugin._home_is_the_operator(config)

    home = config.platforms[PLATFORM].home_channel
    assert (home.chat_id, home.name) == ("operator", "Your operator")
    # Hermes's "gateway is back online" would be a new conversation with the operator at every restart.
    assert config.platforms[PLATFORM].gateway_restart_notification is False


def test_a_home_channel_picked_with_sethome_stays(monkeypatch):
    monkeypatch.delenv("AGENTINBOX_HOME_CHANNEL", raising=False)
    config = GatewayConfig()
    config.platforms[PLATFORM] = PlatformConfig(home_channel=HomeChannel(platform=PLATFORM, chat_id="conv_9",
                                                                          name="Home"))

    plugin._home_is_the_operator(config)

    assert config.platforms[PLATFORM].home_channel.chat_id == "conv_9"
    assert config.platforms[PLATFORM].gateway_restart_notification is True


def test_a_home_channel_the_profile_names_is_taken(monkeypatch):
    monkeypatch.setenv("AGENTINBOX_HOME_CHANNEL", "conv_7")
    config = GatewayConfig()

    plugin._home_is_the_operator(config)

    assert config.platforms[PLATFORM].home_channel.chat_id == "conv_7"
    assert config.platforms[PLATFORM].gateway_restart_notification is True


def test_hermes_loads_the_home_channel_with_its_config(monkeypatch):
    # Hermes's own env pass, as the gateway and cron run it: the platform is enabled and has its home.
    from gateway.config_env import _apply_env_overrides

    monkeypatch.setenv("AGENTINBOX_SECRET_KEY", "sk_test")
    monkeypatch.delenv("AGENTINBOX_HOME_CHANNEL", raising=False)
    config = GatewayConfig()

    _apply_env_overrides(config)

    assert config.platforms[PLATFORM].enabled is True
    assert config.get_home_channel(PLATFORM).chat_id == "operator"
