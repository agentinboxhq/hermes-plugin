"""Loads the plugin as Hermes does, as a package, with Hermes on sys.path (`HERMES_AGENT_DIR`, else the local install),
and registers its platform the way `ctx.register_platform` does."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

# Hermes writes runtime status and locks under its home: never the real one in a test.
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="agentinbox-hermes-test-")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "agentinbox_plugin"

for candidate in (os.environ.get("HERMES_AGENT_DIR"), Path.home() / ".hermes" / "hermes-agent"):
    if candidate and Path(candidate).joinpath("gateway").is_dir():
        sys.path.insert(0, str(candidate))
        break

if PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(PACKAGE, ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = plugin
    spec.loader.exec_module(plugin)


class FakeContext:
    """What `register(ctx)` gets from Hermes, as far as the plugin uses it."""

    def __init__(self):
        self.platforms = []
        self.cli_commands = []
        self.tools = {}
        self.hooks = {}

    def register_platform(self, name, label, adapter_factory, check_fn, validate_config=None, required_env=None,
                          install_hint="", **entry_kwargs):
        from gateway.platform_registry import PlatformEntry, platform_registry

        entry = PlatformEntry(name=name, label=label, adapter_factory=adapter_factory, check_fn=check_fn,
                              validate_config=validate_config, required_env=required_env or [],
                              install_hint=install_hint, source="plugin", plugin_name="agentinbox", **entry_kwargs)
        self.platforms.append(entry)
        if not platform_registry.is_registered(name):
            platform_registry.register(entry)

    def register_cli_command(self, **kwargs):
        self.cli_commands.append(kwargs)

    def register_tool(self, name, toolset, schema, handler, check_fn=None, requires_env=None, is_async=False,
                      description="", emoji="", override=False):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, "check_fn": check_fn,
                            "requires_env": requires_env, "is_async": is_async, "description": description}

    def register_hook(self, hook_name, callback):
        self.hooks.setdefault(hook_name, []).append(callback)

    def session_surface(self):
        return ""

    def current_cron_execution(self):
        return None


REGISTERED = FakeContext()
sys.modules[PACKAGE].register(REGISTERED)
