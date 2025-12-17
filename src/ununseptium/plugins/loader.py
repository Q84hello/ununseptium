"""Plugin loader and registry.

Provides plugin discovery, loading, and management.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ununseptium.plugins.base import Plugin, PluginMetadata, PluginState, PluginType

if TYPE_CHECKING:
    pass


class PluginRegistry:
    """Registry for managing plugins.

    Example:
        ```python
        from ununseptium.plugins import PluginRegistry

        registry = PluginRegistry()

        # Register plugin
        registry.register(MyPlugin)

        # Get plugin instance
        plugin = registry.get("my_plugin")
        plugin.initialize({"key": "value"})

        # Execute
        result = plugin.execute(data)
        ```
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._plugins: dict[str, type[Plugin]] = {}
        self._instances: dict[str, Plugin] = {}

    def register(self, plugin_class: type[Plugin]) -> str:
        """Register a plugin class.

        Args:
            plugin_class: Plugin class to register.

        Returns:
            Plugin name.
        """
        metadata = plugin_class.metadata()
        name = metadata.name

        self._plugins[name] = plugin_class
        return name

    def get(self, name: str) -> Plugin | None:
        """Get a plugin instance.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        if name in self._instances:
            return self._instances[name]

        plugin_class = self._plugins.get(name)
        if plugin_class:
            instance = plugin_class()
            instance.set_state(PluginState.LOADED)
            self._instances[name] = instance
            return instance

        return None

    def get_metadata(self, name: str) -> PluginMetadata | None:
        """Get plugin metadata.

        Args:
            name: Plugin name.

        Returns:
            PluginMetadata or None.
        """
        plugin_class = self._plugins.get(name)
        if plugin_class:
            return plugin_class.metadata()
        return None

    def list_plugins(
        self,
        *,
        plugin_type: PluginType | None = None,
    ) -> list[PluginMetadata]:
        """List registered plugins.

        Args:
            plugin_type: Filter by type.

        Returns:
            List of plugin metadata.
        """
        plugins = []
        for plugin_class in self._plugins.values():
            metadata = plugin_class.metadata()
            if plugin_type is None or metadata.plugin_type == plugin_type:
                plugins.append(metadata)
        return plugins

    def unregister(self, name: str) -> bool:
        """Unregister a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if unregistered.
        """
        if name in self._instances:
            instance = self._instances[name]
            if instance.state in (PluginState.INITIALIZED, PluginState.RUNNING):
                instance.shutdown()
            del self._instances[name]

        if name in self._plugins:
            del self._plugins[name]
            return True

        return False

    def __len__(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)


class PluginLoader:
    """Load plugins from files and entry points.

    Example:
        ```python
        from ununseptium.plugins import PluginLoader

        loader = PluginLoader()

        # Load from file
        loader.load_file("path/to/plugin.py")

        # Load from directory
        loader.load_directory("plugins/")

        # Get registered plugins
        plugins = loader.registry.list_plugins()
        ```
    """

    def __init__(self, registry: PluginRegistry | None = None) -> None:
        """Initialize the loader.

        Args:
            registry: Plugin registry to use.
        """
        self.registry = registry or PluginRegistry()

    def load_file(self, path: Path | str) -> list[str]:
        """Load plugins from a Python file.

        Args:
            path: Path to plugin file.

        Returns:
            List of loaded plugin names.
        """
        path = Path(path)

        if not path.exists():
            msg = f"Plugin file not found: {path}"
            raise FileNotFoundError(msg)

        if not path.suffix == ".py":
            msg = f"Invalid plugin file: {path}"
            raise ValueError(msg)

        # Load module
        module_name = f"ununseptium_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            msg = f"Could not load plugin: {path}"
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find plugin classes
        loaded: list[str] = []
        for item_name in dir(module):
            item = getattr(module, item_name)

            if (
                isinstance(item, type)
                and issubclass(item, Plugin)
                and item is not Plugin
            ):
                name = self.registry.register(item)
                loaded.append(name)

        return loaded

    def load_directory(self, directory: Path | str) -> list[str]:
        """Load all plugins from a directory.

        Args:
            directory: Directory path.

        Returns:
            List of loaded plugin names.
        """
        directory = Path(directory)

        if not directory.exists():
            msg = f"Plugin directory not found: {directory}"
            raise FileNotFoundError(msg)

        loaded: list[str] = []
        for path in directory.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                names = self.load_file(path)
                loaded.extend(names)
            except (ImportError, ValueError) as e:
                # Log error but continue
                print(f"Warning: Could not load {path}: {e}")

        return loaded

    def load_entry_points(self, group: str = "ununseptium.plugins") -> list[str]:
        """Load plugins from entry points.

        Args:
            group: Entry point group name.

        Returns:
            List of loaded plugin names.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            from importlib_metadata import entry_points  # type: ignore[import-untyped,no-redef]

        loaded: list[str] = []

        try:
            eps = entry_points(group=group)
        except TypeError:
            # Python < 3.10
            all_eps = entry_points()
            eps = all_eps.get(group, [])

        for ep in eps:
            try:
                plugin_class = ep.load()
                if isinstance(plugin_class, type) and issubclass(plugin_class, Plugin):
                    name = self.registry.register(plugin_class)
                    loaded.append(name)
            except Exception as e:
                print(f"Warning: Could not load entry point {ep.name}: {e}")

        return loaded

    def initialize_all(self, configs: dict[str, dict[str, Any]] | None = None) -> None:
        """Initialize all loaded plugins.

        Args:
            configs: Plugin configurations keyed by name.
        """
        configs = configs or {}

        for metadata in self.registry.list_plugins():
            plugin = self.registry.get(metadata.name)
            if plugin:
                config = configs.get(metadata.name, {})
                try:
                    plugin.initialize(config)
                    plugin.set_state(PluginState.INITIALIZED)
                except Exception as e:
                    plugin.set_error(e)

    def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        for metadata in self.registry.list_plugins():
            plugin = self.registry.get(metadata.name)
            if plugin and plugin.state in (
                PluginState.INITIALIZED,
                PluginState.RUNNING,
            ):
                try:
                    plugin.shutdown()
                    plugin.set_state(PluginState.STOPPED)
                except Exception as e:
                    plugin.set_error(e)
