"""Tests for plugins module."""

from __future__ import annotations

from typing import Any

from ununseptium.plugins.base import Plugin, PluginMetadata, PluginState, PluginType
from ununseptium.plugins.loader import PluginLoader, PluginRegistry


# Example plugin for testing
class MockPlugin(Plugin):
    """Test plugin implementation."""

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            plugin_type=PluginType.PROCESSOR,
        )

    def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the plugin."""
        self.config = config or {}
        self.set_state(PluginState.INITIALIZED)

    def execute(self, data: Any) -> Any:
        """Execute plugin logic."""
        if isinstance(data, dict):
            return {"processed": True, **data}
        return data

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        self.set_state(PluginState.STOPPED)


class TestPluginRegistry:
    """Test PluginRegistry."""

    def test_registry_creation(self):
        """Test registry instantiation."""
        registry = PluginRegistry()
        assert registry is not None
        assert len(registry) == 0

    def test_register_plugin(self):
        """Test plugin registration."""
        registry = PluginRegistry()
        
        name = registry.register(MockPlugin)
        
        assert name == "test_plugin"
        assert len(registry) == 1

    def test_get_plugin(self):
        """Test getting plugin instance."""
        registry = PluginRegistry()
        registry.register(MockPlugin)
        
        plugin = registry.get("test_plugin")
        
        assert plugin is not None
        assert isinstance(plugin, MockPlugin)
        assert plugin.state == PluginState.LOADED

    def test_get_nonexistent_plugin(self):
        """Test getting non-existent plugin."""
        registry = PluginRegistry()
        
        plugin = registry.get("nonexistent")
        
        assert plugin is None

    def test_get_metadata(self):
        """Test getting plugin metadata."""
        registry = PluginRegistry()
        registry.register(MockPlugin)
        
        metadata = registry.get_metadata("test_plugin")
        
        assert metadata is not None
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"

    def test_list_plugins(self):
        """Test listing plugins."""
        registry = PluginRegistry()
        registry.register(MockPlugin)
        
        plugins = registry.list_plugins()
        
        assert len(plugins) == 1
        assert plugins[0].name == "test_plugin"

    def test_list_plugins_filtered(self):
        """Test listing plugins with filter."""
        registry = PluginRegistry()
        registry.register(MockPlugin)
        
        processors = registry.list_plugins(plugin_type=PluginType.PROCESSOR)
        analyzers = registry.list_plugins(plugin_type=PluginType.DETECTOR)
        
        assert len(processors) == 1
        assert len(analyzers) == 0

    def test_unregister_plugin(self):
        """Test unregistering plugin."""
        registry = PluginRegistry()
        registry.register(MockPlugin)
        
        success = registry.unregister("test_plugin")
        
        assert success is True
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent plugin."""
        registry = PluginRegistry()
        
        success = registry.unregister("nonexistent")
        
        assert success is False


class TestPluginLoader:
    """Test PluginLoader."""

    def test_loader_creation(self):
        """Test loader instantiation."""
        loader = PluginLoader()
        assert loader is not None
        assert loader.registry is not None

    def test_loader_with_custom_registry(self):
        """Test loader with custom registry."""
        registry = PluginRegistry()
        loader = PluginLoader(registry=registry)
        
        assert loader.registry is registry

    def test_initialize_all(self):
        """Test initializing all plugins."""
        loader = PluginLoader()
        loader.registry.register(MockPlugin)
        
        configs = {"test_plugin": {"key": "value"}}
        loader.initialize_all(configs)
        
        plugin = loader.registry.get("test_plugin")
        assert plugin is not None
        assert plugin.state == PluginState.INITIALIZED
        assert hasattr(plugin, "config")
        assert plugin.config["key"] == "value"

    def test_shutdown_all(self):
        """Test shutting down all plugins."""
        loader = PluginLoader()
        loader.registry.register(MockPlugin)
        loader.initialize_all()
        
        loader.shutdown_all()
        
        plugin = loader.registry.get("test_plugin")
        assert plugin is not None
        assert plugin.state == PluginState.STOPPED


class TestPluginExecution:
    """Test plugin execution."""

    def test_plugin_execute(self):
        """Test plugin execution."""
        plugin = MockPlugin()
        plugin.initialize({})
        
        result = plugin.execute({"input": "data"})
        
        assert result["processed"] is True
        assert result["input"] == "data"

    def test_plugin_lifecycle(self):
        """Test complete plugin lifecycle."""
        plugin = MockPlugin()
        
        # Initial state
        assert plugin.state == PluginState.UNLOADED
        
        # Initialize
        plugin.initialize({"setting": "value"})
        assert plugin.state == PluginState.INITIALIZED
        
        # Execute
        result = plugin.execute({"test": "data"})
        assert result["processed"] is True
        
        # Shutdown
        plugin.shutdown()
        assert plugin.state == PluginState.STOPPED
