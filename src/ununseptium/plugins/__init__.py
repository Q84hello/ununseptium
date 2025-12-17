"""Plugin system for extensibility.

Provides plugin discovery, loading, and management.
"""

from ununseptium.plugins.base import Plugin, PluginMetadata
from ununseptium.plugins.loader import PluginLoader, PluginRegistry

__all__ = [
    "Plugin",
    "PluginLoader",
    "PluginMetadata",
    "PluginRegistry",
]
