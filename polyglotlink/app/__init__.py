"""
PolyglotLink Application

Main application module providing CLI and server functionality.
"""

from polyglotlink.app.cli import main
from polyglotlink.app.server import PolyglotLinkServer, create_app, run_server

__all__ = [
    "main",
    "PolyglotLinkServer",
    "create_app",
    "run_server",
]
