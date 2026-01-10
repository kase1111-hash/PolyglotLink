"""
PolyglotLink Main Entry Point

This module provides the main entry point for running PolyglotLink.
"""

import sys

from polyglotlink.app.cli import main


if __name__ == "__main__":
    sys.exit(main())
