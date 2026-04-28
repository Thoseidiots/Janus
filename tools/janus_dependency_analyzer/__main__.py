"""
Entry point for running the Janus Dependency Analyzer as a module.

Allows the package to be invoked directly with:
    python -m janus_dependency_analyzer [command] [options]
"""

from .cli import main

if __name__ == "__main__":
    main()
