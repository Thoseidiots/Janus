"""
Entry point for Universal Oxpecker when run as module or via PyInstaller exe.
"""
import sys
from pathlib import Path

# Add current directory and parent directories to path for bundled execution
for path_insert in [Path(__file__).parent, Path(__file__).parent.parent]:
    if str(path_insert) not in sys.path:
        sys.path.insert(0, str(path_insert))

# Now import and run the CLI
from cli import main

if __name__ == "__main__":
    main()
