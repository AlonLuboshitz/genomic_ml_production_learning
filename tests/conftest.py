"""pytest configuration — sets non-interactive matplotlib backend for headless CI."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
