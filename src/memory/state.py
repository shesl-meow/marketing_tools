"""Agent state helpers (placeholder)."""

from typing import Any, Dict, Optional


def init_state(initial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new mutable state dict for the agent run."""
    return dict(initial or {})
