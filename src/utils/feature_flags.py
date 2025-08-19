"""
Feature flag management for enabling/disabling experimental components
like LangChain and LangGraph without impacting legacy features.
"""

import os
from typing import Dict, Any


class FeatureFlags:
    """Feature flag management system.

    Flags are controlled via environment variables and default to False
    to avoid impacting existing behavior.
    """

    def __init__(self) -> None:
        self.flags = {
            "use_structured_output": os.getenv("USE_STRUCTURED_OUTPUT", "true").lower() == "true",
            "enable_memory": os.getenv("ENABLE_MEMORY", "true").lower() == "true",
        }

    def is_enabled(self, flag_name: str) -> bool:
        return bool(self.flags.get(flag_name, False))

    def get_all(self) -> Dict[str, Any]:
        return dict(self.flags)


# Global instance
feature_flags = FeatureFlags()


