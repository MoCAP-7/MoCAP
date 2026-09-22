"""A minimal code-as-policy agent for the physical YOR robot.

The model writes short Python policies; those policies run on the Jetson against
a registry of named primitive callables that reach the robot through
``YorEnvironment`` -> ``NavigationController`` -> ``YorHardwareBridge``.
"""

from .agents.default import DefaultAgent
from .environment import YorEnvironment, summarize_observation
from .exceptions import Finished, FormatError, ModelError, PrimitiveFailed, YorAgentError
from .executor import PolicyExecutor
from .primitives.navigation import register_navigation_primitives
from .primitives.registry import PrimitiveRegistry
from .trace import Trace

__version__ = "0.1.0"

__all__ = [
    "DefaultAgent",
    "Finished",
    "FormatError",
    "ModelError",
    "PolicyExecutor",
    "PrimitiveFailed",
    "PrimitiveRegistry",
    "Trace",
    "YorAgentError",
    "YorEnvironment",
    "register_navigation_primitives",
    "summarize_observation",
]
