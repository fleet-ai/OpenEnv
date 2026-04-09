"""Gym-Anything Environment adapter for OpenEnv.

Wraps gym-anything's 250+ desktop software environments (CUA-World) as
an OpenEnv-compatible environment for computer-use agent evaluation.
"""

from .client import GymAnythingEnvClient

__all__ = ["GymAnythingEnvClient"]
