"""gsrd — Gray-Scott reaction-diffusion legacy-style CLI."""

from .__about__ import __version__
from .simulate import (
    REQUIRED_KEYS,
    DiffusionError,
    InstabilityError,
    SimError,
    SimParams,
    TimeStepError,
    laplacian,
    simulate,
)

__all__ = [
    "REQUIRED_KEYS",
    "DiffusionError",
    "InstabilityError",
    "SimError",
    "SimParams",
    "TimeStepError",
    "__version__",
    "laplacian",
    "simulate",
]
