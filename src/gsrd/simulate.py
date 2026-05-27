"""Pure Gray-Scott reaction-diffusion logic. No I/O, no printing, no sys.exit."""
# ruff: noqa: N803, N806

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypedDict

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing import NotRequired


class SimParams(TypedDict):
    grid_size: int
    du: float
    dv: float
    F: float
    k: float
    dt: float
    n_steps: int
    seed: "NotRequired[int]"


REQUIRED_KEYS: list[str] = ["grid_size", "du", "dv", "F", "k", "dt", "n_steps"]


class SimError(Exception):
    """Base class for simulation errors."""


class DiffusionError(SimError):
    """Raised when du or dv is not positive."""


class TimeStepError(SimError):
    """Raised when dt is not positive."""


class InstabilityError(SimError):
    """Raised when NaN/Inf appears during integration, or initial-field shapes mismatch."""


def laplacian(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    return (
        -4 * Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )


def simulate(
    params: SimParams,
    progress: Callable[[int, int], None] | None = None,
    u_init: NDArray[np.floating] | None = None,
    v_init: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float, float]:
    """Integrate the Gray-Scott system and return the final state.

    :param params: Simulation parameters; see :class:`SimParams`.
    :param progress: Optional callback invoked as ``progress(step, total)``
        roughly 20 times over the run.
    :param u_init: Optional initial U field of shape ``(grid_size, grid_size)``.
        Ignored unless ``v_init`` is passed as well.
    :param v_init: Optional initial V field, as for ``u_init``.
    :return: ``(U, V, var_v, mean_v)``. A near-zero ``var_v`` means the V field
        decayed to a flat, patternless "trivial steady state", which is the
        legitimate outcome for Gray-Scott parameters outside the pattern-forming
        band rather than a failure. Callers that need to classify such runs
        should compare ``var_v`` against a threshold of their choosing; note it
        can underflow to exactly ``0.0``, so a log-scale consumer must floor it.
    :raises DiffusionError: If ``du`` or ``dv`` is not positive.
    :raises TimeStepError: If ``dt`` is not positive.
    :raises InstabilityError: If NaN/Inf appears during integration, or the
        initial fields do not match ``grid_size``.
    """
    n: int = params["grid_size"]
    du: float = params["du"]
    dv: float = params["dv"]
    F: float = params["F"]
    k: float = params["k"]
    dt: float = params["dt"]
    steps: int = params["n_steps"]
    seed: int | None = params.get("seed", None)

    if du <= 0 or dv <= 0:
        raise DiffusionError("Diffusion constants must be positive")

    if dt <= 0:
        raise TimeStepError("Time step must be positive")

    if seed is not None:
        np.random.seed(seed)

    if u_init is not None and v_init is not None:
        if u_init.shape != (n, n) or v_init.shape != (n, n):
            raise InstabilityError(f"Initial field shape mismatch for grid_size={n}")
        U = u_init.astype(float, copy=True)
        V = v_init.astype(float, copy=True)
    else:
        U = np.ones((n, n))
        V = np.zeros((n, n))
        r: int = n // 10
        c: int = n // 2
        U[c - r : c + r, c - r : c + r] = 0.50
        V[c - r : c + r, c - r : c + r] = 0.25

    report_every: int = max(1, steps // 20)
    for step in range(steps):
        Lu = laplacian(U)
        Lv = laplacian(V)
        uvv = U * V * V
        U += dt * (du * Lu - uvv + F * (1 - U))
        V += dt * (dv * Lv + uvv - (F + k) * V)

        if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
            raise InstabilityError(f"Numerical instability detected at step {step}")

        if progress is not None and (step + 1) % report_every == 0:
            progress(step + 1, steps)

    var_v: float = float(np.var(V))
    mean_v: float = float(np.mean(V))

    return U, V, var_v, mean_v
