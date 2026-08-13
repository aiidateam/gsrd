"""Regenerate the gsrd gallery images used by the aiida-core tutorial.

Run from a checkout of this repository::

    uv run --extra gallery python gallery/generate.py

Writes ``showcase.png``, ``fields.png``, ``dissolved.png`` and ``heatmap.png``
into this directory. All single-field snapshots share one resolution so the
gallery looks uniform wherever the images are embedded.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.typing import NDArray

from gsrd import SimError, SimParams, simulate

OUT: Final = Path(__file__).parent
DPI: Final = 160

# Shared operating point; individual images override F/k.
BASE_PARAMS: Final[SimParams] = {
    "grid_size": 128,
    "du": 0.16,
    "dv": 0.08,
    "F": 0.040,
    "k": 0.060,
    "dt": 1.0,
    "n_steps": 10000,
    "seed": 42,
}

# 2D scan grid (matches the tutorial's module 3b sweep).
F_GRID: Final = (0.038, 0.044, 0.050, 0.056, 0.062)
K_GRID: Final = (0.059, 0.061, 0.063, 0.065, 0.067)

# Runs that decay to a flat field have a variance LogNorm cannot place (it can
# underflow to exactly zero), so clamp them onto the bottom of the colour scale.
VARIANCE_FLOOR: Final = 1e-6


def run(
    *, F: float, k: float
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Run one simulation, returning ``(U, V, variance_of_V)``.

    Every image shares one operating point; only the feed rate ``F`` and kill
    rate ``k`` vary, so the gallery reads as a single parameter scan.

    :param F: Feed rate.
    :param k: Kill rate.
    :return: Final ``U`` and ``V`` fields and the variance of ``V``.
    """
    u, v, var_v, _ = simulate({**BASE_PARAMS, "F": F, "k": k})
    return u, v, var_v


def scan_variance() -> NDArray[np.floating]:
    """Sweep the F x k grid, returning the variance of V per point, floored."""
    variance = np.full((len(K_GRID), len(F_GRID)), VARIANCE_FLOOR)
    for i, k in enumerate(K_GRID):
        for j, f in enumerate(F_GRID):
            try:
                _, _, var_v = run(F=f, k=k)
            except SimError:
                var_v = VARIANCE_FLOOR
            variance[i, j] = max(var_v, VARIANCE_FLOOR)
    return variance


def save_field(field: NDArray[np.floating], *, cmap: str, name: str) -> None:
    """Save one field as a borderless square PNG (uniform resolution)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(field, cmap=cmap, origin="lower")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {name}")


def save_field_pair(
    u: NDArray[np.floating], v: NDArray[np.floating], *, name: str
) -> None:
    """Save the U and V fields of one run side by side."""
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(8, 4))
    ax_u.imshow(u, cmap="viridis", origin="lower")
    ax_u.set_title("U field")
    ax_u.set_axis_off()
    ax_v.imshow(v, cmap="inferno", origin="lower")
    ax_v.set_title("V field")
    ax_v.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {name}")


def save_heatmap(variance: NDArray[np.floating], *, name: str) -> None:
    """Save the F x k variance scan as a log-scale heatmap with a colourbar."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(
        variance,
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=VARIANCE_FLOOR, vmax=float(variance.max())),
        aspect="auto",
    )
    ax.set_xticks(range(len(F_GRID)), [f"{f:.3f}" for f in F_GRID])
    ax.set_yticks(range(len(K_GRID)), [f"{k:.3f}" for k in K_GRID])
    ax.set_xlabel("feed rate F")
    ax.set_ylabel("kill rate k")
    fig.colorbar(im, ax=ax, label="variance(V)")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {name}")


def main() -> None:
    # Showcase: a fully developed spot pattern (module 3a 'spots').
    _, v, _ = run(F=0.030, k=0.062)
    save_field(v, cmap="magma", name="showcase.png")

    # Dissolved: outside the pattern-forming band, where V decays to a flat field.
    _, v, _ = run(F=0.050, k=0.060)
    save_field(v, cmap="magma", name="dissolved.png")

    # Fields: U and V side by side, inside the band so both show structure.
    u, v, _ = run(F=0.040, k=0.060)
    save_field_pair(u, v, name="fields.png")

    save_heatmap(scan_variance(), name="heatmap.png")


if __name__ == "__main__":
    main()
