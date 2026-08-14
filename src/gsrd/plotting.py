"""Matplotlib helpers for visualising ``gsrd`` output.

These operate on plain NumPy arrays and floats (``gsrd`` has no AiiDA
dependency), so they are shared by this package's gallery generator and by the
aiida-core tutorial, which unwraps its AiiDA nodes before calling in.

``matplotlib`` is an optional dependency (``gsrd[plot]``); it is imported lazily
inside each function so ``import gsrd.plotting`` works without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_field_gallery(fields: Mapping[str, NDArray], *, cmap: str = "magma") -> Figure:
    """Plot several final concentration fields side by side, each labelled.

    :param fields: mapping from a label (e.g. ``'labyrinth'``) to a 2D field
        (typically the ``V_final`` array of a ``gsrd`` run).
    :param cmap: matplotlib colormap name.
    :return: the assembled figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax_array = plt.subplots(
        nrows=1, ncols=len(fields), figsize=(4 * len(fields), 4)
    )
    for ax, (label, field) in zip(np.atleast_1d(ax_array), fields.items()):
        ax.imshow(field, cmap=cmap, origin="lower")
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_variance_heatmap(
    grid: NDArray,
    f_grid: Sequence[float],
    k_grid: Sequence[float],
    *,
    dead_threshold: float = 1e-6,
) -> Figure:
    """Render a 2D ``variance(V)`` grid as a log-scale heatmap over ``(F, k)``.

    :param grid: array of shape ``(len(f_grid), len(k_grid))`` holding
        ``variance(V)`` per point; missing entries may be ``nan``.
    :param f_grid: feed-rate axis values (y-axis), in display order.
    :param k_grid: kill-rate axis values (x-axis), in display order.
    :param dead_threshold: values below this floor are clamped, so the log
        colour scale focuses on the physical range rather than numerical
        underflow.
    :return: the assembled figure.
    :raises ValueError: if no positive variance values are present to plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    grid = np.asarray(grid, dtype=float)
    if grid[grid > 0].size == 0:
        msg = "No positive variance values to plot."
        raise ValueError(msg)

    vmin = dead_threshold
    vmax = float(np.nanmax(grid))
    grid_for_plot = np.where(grid >= vmin, grid, vmin)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(
        grid_for_plot,
        origin="lower",
        aspect="auto",
        extent=(min(k_grid), max(k_grid), min(f_grid), max(f_grid)),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )
    ax.set_xlabel("Kill rate k")
    ax.set_ylabel("Feed rate F")
    ax.set_title(
        f"Gray-Scott pattern strength: variance(V) on a {len(f_grid)}x{len(k_grid)} F-by-k grid"
    )
    ax.set_xticks(list(k_grid))
    ax.set_yticks(list(f_grid))
    fig.colorbar(im, ax=ax, label="variance(V)")
    fig.tight_layout()
    return fig
