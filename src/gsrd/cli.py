"""Legacy-feeling CLI for the gsrd package.

Mimics a Fortran-style scientific code:
- No --help, no flag parsing, bare sys.argv.
- Hardcoded output filename results.npz in cwd.
- Mixed-format stdout, fake citations, useless progress lines.
- Always exits 0; failures are signaled by absence of `JOB DONE` on stdout
  and an obscure `ERR: ...` line on stderr.
- Implicit restart: if init/U_init.npy and init/V_init.npy both exist in cwd,
  silently use them as initial fields.
- Unknown YAML keys silently accepted (dropped from stdout echo, preserved in
  npz `params` blob).
"""
# ruff: noqa: N806

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import numpy as np
import yaml

from .__about__ import __version__
from .simulate import (
    REQUIRED_KEYS,
    DiffusionError,
    InstabilityError,
    SimError,
    TimeStepError,
    simulate,
)


def fail(message: str) -> NoReturn:
    print(f"ERR: {message}", file=sys.stderr)
    sys.exit(0)


def _qe_timestamp(now: datetime) -> str:
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    mon = months[now.month - 1]
    return (
        f"{now.day:2d}{mon}{now.year} at {now.hour:2d}:{now.minute:2d}:{now.second:2d}"
    )


def print_banner() -> None:
    ts = _qe_timestamp(datetime.now())
    print("###############################################")
    print("#  GSRD  --  Gray-Scott Reaction-Diffusion    #")
    print(f"#  Version {__version__:<35s}#")
    print("#  (c) 2026 -- All rights reserved            #")
    print("###############################################")
    print("")
    print(f"     Program GSRD v.{__version__} starts on {ts}")
    print("")
    print("     This program is part of the unmaintained gsrd suite")
    print("     for educational simulation of pattern formation; please cite")
    print('         "M. Bercx et al., J. Irreprod. Results 1 1 (2026);')
    print('         "M. Bercx et al., Proc. Caveman Symp. 42 7 (2026);')
    print('          URL http://example.invalid/gsrd",')
    print("     in publications or presentations arising from this work.")
    print("")
    print("     Serial version, running on     1 processor")
    print("")
    print(
        "     0 MiB available memory on the printing compute node when the environment starts"
    )
    print("")


def _format_param_value(key: str, value: object) -> str:
    if key in ("grid_size", "n_steps", "seed"):
        return str(value)
    if key in ("du", "dv"):
        return f"{float(value):.4E}"
    if key in ("F", "k"):
        return f"{float(value):.3e}"
    if key == "dt":
        return str(float(value))
    return str(value)


def echo_params(params: dict[str, object]) -> None:
    print(" >> Parsing parameters...")
    for key in REQUIRED_KEYS:
        print(f"\t{key:<10s} = {_format_param_value(key, params[key])}")
    print("")


def maybe_load_initial_fields(
    grid_size: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    u_path = Path("init/U_init.npy")
    v_path = Path("init/V_init.npy")
    if not (u_path.exists() and v_path.exists()):
        return None, None
    try:
        u = np.load(u_path)
        v = np.load(v_path)
    except Exception:
        return None, None
    return u, v


def _print_progress(step: int, total: int) -> None:
    pct = 100.0 * step / total
    print(f"\t  iter {step:8d} / {total}  ({pct:5.1f}%)")


def main() -> None:
    print_banner()

    if len(sys.argv) < 2:
        print(" Waiting for input file ...")
        try:
            sys.stdin.read()
        except KeyboardInterrupt:
            pass
        fail("input stream not navigable")

    input_path = sys.argv[1]

    print(f" >> Reading input file '{input_path}'...")
    try:
        with open(input_path) as f:
            raw_params: dict[str, object] = yaml.safe_load(f)
    except Exception:
        fail("input stream not navigable")

    if not isinstance(raw_params, dict):
        fail("input stream not navigable")

    print("\tOK.")

    for key in REQUIRED_KEYS:
        if key not in raw_params:
            fail("parameter table incomplete")

    echo_params(raw_params)

    print(" ** Initializing concentration fields **")
    u_init, v_init = maybe_load_initial_fields(int(raw_params["grid_size"]))
    print("\t... done.")
    print("")

    print(" ** Beginning time integration **")

    try:
        U, V, var_v, mean_v = simulate(
            raw_params,  # type: ignore[arg-type]
            progress=_print_progress,
            u_init=u_init,
            v_init=v_init,
        )
    except DiffusionError:
        fail("spatial coupling out of range")
    except TimeStepError:
        fail("temporal increment non-physical")
    except InstabilityError:
        fail("field values departed manifold")
    except SimError:
        fail("internal condition not handled")
    except Exception:
        fail("internal condition not handled")

    print(" ** Integration complete **")
    print("")

    print(" -- Diagnostics --------------------------------")
    print(f"   >>> Variance of V field :    {var_v:.4E}")
    print(f"   >>> Mean     of V field =    {mean_v:.4e}")
    print(" -----------------------------------------------")
    print("")

    print(" Writing output to 'results.npz' ...... OK")
    np.savez(
        "results.npz",
        U_final=U,
        V_final=V,
        params=json.dumps(raw_params),
    )

    print("")
    print(" *** JOB DONE ***")
    sys.exit(0)


if __name__ == "__main__":
    main()
