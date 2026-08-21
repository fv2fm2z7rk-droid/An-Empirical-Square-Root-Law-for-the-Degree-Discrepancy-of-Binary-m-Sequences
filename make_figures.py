#!/usr/bin/env python3
"""Generate the two manuscript figures from full_rho_profiles.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "full_rho_profiles.json").exists() else HERE.parent
sys.path.insert(0, str(ROOT))

from plot_argmax_horizons import extract_table, plot_figure  # noqa: E402

FIG = HERE / "figures"
PROFILES = ROOT / "full_rho_profiles.json"


def plot_normalized_peaks(profiles: dict, out_pdf: Path, out_png: Path) -> None:
    ks, dks = [], []
    for k in range(2, 29):
        rec = profiles[str(k)]
        ks.append(k)
        dks.append(rec["max_Delta"] / math.sqrt(rec["N"]))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0.5, color="0.35", ls="--", lw=0.9, label=r"$1/2$")
    ax.axhline(1.0 / np.sqrt(2), color="0.45", ls=":", lw=0.9, label=r"$1/\sqrt{2}$")
    ax.axhline(1.0, color="0.15", ls="-", lw=0.9, label=r"$1$")
    ax.plot(ks, dks, color="0.75", lw=0.8, zorder=1)
    ax.scatter(ks, dks, s=28, facecolors="0.15", edgecolors="0.15", zorder=2)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$d_k=\max_{r\in R_k}\Delta(r)/2^{k/2}$")
    ax.set_xlim(1.5, 28.5)
    ax.set_ylim(0.0, 1.25)
    ax.set_xticks(range(2, 29, 2))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with open(PROFILES) as f:
        profiles = json.load(f)

    plot_normalized_peaks(
        profiles,
        FIG / "fig_normalized_peaks.pdf",
        FIG / "fig_normalized_peaks.png",
    )

    table = extract_table(profiles, n_top=3)
    plot_figure(
        table,
        FIG / "fig_argmax_horizons.pdf",
        FIG / "fig_argmax_horizons.png",
    )
    print(f"Wrote figures in {FIG}")


if __name__ == "__main__":
    main()
