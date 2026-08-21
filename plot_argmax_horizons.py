#!/usr/bin/env python3
"""
plot_argmax_horizons.py

From the existing k=2..28 profiles, record where max_r Δ(r) sits on the
28-point grid R_k.  A Brownian-bridge heuristic for the *scale* of Δ(r)
has variance ρ(1-ρ), hence a typical peak near ρ = r/M = 1/2.  Complement
symmetry Δ(M-r) = Δ(r) means a recorded maximizer at ρ is interchangeable
with one at 1-ρ, so the folded location min(ρ, 1-ρ) is the quantity that
tests “near mid-path”.

Writes:
  argmax_horizons.json   table of first and top-3 argmaxes
  fig_argmax_horizons.pdf / .png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def unique_profile(rec: dict) -> list[tuple[int, float, float]]:
    """Deduplicate the grid by r (small k collapses several relative lengths)."""
    seen: dict[int, tuple[float, float]] = {}
    for r, rho, dlt in zip(rec["r"], rec["rho"], rec["Delta"]):
        if r not in seen:
            seen[r] = (float(rho), float(dlt))
    # Drop the full-period endpoint (Δ ≡ 0).
    items = [(r, rho, dlt) for r, (rho, dlt) in seen.items() if r < rec["M"]]
    items.sort(key=lambda row: row[0])
    return items


def top_argmaxes(items: list[tuple[int, float, float]], n: int = 3):
    ranked = sorted(items, key=lambda row: row[2], reverse=True)
    return ranked[:n]


def extract_table(profiles: dict, n_top: int = 3) -> dict:
    rows = []
    for k in sorted(int(key) for key in profiles):
        rec = profiles[str(k)]
        items = unique_profile(rec)
        ranked = top_argmaxes(items, n=n_top)
        max_delta = ranked[0][2]
        mid = min(items, key=lambda row: abs(row[1] - 0.5))
        first_r, first_rho, first_delta = ranked[0]
        rows.append(
            {
                "k": k,
                "M": rec["M"],
                "method": rec.get("method", "unknown"),
                "n_grid": len(items),
                "max_Delta": max_delta,
                "r_star": first_r,
                "rho_star": first_rho,
                "rho_star_folded": min(first_rho, 1.0 - first_rho),
                "mid_rho": mid[1],
                "mid_Delta": mid[2],
                "mid_over_max": mid[2] / max_delta if max_delta else 0.0,
                "top": [
                    {
                        "rank": i + 1,
                        "r": r,
                        "rho": rho,
                        "rho_folded": min(rho, 1.0 - rho),
                        "Delta": dlt,
                        "Delta_over_max": dlt / max_delta if max_delta else 0.0,
                    }
                    for i, (r, rho, dlt) in enumerate(ranked)
                ],
            }
        )
    return {"n_top": n_top, "rows": rows}


def plot_figure(table: dict, out_pdf: Path, out_png: Path) -> None:
    rows = table["rows"]
    ks = np.array([row["k"] for row in rows])

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

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)

    rank_style = {
        1: dict(s=42, marker="o", zorder=4, linewidths=0.6),
        2: dict(s=28, marker="s", zorder=3, linewidths=0.6),
        3: dict(s=22, marker="D", zorder=2, linewidths=0.6),
    }
    rank_face = {1: "0.15", 2: "white", 3: "white"}
    rank_edge = {1: "0.15", 2: "C0", 3: "C1"}
    rank_label = {1: "1st", 2: "2nd", 3: "3rd"}

    # Thin polyline through the first argmax so the wander is visible.
    rho1 = [row["rho_star"] for row in rows]
    fold1 = [row["rho_star_folded"] for row in rows]
    axes[0].plot(ks, rho1, color="0.75", lw=0.8, zorder=1)
    axes[1].plot(ks, fold1, color="0.75", lw=0.8, zorder=1)

    for row in rows:
        k = row["k"]
        for entry in row["top"]:
            rank = entry["rank"]
            style = rank_style[rank]
            axes[0].scatter(
                k,
                entry["rho"],
                facecolors=rank_face[rank],
                edgecolors=rank_edge[rank],
                **style,
            )
            axes[1].scatter(
                k,
                entry["rho_folded"],
                facecolors=rank_face[rank],
                edgecolors=rank_edge[rank],
                **style,
            )

    # Distinguish sampled k with a light tick on the x-axis of the top panel.
    sampled = [row["k"] for row in rows if row["method"] == "sampled"]
    axes[0].scatter(
        sampled,
        np.full(len(sampled), -0.03),
        marker="|",
        s=40,
        c="0.4",
        linewidths=1.0,
        clip_on=False,
    )

    ax0, ax1 = axes
    ax0.axhline(0.5, color="0.35", ls="--", lw=0.9)
    ax0.axhspan(0.25, 0.75, color="0.92", zorder=0)
    ax0.set_ylabel(r"recorded maximizer $\rho^\ast=r^\ast/M$")
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_yticks(np.linspace(0.0, 1.0, 5))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=rank_face[1],
            markeredgecolor=rank_edge[1],
            markersize=6.5,
            linestyle="None",
            label=rank_label[1],
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=rank_face[2],
            markeredgecolor=rank_edge[2],
            markersize=6.5,
            linestyle="None",
            label=rank_label[2],
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=rank_face[3],
            markeredgecolor=rank_edge[3],
            markersize=6.5,
            linestyle="None",
            label=rank_label[3],
        ),
    ]
    ax0.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        framealpha=0.95,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.5,
        fontsize=8,
    )

    ax1.axhline(0.5, color="0.35", ls="--", lw=0.9)
    ax1.axhline(0.25, color="0.55", ls=":", lw=0.8)
    ax1.axhspan(0.25, 0.5, color="0.92", zorder=0)
    ax1.set_ylabel(r"folded $\min(\rho^\ast,1-\rho^\ast)$")
    ax1.set_xlabel(r"$k$")
    ax1.set_ylim(-0.02, 0.55)
    ax1.set_yticks([0.0, 0.1, 0.25, 0.4, 0.5])
    ax1.set_xticks(range(2, 29, 2))

    for ax in axes:
        ax.set_xlim(1.5, 28.5)
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def print_summary(table: dict) -> None:
    print(
        f"{'k':>3} {'method':>8} {'ρ*':>7} {'fold':>7} "
        f"{'Δmid/max':>9}  top-3 ρ (Δ/max)"
    )
    print("-" * 78)
    n_short = n_bulk = 0
    for row in table["rows"]:
        fold = row["rho_star_folded"]
        if fold < 0.25:
            n_short += 1
        if fold >= 0.40:
            n_bulk += 1
        tops = ", ".join(
            f"{e['rho']:.3f}({e['Delta_over_max']:.2f})" for e in row["top"]
        )
        print(
            f"{row['k']:3d} {row['method']:>8} {row['rho_star']:7.3f} "
            f"{fold:7.3f} {row['mid_over_max']:9.3f}  {tops}"
        )
    n = len(table["rows"])
    folds = [row["rho_star_folded"] for row in table["rows"]]
    mids = [row["mid_over_max"] for row in table["rows"]]
    print()
    print(
        f"folded ρ* < 0.25 (short/long wing): {n_short}/{n};  "
        f"folded ρ* ≥ 0.40 (near mid-path): {n_bulk}/{n}"
    )
    print(
        f"folded ρ* median = {np.median(folds):.3f};  "
        f"Δ(ρ≈1/2)/max Δ median = {np.median(mids):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot r*(k)/M from existing degree-discrepancy profiles"
    )
    parser.add_argument(
        "--json", type=Path, default=Path("full_rho_profiles.json")
    )
    parser.add_argument(
        "--table", type=Path, default=Path("argmax_horizons.json")
    )
    parser.add_argument(
        "--pdf", type=Path, default=Path("fig_argmax_horizons.pdf")
    )
    parser.add_argument(
        "--png", type=Path, default=Path("fig_argmax_horizons.png")
    )
    args = parser.parse_args()

    with open(args.json) as f:
        profiles = json.load(f)

    table = extract_table(profiles, n_top=3)

    # Consistency with the stored profile maxima.
    for row in table["rows"]:
        stored = profiles[str(row["k"])]["max_Delta"]
        if abs(row["max_Delta"] - stored) > 1e-9:
            raise ValueError(
                f"k={row['k']}: extracted max {row['max_Delta']} "
                f"!= stored max_Delta {stored}"
            )

    with open(args.table, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")

    plot_figure(table, args.pdf, args.png)
    print_summary(table)
    print(f"\nWrote {args.table}, {args.pdf}, {args.png}")


if __name__ == "__main__":
    main()
