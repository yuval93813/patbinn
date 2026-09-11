#!/usr/bin/env python3
"""
Generate the paper figures from the results CSVs produced by aggregate_cv_results.py,
sweep_architecture.py, and sweep_read_counts.py. Each plot is skipped (with a
message) if its input CSV doesn't exist yet, so this can be run incrementally as
results come in.

Colors use the Okabe-Ito colorblind-safe qualitative palette, one fixed hue per
series (never cycled/re-assigned): router=blue, expert=vermillion, flat=green.
Precision is distinguished by linestyle (solid=binary, dashed=float32), not color,
so the two encodings compose independently.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_ROUTER = "#0072B2"
COLOR_EXPERT = "#D55E00"
COLOR_FLAT = "#009E73"
COLOR_MUTED = "#666666"


def read_csv(path):
    if not os.path.exists(path):
        print(f"  (skip) {path} not found")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_size_sweep(router_path, expert_path, out_path):
    """Accuracy against parameter count, for the router and expert sweeps.

    The two sweeps are separate runs and therefore separate files: the router
    sweep holds the experts fixed and the expert sweep holds the router fixed,
    so only one component varies in each.
    """
    router_rows = read_csv(router_path) or []
    expert_rows = read_csv(expert_path) or []
    if not router_rows and not expert_rows:
        return

    # One line per (component, depth). Shapes of different depth are different
    # architecture families, not points on one curve: connecting them sorts a
    # 1-layer and a 2-layer shape into the same zigzag and hides the actual
    # finding, which is that the 2-layer curve sits above the 1-layer one
    # everywhere. Depth is carried by linestyle, component by colour.
    series = [
        ("Router (family)", router_rows, "router_params", "per_read_router_acc",
         "router_hidden_sizes", COLOR_ROUTER, 8),
        ("Expert (variant, oracle-routed)", expert_rows, "expert_params_total",
         "per_read_expert_acc_oracle", "expert_hidden_sizes", COLOR_EXPERT, -14),
    ]
    depth_style = {1: ":", 2: "-", 3: "--"}

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, rows, xcol, ycol, namecol, color, dy in series:
        if not rows:
            continue
        by_depth = {}
        for r in rows:
            by_depth.setdefault(int(r["n_hidden_layers"]), []).append(r)
        for depth in sorted(by_depth):
            pts = sorted(by_depth[depth], key=lambda r: int(r[xcol]))
            x = [int(r[xcol]) for r in pts]
            y = [float(r[ycol]) for r in pts]
            ax.plot(x, y, marker="o", color=color, linewidth=2, markersize=6,
                    linestyle=depth_style.get(depth, "-"),
                    label=f"{label}, {depth} hidden layer{'s' if depth > 1 else ''}")
            for xi, yi, r in zip(x, y, pts):
                ax.annotate(r[namecol], (xi, yi), textcoords="offset points",
                            xytext=(0, dy), fontsize=7.5, color=color, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Per-read accuracy (%)")
    ax.set_title("Accuracy vs. model size\n(labels: hidden-layer widths; router and expert swept separately)",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.99))
    ax.margins(y=0.16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_cv_fold_distribution(hier_path, flat_path, out_path):
    hier_rows = read_csv(hier_path)
    flat_rows = read_csv(flat_path)
    if not hier_rows and not flat_rows:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    positions, labels, data, colors = [], [], [], []

    if hier_rows:
        data.append([float(r["sample_family_acc"]) for r in hier_rows])
        labels.append("MoE\nfamily")
        colors.append(COLOR_ROUTER)
        data.append([float(r["sample_leaf_acc_pipeline"]) for r in hier_rows])
        labels.append("MoE\nleaf (pipeline)")
        colors.append(COLOR_EXPERT)
    if flat_rows:
        data.append([float(r["sample_leaf_acc"]) for r in flat_rows])
        labels.append("Flat\nleaf")
        colors.append(COLOR_FLAT)

    positions = list(range(1, len(data) + 1))
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for i, d in enumerate(data):
        jitter_x = [positions[i] + (j % 2 * 2 - 1) * 0.06 * (j // 2 + 1) for j in range(len(d))]
        ax.scatter(jitter_x, d, color=colors[i], s=22, zorder=3, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-sample majority-vote accuracy (%)")
    ax.set_title("CV fold-accuracy distribution (genome-level held-out)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_read_count_curve(path, out_path):
    rows = read_csv(path)
    if not rows:
        return

    # Tolerate rows with a blank metric: a sweep that was interrupted, or a row
    # appended by hand, can leave a column empty. Skip those values rather than
    # failing the whole figure, and say how many were skipped.
    by_n = {}
    skipped = 0
    for r in rows:
        n = int(r["max_votes_per_sample"])
        by_n.setdefault(n, {"family": [], "leaf": []})
        for key, col in (("family", "sample_family_acc"), ("leaf", "sample_leaf_acc_pipeline")):
            raw = (r.get(col) or "").strip()
            if not raw:
                skipped += 1
                continue
            by_n[n][key].append(float(raw))
    if skipped:
        print(f"  (note) skipped {skipped} blank metric value(s) in {path}")

    # Drop vote counts left with no usable data at all.
    by_n = {n: v for n, v in by_n.items() if v["family"] and v["leaf"]}
    if not by_n:
        print(f"  (skip) no usable rows in {path}")
        return

    ns = sorted(by_n)

    def mean_std(key):
        means = [sum(by_n[n][key]) / len(by_n[n][key]) for n in ns]
        stds = [
            (sum((v - m) ** 2 for v in by_n[n][key]) / len(by_n[n][key])) ** 0.5
            for n, m in zip(ns, means)
        ]
        return means, stds

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, color, label in [("family", COLOR_ROUTER, "Family accuracy"), ("leaf", COLOR_EXPERT, "Leaf accuracy (pipeline)")]:
        means, stds = mean_std(key)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.plot(ns, means, marker="o", color=color, linewidth=2, markersize=6, label=label)
        ax.fill_between(ns, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Reads aggregated per sample (majority vote)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Read-count-to-confidence (mean ± std across CV folds)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from results CSVs")
    parser.add_argument("--router_sweep_csv", default="results/router_sweep.csv")
    parser.add_argument("--expert_sweep_csv", default="results/expert_sweep.csv")
    parser.add_argument("--cv_results_csv", default="results/cv_results.csv")
    parser.add_argument("--cv_results_flat_csv", default="results/cv_results_flat.csv")
    parser.add_argument("--read_count_sweep_csv", default="results/read_count_sweep.csv")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Size sweep:")
    plot_size_sweep(args.router_sweep_csv, args.expert_sweep_csv,
                    os.path.join(args.out_dir, "size_sweep.png"))

    print("CV fold-accuracy distribution:")
    plot_cv_fold_distribution(args.cv_results_csv, args.cv_results_flat_csv,
                               os.path.join(args.out_dir, "cv_fold_distribution.png"))

    print("Read-count-to-confidence:")
    plot_read_count_curve(args.read_count_sweep_csv, os.path.join(args.out_dir, "read_count_curve.png"))


if __name__ == "__main__":
    main()
