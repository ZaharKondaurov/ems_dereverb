#!/usr/bin/env python3
"""Generate bar plots with error bars for each evaluation metric."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "metrics_barplots"

# (value, error) — error is None when not available
MODELS = [
    {
        "name": "Noisy and Reverb",
        "short": "Noisy+Reverb",
        "group": "reference",
        "metrics": {
            "PESQ": (1.68, None),
            "NISQA-MOS": (2.73, None),
            "NISQA-NOISE": (2.40, None),
            "SRMR": (6.64, None),
            "STOI": (0.82, None),
            "RTF": (None, None),
        },
    },
    {
        "name": "Ground Truth",
        "short": "GT",
        "group": "reference",
        "metrics": {
            "PESQ": (None, None),
            "NISQA-MOS": (4.07, None),
            "NISQA-NOISE": (4.12, None),
            "SRMR": (8.94, None),
            "STOI": (1.0, None),
            "RTF": (None, None),
        },
    },
    {
        "name": "GTCRN",
        "short": "GTCRN",
        "group": "competitor",
        "metrics": {
            "PESQ": (2.10, None),
            "NISQA-MOS": (None, None),
            "NISQA-NOISE": (None, None),
            "SRMR": (8.22, None),
            "STOI": (0.86, None),
            "RTF": (0.03, None),
        },
    },
    {
        "name": "FSPEN",
        "short": "FSPEN",
        "group": "baseline",
        "metrics": {
            "PESQ": (2.22, 0.18),
            "NISQA-MOS": (3.55, 0.06),
            "NISQA-NOISE": (3.83, 0.08),
            "SRMR": (8.87, 0.60),
            "STOI": (0.77, 0.00),
            "RTF": (0.08, 0.01),
        },
    },
    {
        "name": "FSPEN + 48 kHz",
        "short": "+48 kHz",
        "group": "proposed",
        "metrics": {
            "PESQ": (2.39, 0.19),
            "NISQA-MOS": (3.74, 0.13),
            "NISQA-NOISE": (3.99, 0.12),
            "SRMR": (9.47, 0.65),
            "STOI": (0.88, 0.02),
            "RTF": (0.11, 0.01),
        },
    },
    {
        "name": "FSPEN + 48 kHz + SBLE",
        "short": "+SBLE",
        "group": "proposed",
        "metrics": {
            "PESQ": (2.32, 0.05),
            "NISQA-MOS": (3.67, 0.13),
            "NISQA-NOISE": (3.87, 0.03),
            "SRMR": (7.8, 0.09),
            "STOI": (0.80, 0.00),
            "RTF": (0.13, 0.03),
        },
    },
    {
        "name": "FSPEN + 48 kHz + SBDC + Overlap",
        "short": "+SBDC+Overlap",
        "group": "proposed",
        "metrics": {
            "PESQ": (2.33, 0.08),
            "NISQA-MOS": (3.64, 0.16),
            "NISQA-NOISE": (3.88, 0.08),
            "SRMR": (7.8, 0.21),
            "STOI": (0.80, 0.01),
            "RTF": (0.13, 0.01),
        },
    },
    {
        "name": "FSPEN + 48 kHz + Overlap",
        "short": "+Overlap",
        "group": "proposed",
        "metrics": {
            "PESQ": (2.35, 0.28),
            "NISQA-MOS": (3.73, 0.13),
            "NISQA-NOISE": (4.00, 0.12),
            "SRMR": (9.6, 0.45),
            "STOI": (0.88, 0.01),
            "RTF": (0.11, 0.01),
        },
    },
]

METRICS = ["PESQ", "NISQA-MOS", "NISQA-NOISE", "SRMR", "STOI", "RTF"]

COLORS = {
    "reference": "#B0B0B0",
    "baseline": "#4C72B0",
    "competitor": "#55A868",
    "proposed": "#DD8452",
}

GROUP_LABELS = {
    "reference": "Reference",
    "baseline": "Baseline (FSPEN)",
    "competitor": "Competitor (GTCRN)",
    "proposed": "Proposed (FSPEN+)",
}


def plot_metric(metric: str, output_dir: Path) -> Path | None:
    entries = []
    for model in MODELS:
        value, err = model["metrics"][metric]
        if value is not None:
            entries.append(
                {
                    "short": model["short"],
                    "value": value,
                    "err": err if err and err > 0 else 0.0,
                    "group": model["group"],
                }
            )

    if not entries:
        return None

    labels = [e["short"] for e in entries]
    values = [e["value"] for e in entries]
    errors = [e["err"] for e in entries]
    colors = [COLORS[e["group"]] for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.72)

    ax.errorbar(
        x,
        values,
        yerr=errors,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f"{metric} comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    y_min = min(v - e for v, e in zip(values, errors))
    y_max = max(v + e for v, e in zip(values, errors))
    margin = (y_max - y_min) * 0.12 if y_max > y_min else 0.1
    # ax.set_ylim(y_min - margin, y_max + margin)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[g], label=GROUP_LABELS[g])
        for g in ("reference", "baseline", "competitor", "proposed")
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    out_path = output_dir / f"{metric.lower().replace('-', '_')}_barplot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_metrics(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in METRICS:
        path = plot_metric(metric, output_dir)
        if path is not None:
            paths.append(path)
    return paths


if __name__ == "__main__":
    created = plot_all_metrics()
    print(f"Created {len(created)} plots in {OUTPUT_DIR}:")
    for p in created:
        print(f"  {p.name}")
