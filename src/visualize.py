"""
src/visualize.py
================
Static 300dpi figures for the report.

* Temporal heatmap (hour x day-of-week)        -> temporal_heatmap.png
* Gi* hotspot significance scatter              -> hotspot_significance.png
* Fairness gap chart (clearance & error rates)  -> fairness_gaps.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, config: dict):
        self.cfg = config
        self.fig_dir = Path(config["paths"]["figures"])
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------- temporal heatmap
    def temporal_heatmap(self, heatmap: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(heatmap, cmap="rocket_r", linewidths=0.3,
                    cbar_kws={"label": "Incident count"}, ax=ax)
        ax.set_title("LAPD Incidents — Hour of Day × Day of Week (2020–2024)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Hour of day"); ax.set_ylabel("")
        fig.tight_layout()
        out = self.fig_dir / "temporal_heatmap.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure -> %s", out)
        return out

    # ----------------------------------------------- Gi* significance
    def hotspot_significance(self, grid: pd.DataFrame) -> Path:
        palette = {"Hot Spot": "#d7191c", "Cold Spot": "#2c7bb6",
                   "Not Significant": "#cccccc"}
        fig, ax = plt.subplots(figsize=(11, 10))
        for cls, color in palette.items():
            sub = grid[grid["hotspot_class"] == cls]
            ax.scatter(sub["cell_lon"], sub["cell_lat"], s=14,
                       c=color, label=f"{cls} ({len(sub)})",
                       alpha=0.8, edgecolor="none")
        ax.set_title("Getis-Ord Gi* Hotspot Significance\n"
                     "(grid cells, p<0.05; red=hot, blue=cold)",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = self.fig_dir / "hotspot_significance.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure -> %s", out)
        return out

    # --------------------------------------------------- fairness gaps
    def fairness_gaps(self, audit_table: pd.DataFrame,
                      attribute: str = "Victim_descent") -> Path:
        sub = (audit_table[audit_table["audited_attribute"] == attribute]
               .dropna(subset=["model_fnr"])
               .sort_values("data_clearance_rate", ascending=False))
        if sub.empty:
            sub = audit_table[audit_table["audited_attribute"] == attribute]

        x = np.arange(len(sub)); w = 0.38
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x - w/2, sub["data_clearance_rate"], w,
               label="Data clearance rate", color="#2c7bb6")
        ax.bar(x + w/2, sub["model_fnr"], w,
               label="Model false-negative rate", color="#d7191c")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"].astype(str), rotation=45, ha="right")
        ax.set_ylabel("Rate")
        ax.set_title(f"Fairness Gaps by {attribute}\n"
                     "Data disparity (clearance rate) vs Model error (FNR)",
                     fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        out = self.fig_dir / "fairness_gaps.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure -> %s", out)
        return out

    def run(self, hotspot_results: dict, audit_table: pd.DataFrame) -> dict:
        return {
            "temporal": self.temporal_heatmap(hotspot_results["temporal"]["heatmap"]),
            "hotspot": self.hotspot_significance(hotspot_results["grid"]),
            "fairness": self.fairness_gaps(audit_table),
        }
