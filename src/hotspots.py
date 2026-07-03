"""
src/hotspots.py
===============
Spatiotemporal hotspot analysis — the geospatial core.

Provides
--------
* Temporal aggregations: counts by hour-of-day, day-of-week, month, and an
  hour x day-of-week matrix.
* Spatial density: a grid aggregation of incidents (used for both KDE-style
  visualisation downstream and the Gi* statistic here).
* A FORMAL hotspot statistic — Getis-Ord Gi* (via ``esda`` + ``libpysal``) — on a
  regular lat/lon grid, classifying each cell as a statistically significant hot
  spot, cold spot, or not significant.
* Crime-type-by-area breakdown.

All spatial work uses ONLY geo-valid rows (the (0,0) / out-of-region exclusion
computed in data_processor).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HotspotAnalyzer:
    def __init__(self, config: dict):
        self.cfg = config
        self.hcfg = config["hotspots"]
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- temporal
    def temporal_patterns(self, df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
        by_hour = df.groupby("hour").size().rename("incidents")
        # Order day names Mon..Sun.
        order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
        by_dow = (
            df.groupby("dayname").size().reindex(order).rename("incidents")
        )
        by_month = df.groupby("month").size().rename("incidents")
        heatmap = (
            df.pivot_table(index="dayname", columns="hour",
                           values="DR_NO", aggfunc="count")
            .reindex(order)
            .fillna(0)
            .astype(int)
        )
        logger.info(
            "Temporal: peak hour=%s, peak day=%s",
            int(by_hour.idxmax()), by_dow.idxmax(),
        )
        return {"by_hour": by_hour, "by_dow": by_dow,
                "by_month": by_month, "heatmap": heatmap}

    # ------------------------------------------------------------- spatial
    def _build_grid(self, geo: pd.DataFrame) -> pd.DataFrame:
        d = self.hcfg["grid_decimals"]
        g = geo.copy()
        g["cell_lat"] = g["LAT"].round(d)
        g["cell_lon"] = g["LON"].round(d)
        grid = (
            g.groupby(["cell_lat", "cell_lon"])
            .size()
            .reset_index(name="n_incidents")
        )
        return grid

    def getis_ord(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Getis-Ord Gi* on a regular grid of incident counts."""
        from libpysal.weights import DistanceBand
        from esda.getisord import G_Local

        geo = df[df["geo_valid"]].copy()
        grid = self._build_grid(geo)
        coords = grid[["cell_lon", "cell_lat"]].values

        w = DistanceBand(
            coords,
            threshold=self.hcfg["gi_threshold"],
            binary=True,
            silence_warnings=True,
        )
        counts = grid["n_incidents"].to_numpy(dtype=float)
        gi = G_Local(
            counts, w, star=True,
            permutations=self.hcfg["gi_permutations"],
        )

        alpha = self.hcfg["significance"]
        grid["gi_z"] = gi.Zs
        grid["gi_p"] = gi.p_sim
        sig = grid["gi_p"] < alpha
        grid["hotspot_class"] = np.select(
            [sig & (grid["gi_z"] > 0), sig & (grid["gi_z"] < 0)],
            ["Hot Spot", "Cold Spot"],
            default="Not Significant",
        )

        n_hot = int((grid["hotspot_class"] == "Hot Spot").sum())
        n_cold = int((grid["hotspot_class"] == "Cold Spot").sum())
        logger.info(
            "Gi*: %s grid cells | %s significant hot spots, %s cold spots (p<%.2f)",
            f"{len(grid):,}", n_hot, n_cold, alpha,
        )

        out = self.metrics_dir / "hotspots.csv"
        grid.sort_values("gi_z", ascending=False).to_csv(out, index=False)
        logger.info("Persisted hotspot metrics -> %s", out)
        self.n_hot, self.n_cold = n_hot, n_cold
        return grid

    # -------------------------------------------------- crime-type by area
    def crime_by_area(self, df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        ct = (
            df.groupby(["Area", "Crime_Code"]).size()
            .reset_index(name="n")
            .sort_values(["Area", "n"], ascending=[True, False])
        )
        top = ct.groupby("Area").head(top_n).reset_index(drop=True)
        out = self.metrics_dir / "crime_by_area.csv"
        top.to_csv(out, index=False)
        logger.info("Top %s crime types per area -> %s", top_n, out)
        return top

    # ----------------------------------------------------------- driver
    def run(self, df: pd.DataFrame) -> dict:
        temporal = self.temporal_patterns(df)
        grid = self.getis_ord(df)
        crime_area = self.crime_by_area(df)
        return {"temporal": temporal, "grid": grid, "crime_area": crime_area}
