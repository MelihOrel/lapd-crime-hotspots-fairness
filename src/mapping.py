"""
src/mapping.py
==============
Interactive (Folium) and static (matplotlib) maps.

* Interactive HTML map: a HeatMap layer of incident density plus a per-area
  choropleth of incident counts, with a layer control. Saved to
  ``reports/maps/crime_heatmap.html``.
* Static 300dpi choropleth of incidents by patrol area. Saved to
  ``reports/figures/area_choropleth.png``.

All maps use ONLY geo-valid rows (the (0,0) / out-of-region exclusion).
The per-area choropleth is built from incident counts aggregated to the
centroid of each patrol Area (the dataset has no polygon geometries, so we
render area centroids sized/coloured by volume — an honest representation that
does not fabricate boundaries we do not have).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MapBuilder:
    def __init__(self, config: dict):
        self.cfg = config
        self.maps_dir = Path(config["paths"]["maps"])
        self.fig_dir = Path(config["paths"]["figures"])
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def _geo(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["geo_valid"]].copy()

    # ---------------------------------------------------- interactive map
    def interactive_map(self, df: pd.DataFrame, sample: int = 60000) -> Path:
        import folium
        from folium.plugins import HeatMap

        geo = self._geo(df)
        center = [geo["LAT"].median(), geo["LON"].median()]
        fmap = folium.Map(location=center, zoom_start=11, tiles="cartodbpositron")

        # --- HeatMap layer (sampled for browser performance) ---
        heat_src = geo.sample(min(sample, len(geo)), random_state=42)
        HeatMap(
            heat_src[["LAT", "LON"]].values.tolist(),
            radius=8, blur=12, min_opacity=0.3,
            name="Incident heatmap",
        ).add_to(fmap)

        # --- Per-area choropleth-style circles (centroid + count) ---
        area_stats = (
            geo.groupby("Area")
            .agg(lat=("LAT", "median"), lon=("LON", "median"),
                 n=("DR_NO", "count"))
            .reset_index()
        )
        vmax = area_stats["n"].max()
        layer = folium.FeatureGroup(name="Incidents by patrol area")
        for _, r in area_stats.iterrows():
            frac = r["n"] / vmax
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=8 + 22 * frac,
                color="#8B0000",
                fill=True,
                fill_color="#FF4500",
                fill_opacity=0.35 + 0.45 * frac,
                weight=1,
                popup=folium.Popup(
                    f"<b>{r['Area']}</b><br>{int(r['n']):,} incidents",
                    max_width=200),
            ).add_to(layer)
        layer.add_to(fmap)
        folium.LayerControl().add_to(fmap)

        out = self.maps_dir / "crime_heatmap.html"
        fmap.save(str(out))
        logger.info("Interactive map -> %s (%s heat points)", out, f"{len(heat_src):,}")
        return out

    # ------------------------------------------------------- static map
    def static_choropleth(self, df: pd.DataFrame) -> Path:
        geo = self._geo(df)
        area_stats = (
            geo.groupby("Area")
            .agg(lat=("LAT", "median"), lon=("LON", "median"),
                 n=("DR_NO", "count"))
            .reset_index()
            .sort_values("n", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(11, 10))
        sc = ax.scatter(
            area_stats["lon"], area_stats["lat"],
            s=area_stats["n"] / area_stats["n"].max() * 1800 + 80,
            c=area_stats["n"], cmap="YlOrRd",
            edgecolor="black", linewidth=0.6, alpha=0.9,
        )
        for _, r in area_stats.iterrows():
            ax.annotate(r["Area"], (r["lon"], r["lat"]),
                        fontsize=7, ha="center", va="center")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Incident count (2020–2024)", fontsize=10)
        ax.set_title("LAPD Incidents by Patrol Area\n"
                     "(marker size & colour = incident volume; geo-valid rows only)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = self.fig_dir / "area_choropleth.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Static choropleth -> %s", out)
        return out

    def run(self, df: pd.DataFrame) -> dict:
        return {
            "interactive": self.interactive_map(df),
            "static": self.static_choropleth(df),
        }
