"""
main.py
=======
Entry point orchestrating the full LAPD hotspot + fairness-audit pipeline:

    Load & Clean  ->  Spatiotemporal Hotspots  ->  Maps
                  ->  Train Clearance Model  ->  Fairness Audit
                  ->  Visualizations  ->  Final summary

Run:
    python main.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.data_processor import LAPDDataProcessor
from src.hotspots import HotspotAnalyzer
from src.mapping import MapBuilder
from src.fairness_audit import FairnessAuditor
from src.visualize import Visualizer


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def banner(title: str) -> None:
    bar = "=" * 78
    logging.info(bar)
    logging.info(title)
    logging.info(bar)


def main(config_path: str) -> None:
    setup_logging()
    log = logging.getLogger("main")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------- 1. Load & clean
    banner("STEP 1/6 — LOAD & CLEAN (with data-quality report)")
    processor = LAPDDataProcessor(cfg)
    df = processor.run()
    rep = processor.report
    log.info(
        "Cleaned %s rows | %s impossible ages nulled | %s phantom (0,0) coords excluded from mapping",
        f"{rep.n_rows_raw:,}", f"{rep.n_impossible_ages:,}",
        f"{rep.n_phantom_coords:,}",
    )
    banner("DATA-QUALITY REPORT")
    for line in rep.as_lines():
        log.info(line)
    log.info("Missingness (post-Unknown encoding): %s",
             {k: f"{v:.1%}" for k, v in rep.null_summary.items()})

    # ----------------------------------------------- 2. Hotspot analysis
    banner("STEP 2/6 — SPATIOTEMPORAL HOTSPOT ANALYSIS")
    analyzer = HotspotAnalyzer(cfg)
    hotspot_results = analyzer.run(df)
    log.info("Gi* hotspots: %s statistically significant hot clusters (p<%.2f)",
             analyzer.n_hot, cfg["hotspots"]["significance"])

    # ------------------------------------------------------- 3. Maps
    banner("STEP 3/6 — GENERATE MAPS")
    mapper = MapBuilder(cfg)
    map_paths = mapper.run(df)

    # --------------------------------------------- 4-5. Model + Audit
    banner("STEP 4/6 — TRAIN CLEARANCE MODEL & RUN FAIRNESS AUDIT")
    auditor = FairnessAuditor(cfg)
    audit_results = auditor.run(df)
    log.info("Clearance model AUC: %.2f", audit_results["auc"])
    for grp, g in audit_results["gaps"].items():
        log.info(
            "Fairness [%s]: data clearance-rate gap of %.1f pp between highest/lowest groups "
            "| DPD=%.3f | EOD=%.3f — see audit table",
            grp, g["data_clearance_gap"] * 100,
            g["demographic_parity_diff"], g["equalized_odds_diff"],
        )

    # ------------------------------------------------ 6. Visualizations
    banner("STEP 5/6 — GENERATE VISUALIZATIONS")
    viz = Visualizer(cfg)
    fig_paths = viz.run(hotspot_results, audit_results["table"])

    # --------------------------------------------------- Final summary
    banner("STEP 6/6 — PIPELINE COMPLETE — SUMMARY")
    log.info("Rows analysed            : %s", f"{rep.n_rows_raw:,}")
    log.info("Geo-valid rows (mapped)  : %s", f"{rep.n_geo_valid:,}")
    log.info("Gi* hot / cold spots     : %s / %s", analyzer.n_hot, analyzer.n_cold)
    log.info("Clearance model AUC      : %.3f", audit_results["auc"])
    log.info("Interactive map          : %s", map_paths["interactive"])
    log.info("Static choropleth        : %s", map_paths["static"])
    log.info("Figures                  : %s",
             ", ".join(str(p.name) for p in fig_paths.values()))
    log.info("Metrics CSVs             : reports/metrics/{hotspots,crime_by_area,fairness_audit}.csv")
    log.info("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LAPD hotspot + fairness-audit pipeline")
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
