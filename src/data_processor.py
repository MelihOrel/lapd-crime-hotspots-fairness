"""
src/data_processor.py
=====================
Data engineering & cleaning for the LAPD crime dataset.

Responsibilities
----------------
1. Load the raw CSV (low_memory=False).
2. Combine ``Date_occured`` + ``Time_occured`` into a single datetime and derive
   hour / day-of-week / month / year. The source mixes two date formats
   (``%m/%d/%y`` and ``%m/%d/%Y``) and stores the time of day separately as an
   HHMM integer, so we rebuild a clean ``HH:MM`` string and let pandas parse the
   mixed formats.
3. Null out impossible victim ages (outside [age_min, age_max]) and log the count.
4. Flag phantom (0, 0) coordinates (and out-of-region points) so geospatial steps
   can exclude them, while keeping the rows for non-spatial counts.
5. Produce an explicit per-column missingness report. Sensitive fields
   (Weapon / Victim_sex / Victim_descent) are NOT imputed — missing values become
   an explicit "Unknown" category.
6. Derive a binary ``is_cleared`` target from ``Status`` per the documented mapping.
7. Persist the cleaned frame to parquet.

Every transformation is logged so the pipeline can print a transparent
data-quality report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Structured record of what cleaning actually did (for transparent logging)."""

    n_rows_raw: int = 0
    n_impossible_ages: int = 0
    n_phantom_coords: int = 0
    n_out_of_region: int = 0
    n_geo_valid: int = 0
    n_cleared: int = 0
    clearance_rate: float = 0.0
    null_summary: Dict[str, float] = field(default_factory=dict)

    def as_lines(self) -> list[str]:
        lines = [
            f"Rows loaded                : {self.n_rows_raw:,}",
            f"Impossible ages nulled     : {self.n_impossible_ages:,}",
            f"Phantom (0,0) coords flagged: {self.n_phantom_coords:,}",
            f"Out-of-region coords flagged: {self.n_out_of_region:,}",
            f"Geo-valid rows (for maps)  : {self.n_geo_valid:,}",
            f"Cleared incidents          : {self.n_cleared:,} "
            f"({self.clearance_rate:.2%})",
        ]
        return lines


class LAPDDataProcessor:
    """Loads and cleans the LAPD incident CSV per the project's config."""

    def __init__(self, config: dict):
        self.cfg = config
        self.clean_cfg = config["cleaning"]
        self.report = CleaningReport()

    # ------------------------------------------------------------------ load
    def load(self, csv_path: str | Path) -> pd.DataFrame:
        logger.info("Loading raw CSV: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        self.report.n_rows_raw = len(df)
        logger.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])
        return df

    # -------------------------------------------------------------- datetime
    @staticmethod
    def _build_datetime(df: pd.DataFrame) -> pd.Series:
        """Combine the date part of ``Date_occured`` with the HHMM ``Time_occured``.

        ``Date_occured`` looks like ``03/01/20 0:00`` or ``08/17/2020 0:00`` — the
        embedded time is always midnight and meaningless; the real time of day is
        the separate integer ``Time_occured`` (e.g. 2130 -> 21:30, 1 -> 00:01).
        """
        date_part = df["Date_occured"].astype(str).str.split(" ").str[0]
        hhmm = df["Time_occured"].fillna(0).astype(int).clip(0, 2359)
        hhmm = hhmm.astype(str).str.zfill(4)
        time_part = hhmm.str[:2] + ":" + hhmm.str[2:]
        combined = date_part + " " + time_part
        # format="mixed" handles both 2-digit (%y) and 4-digit (%Y) years.
        return pd.to_datetime(combined, format="mixed", errors="coerce")

    # --------------------------------------------------------------- cleaning
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- 1. datetime ----------------------------------------------------
        df["datetime"] = self._build_datetime(df)
        n_bad_dt = int(df["datetime"].isna().sum())
        if n_bad_dt:
            logger.warning("%s rows had unparseable datetimes (left as NaT)", n_bad_dt)
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek            # 0=Mon
        df["dayname"] = df["datetime"].dt.day_name()
        df["month"] = df["datetime"].dt.month
        df["year"] = df["datetime"].dt.year

        # --- 2. impossible victim ages -------------------------------------
        amin, amax = self.clean_cfg["age_min"], self.clean_cfg["age_max"]
        bad_age = (df["Victim_age"] < amin) | (df["Victim_age"] > amax)
        self.report.n_impossible_ages = int(bad_age.sum())
        df.loc[bad_age, "Victim_age"] = np.nan
        logger.info(
            "Nulled %s impossible victim ages (outside [%s, %s])",
            f"{self.report.n_impossible_ages:,}", amin, amax,
        )

        # Age band (used as an audited group, NOT a model input). NaN -> Unknown.
        df["age_band"] = pd.cut(
            df["Victim_age"],
            bins=[-0.1, 12, 17, 25, 40, 60, 100],
            labels=["0-12", "13-17", "18-25", "26-40", "41-60", "61-100"],
        ).cat.add_categories(["Unknown"]).fillna("Unknown")

        # --- 3. phantom & out-of-region coordinates ------------------------
        bbox = self.clean_cfg["la_bbox"]
        phantom = (df["LAT"] == 0.0) & (df["LON"] == 0.0)
        self.report.n_phantom_coords = int(phantom.sum())
        in_box = (
            df["LAT"].between(bbox["lat_min"], bbox["lat_max"])
            & df["LON"].between(bbox["lon_min"], bbox["lon_max"])
        )
        # geo_valid = real LA coordinates only. Phantom rows fail in_box anyway,
        # but we count them separately for the report.
        df["geo_valid"] = in_box & ~phantom
        self.report.n_out_of_region = int((~in_box & ~phantom).sum())
        self.report.n_geo_valid = int(df["geo_valid"].sum())
        logger.info(
            "Flagged %s phantom (0,0) coords + %s out-of-region; %s geo-valid rows kept for mapping",
            f"{self.report.n_phantom_coords:,}",
            f"{self.report.n_out_of_region:,}",
            f"{self.report.n_geo_valid:,}",
        )

        # --- 4. explicit Unknown for sensitive / sparse categoricals -------
        tok = self.clean_cfg["unknown_token"]
        for col in ["Victim_sex", "Victim_descent", "Weapon", "Premis"]:
            df[col] = df[col].fillna(tok).replace({"-": tok, "": tok})

        # weapon-present flag (operational feature) — derived BEFORE any impute,
        # "Unknown" means no weapon was recorded.
        df["weapon_present"] = (df["Weapon"] != tok).astype(int)

        # --- 5. clearance target -------------------------------------------
        cleared_set = set(self.cfg["clearance"]["cleared_statuses"])
        df["is_cleared"] = df["Status"].isin(cleared_set).astype(int)
        self.report.n_cleared = int(df["is_cleared"].sum())
        self.report.clearance_rate = float(df["is_cleared"].mean())
        logger.info(
            "Derived is_cleared: %s cleared (%.2f%%) using statuses %s",
            f"{self.report.n_cleared:,}",
            self.report.clearance_rate * 100,
            sorted(cleared_set),
        )

        # --- 6. missingness report (on ORIGINAL raw nullness) --------------
        self.report.null_summary = (
            df[["Weapon", "Victim_sex", "Victim_descent", "Premis"]]
            .eq(tok)
            .mean()
            .round(4)
            .to_dict()
        )
        return df

    # --------------------------------------------------------------- persist
    def save(self, df: pd.DataFrame, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved cleaned frame -> %s (%s rows)", out_path, f"{len(df):,}")

    # ----------------------------------------------------------------- driver
    def run(self) -> pd.DataFrame:
        df = self.load(self.cfg["paths"]["raw_csv"])
        df = self.clean(df)
        self.save(df, self.cfg["paths"]["clean_parquet"])
        return df
