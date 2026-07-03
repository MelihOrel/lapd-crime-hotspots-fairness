"""
src/fairness_audit.py
=====================
The responsible-ML core and the project's differentiator.

What this module does
---------------------
1. Trains a transparent case-clearance classifier (LightGBM, with a logistic
   regression baseline) predicting ``is_cleared`` from OPERATIONAL features only
   — crime type, patrol area, premise, hour of day, weapon-present flag.
   Victim demographics are DELIBERATELY EXCLUDED from the model inputs.

2. Runs a group-fairness audit with Fairlearn across audited groups
   (victim descent, sex, age band):
       * raw (data) clearance rate per group  -> disparity ALREADY in the data
       * model selection rate / FNR per group  -> disparity from the MODEL
       * demographic-parity difference
       * equalized-odds difference
       * false-negative-rate gap
   We explicitly separate data-disparity from model-disparity.

3. Computes SHAP global feature importance for the clearance model.

4. Persists the audit table to ``reports/metrics/fairness_audit.csv``.

Important framing: we are AUDITING a clearance model for demographic disparities,
NOT deploying it to direct policing. See README "Scope & Ethics".
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


class FairnessAuditor:
    def __init__(self, config: dict):
        self.cfg = config
        self.fcfg = config["fairness"]
        self.fig_dir = Path(config["paths"]["figures"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.auc_ = None
        self.gaps_ = {}

    # ------------------------------------------------------- train model
    def _build_pipeline(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        import lightgbm as lgb

        cat = self.fcfg["model_features_cat"]
        pre = ColumnTransformer(
            [("cat",
              OneHotEncoder(handle_unknown="ignore",
                            max_categories=self.fcfg["max_ohe_categories"]),
              cat)],
            remainder="passthrough",
        )
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.fcfg["random_state"], verbose=-1,
        )
        from sklearn.pipeline import Pipeline as P
        return P([("pre", pre), ("model", model)])

    def train(self, df: pd.DataFrame):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        cat = self.fcfg["model_features_cat"]
        num = self.fcfg["model_features_num"]
        feats = cat + num
        groups = self.fcfg["audited_groups"]

        # Drop rows with no parsed hour (can't form the operational feature).
        work = df.dropna(subset=["hour"]).copy()
        work["hour"] = work["hour"].astype(int)

        X = work[feats]
        y = work["is_cleared"]
        S = work[groups]

        Xtr, Xte, ytr, yte, Str, Ste = train_test_split(
            X, y, S,
            test_size=self.fcfg["test_size"],
            random_state=self.fcfg["random_state"],
            stratify=y,
        )

        pipe = self._build_pipeline().fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        self.auc_ = float(roc_auc_score(yte, proba))
        logger.info("Clearance model AUC: %.3f", self.auc_)

        self.pipe_ = pipe
        self.Xte_, self.yte_, self.pred_, self.Ste_ = Xte, yte, pred, Ste
        return pipe

    # ------------------------------------------------------- fairness audit
    def audit(self, df: pd.DataFrame) -> pd.DataFrame:
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference,
            equalized_odds_difference, false_negative_rate,
            selection_rate, true_positive_rate,
        )

        rows = []
        min_n = self.fcfg["min_group_size"]

        for grp in self.fcfg["audited_groups"]:
            sf = self.Ste_[grp]

            # ---- data disparity: raw clearance rate by group (full data) ----
            raw = (
                df.groupby(grp)["is_cleared"]
                .agg(["mean", "size"])
                .rename(columns={"mean": "data_clearance_rate", "size": "n"})
            )
            raw = raw[raw["n"] >= min_n]

            # ---- model disparity: per-group metrics on the test set ----
            mf = MetricFrame(
                metrics={
                    "model_selection_rate": selection_rate,
                    "fnr": false_negative_rate,
                    "tpr": true_positive_rate,
                },
                y_true=self.yte_, y_pred=self.pred_,
                sensitive_features=sf,
            )
            by = mf.by_group

            dpd = demographic_parity_difference(
                self.yte_, self.pred_, sensitive_features=sf)
            eod = equalized_odds_difference(
                self.yte_, self.pred_, sensitive_features=sf)

            # FNR gap across groups (max-min), ignoring NaN groups.
            fnr_vals = by["fnr"].dropna()
            fnr_gap = float(fnr_vals.max() - fnr_vals.min()) if len(fnr_vals) > 1 else np.nan

            self.gaps_[grp] = {
                "demographic_parity_diff": round(float(dpd), 4),
                "equalized_odds_diff": round(float(eod), 4),
                "fnr_gap": round(fnr_gap, 4) if fnr_gap == fnr_gap else None,
                "data_clearance_gap": round(
                    float(raw["data_clearance_rate"].max()
                          - raw["data_clearance_rate"].min()), 4),
            }

            for g_name, r in raw.iterrows():
                m = by.loc[g_name] if g_name in by.index else None
                rows.append({
                    "audited_attribute": grp,
                    "group": g_name,
                    "n": int(r["n"]),
                    "data_clearance_rate": round(float(r["data_clearance_rate"]), 4),
                    "model_selection_rate": (
                        round(float(m["model_selection_rate"]), 4)
                        if m is not None and m["model_selection_rate"] == m["model_selection_rate"]
                        else None),
                    "model_fnr": (
                        round(float(m["fnr"]), 4)
                        if m is not None and m["fnr"] == m["fnr"] else None),
                    "model_tpr": (
                        round(float(m["tpr"]), 4)
                        if m is not None and m["tpr"] == m["tpr"] else None),
                })

            logger.info(
                "Fairness [%s]: data clearance-rate gap=%.1fpp | DPD=%.3f | EOD=%.3f | FNR gap=%s",
                grp,
                self.gaps_[grp]["data_clearance_gap"] * 100,
                dpd, eod,
                self.gaps_[grp]["fnr_gap"],
            )

        table = pd.DataFrame(rows)
        out = self.metrics_dir / "fairness_audit.csv"
        table.to_csv(out, index=False)
        logger.info("Fairness audit table -> %s (%s rows)", out, len(table))
        self.audit_table_ = table
        return table

    # --------------------------------------------------------------- SHAP
    def shap_summary(self) -> Path:
        import shap

        pre = self.pipe_.named_steps["pre"]
        model = self.pipe_.named_steps["model"]
        n = min(self.fcfg["shap_sample"], len(self.Xte_))
        Xs = self.Xte_.iloc[:n]
        Xt = pre.transform(Xs)
        # Densify: SHAP's TreeExplainer + summary_plot expect a dense 2-D array,
        # and a sparse matrix gets mis-handled (np.matrix indexing error).
        Xt = np.asarray(Xt.toarray() if hasattr(Xt, "toarray") else Xt)
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(Xt.shape[1])]

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        # SHAP/LightGBM binary output is version-dependent:
        #   * list of two arrays [class0, class1] -> take the positive class
        #   * single 2-D array                    -> use as-is
        #   * 3-D array (n, features, classes)    -> take the positive class
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        plt.figure()
        shap.summary_plot(
            sv, Xt,
            feature_names=feat_names,
            plot_type="bar", max_display=15, show=False,
        )
        out = self.fig_dir / "clearance_shap.png"
        plt.title("Clearance Model — Global Feature Importance (SHAP)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary -> %s", out)
        return out

    # ------------------------------------------------------------- driver
    def run(self, df: pd.DataFrame) -> dict:
        self.train(df)
        table = self.audit(df)
        shap_path = self.shap_summary()
        return {"auc": self.auc_, "table": table,
                "gaps": self.gaps_, "shap": shap_path}
