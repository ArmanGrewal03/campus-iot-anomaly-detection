#!/usr/bin/env python3
"""
Generate a testing report with text, figures, tables, and charts for the Campus IoT Anomaly Detection project.

Outputs (saved under --out-dir):
- metrics_table.md: Markdown table of core metrics per model
- fig_cm_<model>.png: Confusion matrix (if available)
- fig_roc_<model>.png: ROC curve (if y_true/y_prob_unsafe available)
- fig_pr_<model>.png: Precision–Recall curve (if y_true/y_prob_unsafe available)
- fig_calib_<model>.png: Calibration curve (if y_true/y_prob_unsafe available)
- fig_if_scores.png: Isolation Forest score histogram (if IF metadata + scores available)
- fig_ae_errors.png: Autoencoder reconstruction error histogram (if AE metadata + errors available)
- report.md: Compact poster-friendly summary with links to figures

This script works in two modes:
1) Online: Calls the Gateway /test endpoint to evaluate a model and read back metrics.
2) Offline: Reads the latest *_metadata.json from 02_Model_Service/models and renders available charts.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    sns.set()
    SEABORN_OK = True
except Exception:
    SEABORN_OK = False

try:
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
    from sklearn.calibration import calibration_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "02_Model_Service" / "models"


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metadata_by_model_name(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata JSON by sanitized model name.
    """
    sanitized = model_name.replace("/", "_").replace("\\", "_").replace("..", "_")
    meta_path = MODEL_DIR / f"{sanitized}_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_metadata(prefix: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Find the most recently modified metadata file that starts with prefix, e.g., 'rf', 'if', 'ae'.
    """
    if not MODEL_DIR.exists():
        return None
    candidates: List[Path] = sorted(MODEL_DIR.glob(f"{prefix}*_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    name = candidates[0].name.replace("_metadata.json", "")
    with open(candidates[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return name, meta


def call_gateway_test(gateway_url: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    POST /test via Gateway to compute fresh metrics. Requires requests.
    """
    if not REQUESTS_OK:
        return None
    url = f"{gateway_url.rstrip('/')}/test"
    headers = {"model_name": model_name}
    try:
        resp = requests.post(url, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def save_confusion_matrix(fig_dir: Path, model_key: str, cm: List[List[int]]) -> Optional[Path]:
    try:
        arr = np.array(cm, dtype=int)
        plt.figure(figsize=(6, 5), dpi=220)
        if SEABORN_OK:
            import seaborn as sns  # local import for mypy
            sns.heatmap(arr, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Safe", "Unsafe"], yticklabels=["Safe", "Unsafe"])
        else:
            plt.imshow(arr, cmap="Blues")
            for (i, j), z in np.ndenumerate(arr):
                plt.text(j, i, str(z), ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Safe", "Unsafe"])
            plt.yticks([0, 1], ["Safe", "Unsafe"])
        plt.title(f"{model_key} Confusion Matrix (Test)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        out = fig_dir / f"fig_cm_{model_key}.png"
        plt.savefig(out)
        plt.close()
        return out
    except Exception:
        return None


def save_roc_pr_calibration(fig_dir: Path, model_key: str, y_true: np.ndarray, y_prob: np.ndarray) -> List[Path]:
    outs: List[Path] = []
    if not SKLEARN_OK:
        return outs
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5), dpi=220)
        plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        plt.plot([0, 1], [0, 1], 'k--', alpha=.4)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"{model_key} ROC")
        plt.legend()
        plt.tight_layout()
        out = fig_dir / f"fig_roc_{model_key}.png"
        plt.savefig(out)
        plt.close()
        outs.append(out)
    except Exception:
        pass
    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5), dpi=220)
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{model_key} Precision–Recall")
        plt.tight_layout()
        out = fig_dir / f"fig_pr_{model_key}.png"
        plt.savefig(out)
        plt.close()
        outs.append(out)
    except Exception:
        pass
    # Calibration
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        plt.figure(figsize=(6, 5), dpi=220)
        plt.plot([0, 1], [0, 1], 'k--', alpha=.4, label="Perfectly calibrated")
        plt.plot(prob_pred, prob_true, marker="o", label="Observed")
        plt.xlabel("Predicted probability (unsafe)")
        plt.ylabel("Observed frequency (unsafe)")
        plt.title(f"{model_key} Calibration")
        plt.legend()
        plt.tight_layout()
        out = fig_dir / f"fig_calib_{model_key}.png"
        plt.savefig(out)
        plt.close()
        outs.append(out)
    except Exception:
        pass
    return outs


def save_histogram(fig_dir: Path, title: str, data: np.ndarray, xlabel: str, out_name: str, threshold: Optional[float] = None) -> Optional[Path]:
    try:
        plt.figure(figsize=(6, 5), dpi=220)
        plt.hist(data, bins=40, color="#888", edgecolor="#333")
        if threshold is not None:
            plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold:.4f}")
            plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        out = fig_dir / out_name
        plt.savefig(out)
        plt.close()
        return out
    except Exception:
        return None


def write_metrics_table(md_path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            line = [
                str(r.get("Model", "")),
                _fmt_num(r.get("Accuracy")),
                _fmt_num(r.get("Precision")),
                _fmt_num(r.get("Recall")),
                _fmt_num(r.get("F1")),
                _fmt_num(r.get("AUC_ROC")),
                _fmt_num(r.get("AUC_PR")),
            ]
            f.write("| " + " | ".join(line) + " |\n")


def _fmt_num(x: Any) -> str:
    try:
        if x is None:
            return ""
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def build_report_md(out_dir: Path, metrics_rows: List[Dict[str, Any]], figures: List[Path]) -> None:
    md = out_dir / "report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Testing Report\n\nGenerated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("## Summary Table\n\n")
        f.write("(see metrics_table.md for a copyable version)\n\n")
        # quick inline table
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in metrics_rows:
            line = [
                str(r.get("Model", "")),
                _fmt_num(r.get("Accuracy")),
                _fmt_num(r.get("Precision")),
                _fmt_num(r.get("Recall")),
                _fmt_num(r.get("F1")),
                _fmt_num(r.get("AUC_ROC")),
                _fmt_num(r.get("AUC_PR")),
            ]
            f.write("| " + " | ".join(line) + " |\n")
        f.write("\n## Figures\n\n")
        for p in figures:
            rel = p.name
            f.write(f"![{rel}]({rel})\n\n")


def extract_basic_metrics(model_key: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Model": model_key,
        "Accuracy": metrics.get("accuracy"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "F1": metrics.get("f1_score"),
        "AUC_ROC": metrics.get("auc_roc"),
        "AUC_PR": metrics.get("auc_pr"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate testing report with figures and tables.")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8003", help="Gateway base URL for /test")
    parser.add_argument("--models", nargs="*", default=["rf_latest", "if_latest", "ae_latest"], help="Model names to evaluate/read")
    parser.add_argument("--out-dir", default="test_report", help="Output directory for figures and markdown")
    parser.add_argument("--offline-only", action="store_true", help="Do not call the gateway; read metadata only")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    all_rows: List[Dict[str, Any]] = []
    all_figs: List[Path] = []

    for model_name in args.models:
        # 1) Try online test (if allowed)
        test_payload = None
        if not args.offline_only:
            test_payload = call_gateway_test(args.gateway_url, model_name)
        # 2) Load metadata
        meta = load_metadata_by_model_name(model_name)
        metrics: Dict[str, Any] = {}
        if test_payload and isinstance(test_payload, dict):
            metrics = test_payload.get("metrics", {}) or {}
        if not metrics and meta:
            metrics = meta.get("metrics", {}) or {}

        model_key = model_name
        if metrics:
            all_rows.append(extract_basic_metrics(model_key, metrics))
            # Confusion matrix
            cm = metrics.get("confusion_matrix")
            if cm:
                p = save_confusion_matrix(out_dir, model_key, cm)
                if p:
                    all_figs.append(p)

            # ROC/PR/Calibration if arrays present
            y_true = metrics.get("y_true")
            y_prob = metrics.get("y_prob_unsafe")
            if y_true and y_prob:
                try:
                    y_t = np.array(y_true, dtype=int)
                    y_p = np.array(y_prob, dtype=float)
                    figs = save_roc_pr_calibration(out_dir, model_key, y_t, y_p)
                    all_figs.extend(figs)
                except Exception:
                    pass

        # IF extras
        if meta and model_key.lower().startswith("if"):
            training_params = (meta.get("training_params") or {})
            thr = training_params.get("if_decision_threshold", 0.0)
            scores = metrics.get("if_test_scores") or []
            if scores:
                data = np.array(scores, dtype=float)
                p = save_histogram(out_dir, "IFv1 Score Distribution (Test)", data, "Score (higher=more anomalous)", "fig_if_scores.png", threshold=float(thr))
                if p:
                    all_figs.append(p)

        # AE extras
        if meta and model_key.lower().startswith("ae"):
            thr = (meta.get("metrics") or {}).get("threshold")
            errs = metrics.get("ae_test_errors") or []
            if errs:
                data = np.array(errs, dtype=float)
                p = save_histogram(out_dir, "AEv1 Reconstruction Error (Test)", data, "MSE", "fig_ae_errors.png", threshold=float(thr) if thr is not None else None)
                if p:
                    all_figs.append(p)

    # Write tables and report
    write_metrics_table(out_dir / "metrics_table.md", all_rows)
    build_report_md(out_dir, all_rows, all_figs)

    print(f"Report generated at: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

