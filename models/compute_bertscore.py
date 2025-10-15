from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 13) -> Tuple[float, float]:
    """Non-parametric bootstrap 95% CI for the mean."""
    if not values:
        return (None, None)
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        return (None, None)
    samples = rng.choice(vals, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def compute_bertscore_for_csv(
    csv_path: str,
    reference_col: str,
    model_cols: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    bootstrap: bool = True,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Compute BERTScore P/R/F1 for each model column against the reference column.

    Returns a DataFrame with per-model averages and optional 95% bootstrap CIs.
    """
    try:
        import evaluate  # lazy import
    except Exception as e:
        raise RuntimeError(
            "The 'evaluate' package is required. Install: pip install evaluate bert-score"
        ) from e

    df = pd.read_csv(csv_path)
    metric = evaluate.load("bertscore")

    rows = []
    for col in model_cols:
        if col not in df.columns:
            rows.append({
                "model": col,
                "precision": None, "recall": None, "f1": None,
                "f1_ci_low": None, "f1_ci_high": None,
                "n": 0, "note": "column not found"
            })
            continue

        pairs = df[[reference_col, col]].dropna()
        refs = pairs[reference_col].astype(str).tolist()
        preds = pairs[col].astype(str).tolist()

        if not preds or not refs:
            rows.append({
                "model": col,
                "precision": None, "recall": None, "f1": None,
                "f1_ci_low": None, "f1_ci_high": None,
                "n": 0
            })
            continue

        scores = metric.compute(
            predictions=preds,
            references=refs,
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=True,
        )
        # scores keys: precision, recall, f1 (lists)
        p = np.array(scores.get("precision", []), dtype=float)
        r = np.array(scores.get("recall", []), dtype=float)
        f = np.array(scores.get("f1", []), dtype=float)

        f1_ci_low = f1_ci_high = None
        if bootstrap and len(f) > 1:
            f1_ci_low, f1_ci_high = _bootstrap_ci(f.tolist(), n_boot=n_boot)

        rows.append({
            "model": col,
            "precision": float(np.mean(p)) if p.size else None,
            "recall": float(np.mean(r)) if r.size else None,
            "f1": float(np.mean(f)) if f.size else None,
            "f1_ci_low": f1_ci_low,
            "f1_ci_high": f1_ci_high,
            "n": len(f),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute BERTScore for summaries in a CSV")
    parser.add_argument("--task", choices=["interview", "reading"], help="Use project defaults", default=None)
    parser.add_argument("--csv_path", type=str, default="", help="Path to CSV file")
    parser.add_argument("--reference_col", type=str, default="", help="Reference summary column name")
    parser.add_argument("--model_cols", nargs="+", default=[], help="One or more model summary column names")
    parser.add_argument("--model_type", type=str, default="microsoft/deberta-xlarge-mnli", help="HF model id")
    parser.add_argument("--lang", type=str, default="en", help="Language for BERTScore")
    parser.add_argument("--no_bootstrap", action="store_true", help="Disable bootstrap CI")
    parser.add_argument("--n_boot", type=int, default=2000, help="# bootstrap samples")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to write outputs")
    parser.add_argument("--output_prefix", type=str, default="bertscore", help="Base name for outputs")
    args = parser.parse_args()

    base = "/orange/ufdatastudios/c.okocha/child__speech_analysis"
    interview_defaults = {
        "csv_path": f"{base}/results/Interview/interview_combined.enriched.csv",
        "reference_col": "Llama_summary",
        "model_cols": ["AF3_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"],
        "output_dir": f"{base}/results/Interview/BERTScore",
    }
    reading_defaults = {
        "csv_path": f"{base}/results/reading/reading_combined.enriched.csv",
        "reference_col": "LlamaRead_summary",
        "model_cols": ["AF35_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"],
        "output_dir": f"{base}/results/reading/BERTScore",
    }

    if args.task == "interview":
        csv_path = interview_defaults["csv_path"]
        reference_col = interview_defaults["reference_col"]
        model_cols = interview_defaults["model_cols"]
        output_dir = interview_defaults["output_dir"]
    elif args.task == "reading":
        csv_path = reading_defaults["csv_path"]
        reference_col = reading_defaults["reference_col"]
        model_cols = reading_defaults["model_cols"]
        output_dir = reading_defaults["output_dir"]
    else:
        if not args.csv_path or not args.reference_col or not args.model_cols:
            raise SystemExit("Provide --task or specify --csv_path, --reference_col, and --model_cols")
        csv_path = args.csv_path
        reference_col = args.reference_col
        model_cols = args.model_cols
        output_dir = args.output_dir or str(Path(csv_path).parent / "BERTScore")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(output_dir) / f"{args.output_prefix}.csv"
    out_json = Path(output_dir) / f"{args.output_prefix}.json"

    table = compute_bertscore_for_csv(
        csv_path=csv_path,
        reference_col=reference_col,
        model_cols=model_cols,
        model_type=args.model_type,
        lang=args.lang,
        bootstrap=(not args.no_bootstrap),
        n_boot=args.n_boot,
    )

    table.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)

    print(f"Saved BERTScore results to:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()



