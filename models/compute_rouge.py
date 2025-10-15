from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def compute_rouge_for_csv(csv_path: str, reference_col: str, model_cols: List[str]) -> pd.DataFrame:
    """Compute ROUGE-1/2/Lsum (F1) for each model column against the reference column.

    Returns a DataFrame with columns: model, rouge1_f1, rouge2_f1, rougeLsum_f1, n
    """
    try:
        import evaluate  # lazy import to avoid import cost if unused
    except Exception as e:
        raise RuntimeError(
            "The 'evaluate' package is required. Please install with: pip install evaluate rouge-score"
        ) from e

    df = pd.read_csv(csv_path)
    metric = evaluate.load("rouge")  # supports rouge1, rouge2, rougeLsum

    results = []
    for col in model_cols:
        if col not in df.columns:
            results.append({
                "model": col,
                "rouge1_f1": None,
                "rouge2_f1": None,
                "rougeLsum_f1": None,
                "n": 0,
                "note": "column not found"
            })
            continue

        pairs = df[[reference_col, col]].dropna()
        refs = pairs[reference_col].astype(str).tolist()
        preds = pairs[col].astype(str).tolist()

        if not preds or not refs:
            results.append({
                "model": col,
                "rouge1_f1": None,
                "rouge2_f1": None,
                "rougeLsum_f1": None,
                "n": 0
            })
            continue

        scores = metric.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rouge1", "rouge2", "rougeLsum"],
            use_stemmer=True,
        )
        results.append({
            "model": col,
            "rouge1_f1": float(scores.get("rouge1", 0.0)),
            "rouge2_f1": float(scores.get("rouge2", 0.0)),
            "rougeLsum_f1": float(scores.get("rougeLsum", 0.0)),
            "n": len(preds)
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE-1/2/Lsum (F1) for summaries in a CSV")
    parser.add_argument(
        "--task",
        choices=["interview", "reading"],
        help="Use project defaults for the selected task (paths and columns)",
    )
    parser.add_argument("--csv_path", type=str, help="Path to CSV file", default="")
    parser.add_argument("--reference_col", type=str, help="Reference summary column name", default="")
    parser.add_argument(
        "--model_cols",
        nargs="+",
        help="One or more model summary column names",
        default=[],
    )
    parser.add_argument("--output_dir", type=str, default="", help="Directory to write outputs")
    parser.add_argument("--output_prefix", type=str, default="rouge", help="Base name for outputs")
    args = parser.parse_args()

    # Project defaults (absolute paths) for convenience
    base = "/orange/ufdatastudios/c.okocha/child__speech_analysis"
    interview_defaults = {
        "csv_path": f"{base}/results/Interview/interview_combined.enriched.csv",
        "reference_col": "Llama_summary",
        "model_cols": ["AF3_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"],
        "output_dir": f"{base}/results/Interview/ROUGE",
    }
    reading_defaults = {
        "csv_path": f"{base}/results/reading/reading_combined.enriched.csv",
        "reference_col": "LlamaRead_summary",
        "model_cols": ["AF35_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"],
        "output_dir": f"{base}/results/reading/ROUGE",
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
        # Custom inputs
        if not args.csv_path or not args.reference_col or not args.model_cols:
            raise SystemExit("Provide --task or specify --csv_path, --reference_col, and --model_cols")
        csv_path = args.csv_path
        reference_col = args.reference_col
        model_cols = args.model_cols
        output_dir = args.output_dir or str(Path(csv_path).parent / "ROUGE")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_csv = Path(output_dir) / f"{args.output_prefix}.csv"
    out_json = Path(output_dir) / f"{args.output_prefix}.json"

    table = compute_rouge_for_csv(csv_path, reference_col, model_cols)
    table.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2)

    print(f"Saved ROUGE results to:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()


