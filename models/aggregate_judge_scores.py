from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import json


JUDGES = ["Llama", "Qwen", "Mistral"]
READING_MODELS = ["AF35_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"]
INTERVIEW_MODELS = ["AF3_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"]
SCORE_COLS = [
    "score_overall",
    "score_fluency",
    "score_faithfulness",
    "score_coverage",
    "score_purity",
    "score_usefulness",
]


def load_eval_csvs(base_dir: Path, task: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all judge CSVs for a task into a dict: judge -> model_name -> df."""
    task_dir = base_dir / task
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for judge in JUDGES:
        judge_dir = task_dir / judge
        if not judge_dir.exists():
            continue
        out[judge] = {}
        for csv_path in judge_dir.rglob("*.csv"):
            name = csv_path.stem  # e.g., AF35_summary_evaluations or AF35_summary_qwen2_eval
            # infer model name by taking prefix before first judge/id token
            model_name = name.split("_")[0]
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            # Ensure only score columns are present among others
            if not any(c in df.columns for c in SCORE_COLS):
                continue
            out[judge][model_name] = df
    return out


def average_scores_per_model(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, df in dfs.items():
        row = {"model": model_name}
        for col in SCORE_COLS:
            if col in df.columns and not df[col].dropna().empty:
                row[col] = float(df[col].mean())
            else:
                row[col] = None
        rows.append(row)
    return pd.DataFrame(rows)


def combine_across_judges(data: Dict[str, Dict[str, pd.DataFrame]], model_list: List[str]) -> pd.DataFrame:
    """Return a table with averages per judge and overall, per model."""
    # Build per-judge averages
    judge_tables = {}
    for judge, model_to_df in data.items():
        judge_tables[judge] = average_scores_per_model(model_to_df)
        judge_tables[judge]["judge"] = judge

    # Concatenate and pivot to have judge as a level in columns
    combined = None
    if judge_tables:
        combined = pd.concat(judge_tables.values(), ignore_index=True)
    else:
        return pd.DataFrame()

    # Ensure all requested models are present; if missing, add NaN rows
    for m in model_list:
        if m not in combined["model"].values:
            combined = pd.concat([combined, pd.DataFrame([{"model": m}])], ignore_index=True)

    # Compute overall averages across judges per model
    overall = (
        combined.groupby("model")[SCORE_COLS].mean(numeric_only=True).reset_index()
    )
    overall["judge"] = "overall"

    # Merge judge-wise and overall
    final = pd.concat([combined, overall], ignore_index=True)
    # Sort for readability
    final = final.sort_values(by=["judge", "model"], na_position="last")
    return final


def to_nested_json(table: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, row in table.iterrows():
        judge = str(row.get("judge", "")).lower() or "unknown"
        model = str(row.get("model", "unknown"))
        out.setdefault(judge, {})[model] = {k: (None if pd.isna(row.get(k)) else float(row.get(k))) for k in SCORE_COLS}
    return out


def main():
    ap = argparse.ArgumentParser(description="Aggregate LLM-as-a-Judge scores across judges and models")
    ap.add_argument("--eval_base_dir", type=str,
                    default="/orange/ufdatastudios/c.okocha/child__speech_analysis/Evaluation/LLM_Eval",
                    help="Base dir holding Reading/ and Interview/ judge results")
    ap.add_argument("--out_dir", type=str,
                    default="/orange/ufdatastudios/c.okocha/child__speech_analysis/results/LLM_Judge_Aggregates",
                    help="Output directory for combined tables")
    args = ap.parse_args()

    base = Path(args.eval_base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reading
    reading_data = load_eval_csvs(base, "Reading")
    reading_table = combine_across_judges(reading_data, READING_MODELS)
    if not reading_table.empty:
        reading_csv = out_dir / "reading_judge_aggregate.csv"
        reading_json = out_dir / "reading_judge_aggregate.json"
        reading_table.to_csv(reading_csv, index=False)
        with open(reading_json, "w", encoding="utf-8") as f:
            json.dump(to_nested_json(reading_table), f, indent=2)
        print(f"Saved reading aggregates to:\n  {reading_csv}\n  {reading_json}")
    else:
        print("No reading judge CSVs found or parsed.")

    # Interview
    interview_data = load_eval_csvs(base, "Interview")
    interview_table = combine_across_judges(interview_data, INTERVIEW_MODELS)
    if not interview_table.empty:
        interview_csv = out_dir / "interview_judge_aggregate.csv"
        interview_json = out_dir / "interview_judge_aggregate.json"
        interview_table.to_csv(interview_csv, index=False)
        with open(interview_json, "w", encoding="utf-8") as f:
            json.dump(to_nested_json(interview_table), f, indent=2)
        print(f"Saved interview aggregates to:\n  {interview_csv}\n  {interview_json}")
    else:
        print("No interview judge CSVs found or parsed.")


if __name__ == "__main__":
    main()


