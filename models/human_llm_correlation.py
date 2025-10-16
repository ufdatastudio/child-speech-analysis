#!/usr/bin/env python3
"""
Human-LLM Evaluation Correlation Analysis

This script computes correlations and agreement metrics between human evaluations
and LLM-as-a-Judge evaluations for both reading and interview tasks.

Metrics computed:
- Pearson correlation coefficients
- Kendall's tau rank correlation
- Cohen's kappa for agreement
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


# Define the scoring criteria
SCORE_COLS = [
    "score_overall",
    "score_fluency", 
    "score_faithfulness",
    "score_coverage",
    "score_purity",
    "score_usefulness"
]

# Define LLM judges
LLM_JUDGES = ["Llama", "Mistral", "Qwen"]

# Define models for each task
INTERVIEW_MODELS = ["AF3_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"]
READING_MODELS = ["AF35_summary", "Qwen_summary", "Kimi_summary", "Salmon_summary", "GAMA_summary"]


def load_human_evaluations(human_dir: Path, task: str) -> Dict[str, pd.DataFrame]:
    """Load human evaluation CSV files for a task."""
    task_dir = human_dir / f"{task.title()}_Human"
    if not task_dir.exists():
        return {}
    
    human_evals = {}
    for csv_file in task_dir.glob("*_evaluations.csv"):
        model_name = csv_file.stem.replace("_evaluations", "")
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"Successfully loaded {csv_file} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"Could not load {csv_file} with any encoding")
                continue
            
            # Check for required columns - be flexible about which score columns are present
            required_cols = ["audio_id"]
            available_score_cols = [col for col in SCORE_COLS if col in df.columns]
            
            if all(col in df.columns for col in required_cols) and available_score_cols:
                # Add missing score columns with NaN values
                for col in SCORE_COLS:
                    if col not in df.columns:
                        df[col] = np.nan
                        print(f"Added missing column {col} to {model_name}")
                
                human_evals[model_name] = df
                print(f"Loaded human evaluations for {model_name}: {len(df)} samples")
                print(f"Available score columns: {available_score_cols}")
            else:
                print(f"Warning: Missing required columns in {csv_file}")
                print(f"Available columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    return human_evals


def load_llm_evaluations(llm_dir: Path, task: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load LLM evaluation results for a task."""
    # Map task names to directory names
    task_mapping = {
        "interview": "Interview",
        "reading": "Reading"
    }
    
    actual_task_dir = task_mapping.get(task, task.title())
    task_dir = llm_dir / actual_task_dir
    print(f"Looking for LLM evaluations in: {task_dir}")
    
    if not task_dir.exists():
        print(f"Task directory does not exist: {task_dir}")
        return {}
    
    llm_evals = {}
    for judge in LLM_JUDGES:
        judge_dir = task_dir / judge
        if not judge_dir.exists():
            continue
        
        llm_evals[judge] = {}
        
        # Look for CSV files in the judge directory
        for csv_file in judge_dir.rglob("*.csv"):
            if not csv_file.name.endswith(("_evaluations.csv", "_eval.csv")):
                continue
            
            # Extract model name from filename
            model_name = csv_file.stem.replace("_evaluations", "").replace("_eval", "")
            # Remove judge suffix if present
            for judge_suffix in ["_llama", "_mistral", "_qwen", "_qwen2"]:
                if model_name.endswith(judge_suffix):
                    model_name = model_name[:-len(judge_suffix)]
                    break
            
            try:
                df = pd.read_csv(csv_file)
                if all(col in df.columns for col in SCORE_COLS + ["audio_id"]):
                    llm_evals[judge][model_name] = df
                    print(f"Loaded {judge} evaluations for {model_name}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    return llm_evals


def merge_evaluations(human_df: pd.DataFrame, llm_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Merge human and LLM evaluations on audio_id."""
    merged = {}
    
    for judge, llm_df in llm_dfs.items():
        # Merge on audio_id
        merged_df = pd.merge(human_df, llm_df, on="audio_id", suffixes=("_human", f"_{judge}"))
        
        # Check if we have overlapping samples
        if len(merged_df) == 0:
            print(f"Warning: No overlapping audio_ids between human and {judge} evaluations")
            continue
        
        print(f"Merged {judge}: {len(merged_df)} overlapping samples")
        merged[judge] = merged_df
    
    return merged


def compute_correlation_metrics(human_scores: np.ndarray, llm_scores: np.ndarray) -> Dict[str, float]:
    """Compute various correlation and agreement metrics."""
    # Remove NaN values
    valid_mask = ~(np.isnan(human_scores) | np.isnan(llm_scores))
    if np.sum(valid_mask) < 2:
        return {"n_samples": np.sum(valid_mask)}
    
    human_valid = human_scores[valid_mask]
    llm_valid = llm_scores[valid_mask]
    
    metrics = {
        "n_samples": len(human_valid),
        "pearson_r": pearsonr(human_valid, llm_valid)[0],
        "pearson_p": pearsonr(human_valid, llm_valid)[1],
        "kendall_tau": kendalltau(human_valid, llm_valid)[0],
        "kendall_p": kendalltau(human_valid, llm_valid)[1],
        "spearman_r": spearmanr(human_valid, llm_valid)[0],
        "spearman_p": spearmanr(human_valid, llm_valid)[1],
        "mae": mean_absolute_error(human_valid, llm_valid),
        "rmse": np.sqrt(mean_squared_error(human_valid, llm_valid)),
    }
    
    # Cohen's kappa for categorical agreement (round scores to nearest integer)
    human_cat = np.round(human_valid).astype(int)
    llm_cat = np.round(llm_valid).astype(int)
    metrics["cohen_kappa"] = cohen_kappa_score(human_cat, llm_cat)
    
    # Agreement within 1 point
    within_1 = np.abs(human_valid - llm_valid) <= 1.0
    metrics["agreement_within_1"] = np.mean(within_1)
    
    # Exact agreement
    exact_agreement = np.abs(human_valid - llm_valid) < 0.1
    metrics["exact_agreement"] = np.mean(exact_agreement)
    
    return metrics


def analyze_human_llm_correlation(merged_data: Dict[str, pd.DataFrame], 
                                task: str, model: str, output_dir: Path) -> pd.DataFrame:
    """Analyze correlations between human and LLM evaluations."""
    
    results = []
    
    for judge, df in merged_data.items():
        print(f"\nAnalyzing {task} - {model} - {judge}")
        
        for score_col in SCORE_COLS:
            human_col = f"{score_col}_human"
            llm_col = f"{score_col}_{judge}"
            
            if human_col not in df.columns or llm_col not in df.columns:
                print(f"Warning: Missing columns {human_col} or {llm_col}")
                continue
            
            human_scores = df[human_col].values
            llm_scores = df[llm_col].values
            
            metrics = compute_correlation_metrics(human_scores, llm_scores)
            
            result = {
                "task": task,
                "model": model,
                "judge": judge,
                "criterion": score_col.replace("score_", ""),
                **metrics
            }
            results.append(result)
    
    return pd.DataFrame(results)


def create_correlation_heatmap(correlation_df: pd.DataFrame, output_dir: Path, task: str):
    """Create correlation heatmaps for visualization."""
    
    # Prepare data for heatmap
    pivot_data = correlation_df.pivot_table(
        index=["model", "criterion"], 
        columns="judge", 
        values="pearson_r"
    )
    
    if pivot_data.empty:
        print(f"No data for {task} heatmap")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap="RdBu_r", 
        center=0,
        vmin=-1, vmax=1,
        fmt=".3f",
        cbar_kws={"label": "Pearson Correlation"}
    )
    
    plt.title(f"Human-LLM Correlation: {task.title()} Task")
    plt.xlabel("LLM Judge")
    plt.ylabel("Model - Criterion")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{task}_human_llm_correlation_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to {output_file}")


def create_agreement_analysis(merged_data: Dict[str, pd.DataFrame], 
                            task: str, model: str, output_dir: Path):
    """Create detailed agreement analysis plots."""
    
    for judge, df in merged_data.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, score_col in enumerate(SCORE_COLS):
            human_col = f"{score_col}_human"
            llm_col = f"{score_col}_{judge}"
            
            if human_col not in df.columns or llm_col not in df.columns:
                continue
            
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(df[human_col], df[llm_col], alpha=0.6)
            
            # Perfect agreement line
            min_val = min(df[human_col].min(), df[llm_col].min())
            max_val = max(df[human_col].max(), df[llm_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Within 1 point lines
            ax.plot([min_val, max_val], [min_val+1, max_val+1], 'g--', alpha=0.5)
            ax.plot([min_val, max_val], [min_val-1, max_val-1], 'g--', alpha=0.5)
            
            ax.set_xlabel("Human Score")
            ax.set_ylabel(f"{judge} Score")
            ax.set_title(f"{score_col.replace('score_', '').title()}")
            
            # Add correlation info
            corr = df[human_col].corr(df[llm_col])
            ax.text(0.05, 0.95, f"r = {corr:.3f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f"Human vs {judge} Agreement: {task.title()} - {model}")
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / f"{task}_{model}_{judge}_agreement_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved agreement analysis to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze correlations between human and LLM evaluations")
    parser.add_argument("--human_eval_dir", type=str,
                       default="/orange/ufdatastudios/c.okocha/child__speech_analysis/Evaluation/Human_Eval",
                       help="Directory containing human evaluations")
    parser.add_argument("--llm_eval_dir", type=str,
                       default="/orange/ufdatastudios/c.okocha/child__speech_analysis/Evaluation/LLM_Eval",
                       help="Directory containing LLM evaluations")
    parser.add_argument("--output_dir", type=str,
                       default="/orange/ufdatastudios/c.okocha/child__speech_analysis/results/HumanLLMCorrelation",
                       help="Output directory for analysis results")
    parser.add_argument("--tasks", nargs="+", default=["interview", "reading"],
                       help="Tasks to analyze")
    
    args = parser.parse_args()
    
    human_dir = Path(args.human_eval_dir)
    llm_dir = Path(args.llm_eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for task in args.tasks:
        print(f"\n{'='*60}")
        print(f"Processing {task.upper()} task")
        print(f"{'='*60}")
        
        # Load evaluations
        human_evals = load_human_evaluations(human_dir, task)
        llm_evals = load_llm_evaluations(llm_dir, task)
        
        print(f"Human evaluations loaded: {list(human_evals.keys())}")
        print(f"LLM evaluations loaded: {list(llm_evals.keys())}")
        for judge in llm_evals:
            print(f"  {judge}: {list(llm_evals[judge].keys())}")
        
        if not human_evals or not llm_evals:
            print(f"No evaluations found for {task}")
            continue
        
        # Define models for this task
        task_models = INTERVIEW_MODELS if task == "interview" else READING_MODELS
        
        # Analyze each model
        for model in task_models:
            if model not in human_evals:
                print(f"No human evaluations for {model}")
                continue
            
            human_df = human_evals[model]
            
            # Collect LLM evaluations for this model
            model_llm_evals = {}
            for judge in LLM_JUDGES:
                if judge in llm_evals and model in llm_evals[judge]:
                    model_llm_evals[judge] = llm_evals[judge][model]
            
            if not model_llm_evals:
                print(f"No LLM evaluations for {model}")
                continue
            
            # Merge human and LLM evaluations
            merged_data = merge_evaluations(human_df, model_llm_evals)
            
            if not merged_data:
                print(f"No overlapping data for {model}")
                continue
            
            # Analyze correlations
            task_results = analyze_human_llm_correlation(merged_data, task, model, output_dir)
            all_results.append(task_results)
            
            # Create visualizations
            create_agreement_analysis(merged_data, task, model, output_dir)
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save detailed results
        results_csv = output_dir / "human_llm_correlation_detailed.csv"
        combined_df.to_csv(results_csv, index=False)
        print(f"\nSaved detailed results to {results_csv}")
        
        # Create summary statistics
        summary_stats = combined_df.groupby(["task", "judge", "criterion"]).agg({
            "pearson_r": ["mean", "std", "count"],
            "kendall_tau": ["mean", "std"],
            "cohen_kappa": ["mean", "std"],
            "agreement_within_1": ["mean", "std"],
            "mae": ["mean", "std"],
            "rmse": ["mean", "std"]
        }).round(3)
        
        summary_csv = output_dir / "human_llm_correlation_summary.csv"
        summary_stats.to_csv(summary_csv)
        print(f"Saved summary statistics to {summary_csv}")
        
        # Create overall correlation heatmaps
        for task in args.tasks:
            task_data = combined_df[combined_df["task"] == task]
            if not task_data.empty:
                create_correlation_heatmap(task_data, output_dir, task)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY OF HUMAN-LLM CORRELATIONS")
        print(f"{'='*60}")
        
        for task in args.tasks:
            task_data = combined_df[combined_df["task"] == task]
            if task_data.empty:
                continue
                
            print(f"\n{task.upper()} TASK:")
            for judge in LLM_JUDGES:
                judge_data = task_data[task_data["judge"] == judge]
                if judge_data.empty:
                    continue
                    
                avg_pearson = judge_data["pearson_r"].mean()
                avg_kappa = judge_data["cohen_kappa"].mean()
                avg_agreement = judge_data["agreement_within_1"].mean()
                
                print(f"  {judge}:")
                print(f"    Average Pearson r: {avg_pearson:.3f}")
                print(f"    Average Cohen's κ: {avg_kappa:.3f}")
                print(f"    Agreement within ±1: {avg_agreement:.3f}")
        
        # Save JSON summary
        summary_json = {}
        for task in args.tasks:
            task_data = combined_df[combined_df["task"] == task]
            if task_data.empty:
                continue
                
            summary_json[task] = {}
            for judge in LLM_JUDGES:
                judge_data = task_data[task_data["judge"] == judge]
                if judge_data.empty:
                    continue
                    
                summary_json[task][judge] = {
                    "avg_pearson_r": float(judge_data["pearson_r"].mean()),
                    "avg_cohen_kappa": float(judge_data["cohen_kappa"].mean()),
                    "avg_agreement_within_1": float(judge_data["agreement_within_1"].mean()),
                    "avg_mae": float(judge_data["mae"].mean()),
                    "avg_rmse": float(judge_data["rmse"].mean())
                }
        
        json_file = output_dir / "human_llm_correlation_summary.json"
        with open(json_file, "w") as f:
            json.dump(summary_json, f, indent=2)
        print(f"\nSaved JSON summary to {json_file}")
    
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
