from __future__ import annotations

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt


def load_evaluation_results(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all evaluation JSON files and return as DataFrames by judge model."""
    results = {}
    
    # Look for files in subdirectories: Judge/ModelJudge/*.json or Judge/*.json
    for json_file in eval_dir.rglob("*.json"):
        if not json_file.name.endswith(("_eval.json", "_evaluations.json")):
            continue
            
        # Extract judge name from directory structure or filename
        judge_name = None
        
        # Try to get judge from parent directory names
        parent_dirs = [p.name.lower() for p in json_file.parents]
        for potential_judge in ["llama", "mistral", "qwen", "qwen2"]:
            if potential_judge in parent_dirs:
                judge_name = potential_judge
                break
        
        # Fallback: extract from filename
        if not judge_name:
            stem = json_file.stem
            if "_eval" in stem:
                parts = stem.split("_")
                for part in parts:
                    if part.lower() in ["llama", "mistral", "qwen", "qwen2"]:
                        judge_name = part.lower()
                        break
        
        if not judge_name:
            print(f"Could not determine judge for {json_file}")
            continue
            
        # Extract model name from filename
        stem = json_file.stem
        if stem.endswith("_evaluations"):
            model_name = stem[:-12]  # remove _evaluations
        elif stem.endswith("_eval"):
            # For pattern: {model}_{judge}_eval.json
            parts = stem.split("_")
            if len(parts) >= 3 and parts[-1] == "eval":
                model_name = "_".join(parts[:-2])  # everything except judge_eval
            else:
                model_name = "_".join(parts[:-1])  # everything except eval
        else:
            model_name = stem
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            if not df.empty:
                df['judge_model'] = judge_name
                df['evaluated_model'] = model_name
                
                if judge_name not in results:
                    results[judge_name] = []
                results[judge_name].append(df)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    # Concatenate all DataFrames for each judge
    final_results = {}
    for judge, dfs in results.items():
        if dfs:
            final_results[judge] = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {judge}: {len(final_results[judge])} evaluations")
    
    return final_results


def compute_pairwise_correlations(df1: pd.DataFrame, df2: pd.DataFrame, 
                                judge1: str, judge2: str) -> Dict[str, Dict[str, float]]:
    """Compute correlations between two judges across all scoring dimensions."""
    score_columns = [
        "score_overall", "score_fluency", "score_faithfulness", 
        "score_coverage", "score_purity", "score_usefulness"
    ]
    
    # Merge on audio_id and evaluated_model to align scores
    merged = pd.merge(df1, df2, on=['audio_id', 'evaluated_model'], 
                     suffixes=(f'_{judge1}', f'_{judge2}'))
    
    correlations = {}
    
    for score_col in score_columns:
        col1 = f"{score_col}_{judge1}"
        col2 = f"{score_col}_{judge2}"
        
        if col1 in merged.columns and col2 in merged.columns:
            # Remove rows with missing scores
            valid_data = merged[[col1, col2]].dropna()
            
            if len(valid_data) > 1:
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(valid_data[col1], valid_data[col2])
                
                # Kendall's tau
                kendall_tau, kendall_p = kendalltau(valid_data[col1], valid_data[col2])
                
                correlations[score_col] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'kendall_tau': kendall_tau,
                    'kendall_p': kendall_p,
                    'n_samples': len(valid_data)
                }
    
    return correlations


def compute_agreement_metrics(df1: pd.DataFrame, df2: pd.DataFrame, 
                            judge1: str, judge2: str) -> Dict[str, float]:
    """Compute agreement metrics between two judges."""
    score_columns = [
        "score_overall", "score_fluency", "score_faithfulness", 
        "score_coverage", "score_purity", "score_usefulness"
    ]
    
    # Merge dataframes
    merged = pd.merge(df1, df2, on=['audio_id', 'evaluated_model'], 
                     suffixes=(f'_{judge1}', f'_{judge2}'))
    
    agreement_metrics = {}
    
    for score_col in score_columns:
        col1 = f"{score_col}_{judge1}"
        col2 = f"{score_col}_{judge2}"
        
        if col1 in merged.columns and col2 in merged.columns:
            valid_data = merged[[col1, col2]].dropna()
            
            if len(valid_data) > 1:
                # Exact agreement rate
                exact_agreement = (valid_data[col1] == valid_data[col2]).mean()
                
                # Within-1-point agreement rate
                within_1_agreement = (abs(valid_data[col1] - valid_data[col2]) <= 1).mean()
                
                # Cohen's Kappa (treating as ordinal categories)
                try:
                    kappa = cohen_kappa_score(valid_data[col1], valid_data[col2])
                except:
                    kappa = np.nan
                
                agreement_metrics[score_col] = {
                    'exact_agreement': exact_agreement,
                    'within_1_agreement': within_1_agreement,
                    'cohen_kappa': kappa,
                    'n_samples': len(valid_data)
                }
    
    return agreement_metrics


def create_correlation_matrix(results: Dict[str, pd.DataFrame], 
                            output_dir: Path, task_name: str):
    """Create correlation heatmaps for each scoring dimension."""
    score_columns = [
        "score_overall", "score_fluency", "score_faithfulness", 
        "score_coverage", "score_purity", "score_usefulness"
    ]
    
    judges = list(results.keys())
    
    for score_col in score_columns:
        # Skip purity for reading tasks
        if task_name == "reading" and score_col == "score_purity":
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pearson correlation matrix
        pearson_matrix = np.full((len(judges), len(judges)), np.nan)
        kendall_matrix = np.full((len(judges), len(judges)), np.nan)
        
        for i, judge1 in enumerate(judges):
            for j, judge2 in enumerate(judges):
                if i == j:
                    pearson_matrix[i, j] = 1.0
                    kendall_matrix[i, j] = 1.0
                elif i < j:  # Only compute upper triangle
                    corrs = compute_pairwise_correlations(
                        results[judge1], results[judge2], judge1, judge2
                    )
                    if score_col in corrs:
                        pearson_matrix[i, j] = corrs[score_col]['pearson_r']
                        pearson_matrix[j, i] = corrs[score_col]['pearson_r']
                        kendall_matrix[i, j] = corrs[score_col]['kendall_tau']
                        kendall_matrix[j, i] = corrs[score_col]['kendall_tau']
        
        # Plot Pearson correlations
        sns.heatmap(pearson_matrix, annot=True, fmt='.3f', 
                   xticklabels=judges, yticklabels=judges,
                   cmap='RdYlBu_r', center=0, ax=ax1)
        ax1.set_title(f'Pearson Correlation - {score_col}')
        
        # Plot Kendall's tau
        sns.heatmap(kendall_matrix, annot=True, fmt='.3f',
                   xticklabels=judges, yticklabels=judges,
                   cmap='RdYlBu_r', center=0, ax=ax2)
        ax2.set_title(f'Kendall τ - {score_col}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{task_name}_{score_col}_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_inter_judge_consistency(eval_dir: Path, task_name: str, output_dir: Path):
    """Main analysis function for inter-judge consistency."""
    print(f"\n=== Analyzing Inter-Judge Consistency for {task_name.title()} ===")
    print(f"Looking in: {eval_dir}")
    
    # Load all evaluation results
    results = load_evaluation_results(eval_dir)
    judges = list(results.keys())
    
    if len(judges) < 2:
        print(f"Need at least 2 judges, found {len(judges)}: {judges}")
        if len(judges) == 1:
            print(f"Available judge: {judges[0]} with {len(results[judges[0]])} evaluations")
        return
    
    print(f"Found {len(judges)} judges: {judges}")
    for judge, df in results.items():
        print(f"  {judge}: {len(df)} evaluations")
    
    # Compute all pairwise correlations and agreements
    all_correlations = {}
    all_agreements = {}
    
    for i, judge1 in enumerate(judges):
        for j, judge2 in enumerate(judges):
            if i < j:  # Only compute upper triangle
                pair_key = f"{judge1}_vs_{judge2}"
                print(f"\nComputing correlations: {pair_key}")
                
                correlations = compute_pairwise_correlations(
                    results[judge1], results[judge2], judge1, judge2
                )
                agreements = compute_agreement_metrics(
                    results[judge1], results[judge2], judge1, judge2
                )
                
                all_correlations[pair_key] = correlations
                all_agreements[pair_key] = agreements
    
    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{task_name}_correlations.json", 'w') as f:
        json.dump(all_correlations, f, indent=2, default=str)
    
    with open(output_dir / f"{task_name}_agreements.json", 'w') as f:
        json.dump(all_agreements, f, indent=2, default=str)
    
    # Create summary report
    summary_report = []
    
    for pair_key, correlations in all_correlations.items():
        for score_dim, metrics in correlations.items():
            summary_report.append({
                'task': task_name,
                'judge_pair': pair_key,
                'score_dimension': score_dim,
                'pearson_r': metrics['pearson_r'],
                'pearson_p': metrics['pearson_p'],
                'kendall_tau': metrics['kendall_tau'],
                'kendall_p': metrics['kendall_p'],
                'n_samples': metrics['n_samples'],
                'exact_agreement': all_agreements[pair_key][score_dim]['exact_agreement'],
                'within_1_agreement': all_agreements[pair_key][score_dim]['within_1_agreement'],
                'cohen_kappa': all_agreements[pair_key][score_dim]['cohen_kappa']
            })
    
    summary_df = pd.DataFrame(summary_report)
    summary_df.to_csv(output_dir / f"{task_name}_inter_judge_summary.csv", index=False)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics for {task_name.title()} ===")
    print("Average Pearson correlations by dimension:")
    for dim in summary_df['score_dimension'].unique():
        dim_data = summary_df[summary_df['score_dimension'] == dim]
        avg_pearson = dim_data['pearson_r'].mean()
        avg_kendall = dim_data['kendall_tau'].mean()
        avg_agreement = dim_data['exact_agreement'].mean()
        print(f"  {dim:20s}: r={avg_pearson:.3f}, τ={avg_kendall:.3f}, exact_agr={avg_agreement:.3f}")
    
    # Create correlation heatmaps
    create_correlation_matrix(results, output_dir, task_name)
    
    print(f"\nResults saved to: {output_dir}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Analyze inter-judge consistency across LLM judges")
    parser.add_argument("--eval_base_dir", type=str, 
                       default="/orange/ufdatastudios/c.okocha/child__speech_analysis/Evaluation/LLM_Eval",
                       help="Base directory containing evaluation results")
    parser.add_argument("--output_dir", type=str,
                       default="/orange/ufdatastudios/c.okocha/child__speech_analysis/results/InterJudgeConsistency",
                       help="Output directory for consistency analysis")
    parser.add_argument("--tasks", nargs="+", default=["reading", "interview"],
                       help="Tasks to analyze (subdirectories in eval_base_dir)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.eval_base_dir)
    output_dir = Path(args.output_dir)
    
    all_summaries = []
    
    for task in args.tasks:
        task_eval_dir = base_dir / task
        if not task_eval_dir.exists():
            print(f"Warning: {task_eval_dir} does not exist, skipping {task}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {task.upper()} evaluations from {task_eval_dir}")
        
        task_output_dir = output_dir / task
        summary_df = analyze_inter_judge_consistency(task_eval_dir, task, task_output_dir)
        
        if summary_df is not None:
            summary_df['task'] = task
            all_summaries.append(summary_df)
    
    # Combine all tasks for overall analysis
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        combined_df.to_csv(output_dir / "combined_inter_judge_summary.csv", index=False)
        
        print(f"\n{'='*60}")
        print("OVERALL INTER-JUDGE CONSISTENCY SUMMARY")
        print(f"{'='*60}")
        
        # Overall averages by dimension across all tasks
        print("\nAverage correlations across all tasks and judge pairs:")
        for dim in combined_df['score_dimension'].unique():
            dim_data = combined_df[combined_df['score_dimension'] == dim]
            avg_pearson = dim_data['pearson_r'].mean()
            avg_kendall = dim_data['kendall_tau'].mean()
            avg_exact = dim_data['exact_agreement'].mean()
            avg_within1 = dim_data['within_1_agreement'].mean()
            
            print(f"  {dim:20s}: r={avg_pearson:.3f}, τ={avg_kendall:.3f}, "
                  f"exact={avg_exact:.3f}, ±1={avg_within1:.3f}")
        
        # Task-specific breakdown
        print("\nBy task:")
        for task in combined_df['task'].unique():
            task_data = combined_df[combined_df['task'] == task]
            avg_pearson = task_data['pearson_r'].mean()
            avg_kendall = task_data['kendall_tau'].mean()
            print(f"  {task:12s}: r={avg_pearson:.3f}, τ={avg_kendall:.3f}")
        
        print(f"\nDetailed results saved to: {output_dir}")
        print(f"Combined summary: {output_dir}/combined_inter_judge_summary.csv")


if __name__ == "__main__":
    main()
