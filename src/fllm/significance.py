import json
from typing import Any
import numpy as np
from scipy import stats
from pathlib import Path


def load_eval_results(file_path):
    """Load evaluation results from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def extract_scores(results, metric_name):
    """
    Extract scores for a specific metric from results.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of the metric (e.g., "Overall Rating", "Active Listening")
    
    Returns:
        List of scores for that metric
    """
    scores = []
    for result in results:
        if metric_name in result:
            scores.append(result[metric_name])
    return scores


def get_all_metrics(results):
    """Get all metric names from results (excluding question and response)."""
    if not results:
        return []
    
    metrics = set()
    for result in results:
        for key in result.keys():
            if key not in ["question", "response"]:
                metrics.add(key)
    return sorted(list(metrics))


def compare_evaluations(file1_path, file2_path, alpha=0.05):
    """
    Compare two evaluation result files and test for statistical significance.
    
    Args:
        file1_path: Path to first evaluation results file
        file2_path: Path to second evaluation results file
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with comparison results
    """
    # Load both files
    data1 = load_eval_results(file1_path)
    data2 = load_eval_results(file2_path)
    
    results1 = data1["results"]
    results2 = data2["results"]
    
    # Get all metrics
    metrics1 = get_all_metrics(results1)
    metrics2 = get_all_metrics(results2)
    
    # Find common metrics
    common_metrics = sorted(list(set(metrics1) & set(metrics2)))
    
    if not common_metrics:
        raise ValueError("No common metrics found between the two files")
    
    comparison_results = {
        "file1": str(file1_path),
        "file2": str(file2_path),
        "file1_mean_scores": data1.get("mean_scores", {}),
        "file2_mean_scores": data2.get("mean_scores", {}),
        "metrics": {}
    }
    
    # Create a mapping of questions to results for both files
    results1_by_question = {r["question"]: r for r in results1}
    results2_by_question = {r["question"]: r for r in results2}
    
    # Find common questions (questions that appear in both files)
    common_questions = set(results1_by_question.keys()) & set(results2_by_question.keys())
    
    if not common_questions:
        raise ValueError("No common questions found between the two files. Cannot perform paired t-test.")
    
    print(f"Found {len(common_questions)} common questions for paired comparison.")
    print(f"File 1 has {len(results1)} total examples, File 2 has {len(results2)} total examples.")
    if len(common_questions) < len(results1) or len(common_questions) < len(results2):
        print(f"Warning: {len(results1) - len(common_questions)} questions from File 1 and {len(results2) - len(common_questions)} questions from File 2 will be excluded from paired comparison.")
    
    # Compare each metric using paired t-test on matched questions
    for metric in common_metrics:
        # Extract paired scores (same questions from both files, only if both have the metric)
        paired_scores1 = []
        paired_scores2 = []
        
        for question in common_questions:
            result1 = results1_by_question[question]
            result2 = results2_by_question[question]
            
            # Only include if both results have this metric
            if metric in result1 and metric in result2:
                paired_scores1.append(result1[metric])
                paired_scores2.append(result2[metric])
        
        if len(paired_scores1) < 2:
            print(f"Warning: Only {len(paired_scores1)} paired scores found for {metric}. Skipping.")
            continue
        
        if len(paired_scores1) != len(paired_scores2):
            raise ValueError(f"Paired scores mismatch: {len(paired_scores1)} vs {len(paired_scores2)} for {metric}")
        
        # Use paired t-test since we're comparing the same questions
        stat, p_value = stats.ttest_rel(paired_scores1, paired_scores2)
        test_type = "paired_t_test"
        
        mean1 = np.mean(paired_scores1)
        mean2 = np.mean(paired_scores2)
        diff = mean2 - mean1
        is_significant = bool(p_value < alpha)  # Convert to Python bool
        
        comparison_results["metrics"][metric] = {
            "file1_mean": float(mean1),
            "file2_mean": float(mean2),
            "difference": float(diff),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "test_type": test_type,
            "statistic": float(stat),
            "n_pairs": len(paired_scores1)  # Number of paired comparisons
        }
    
    return comparison_results


def print_comparison(comparison_results):
    """Print comparison results in a readable format."""
    print("=" * 80)
    print("EVALUATION COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nFile 1: {comparison_results['file1']}")
    print(f"File 2: {comparison_results['file2']}")
    print("\n" + "-" * 80)
    
    print("\nMEAN SCORES COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<35} {'File 1 Mean':<15} {'File 2 Mean':<15} {'Difference':<15} {'Significant':<12}")
    print("-" * 80)
    
    for metric, results in comparison_results["metrics"].items():
        sig_marker = "***" if results["is_significant"] else ""
        print(f"{metric:<35} {results['file1_mean']:<15.3f} {results['file2_mean']:<15.3f} "
              f"{results['difference']:<15.3f} {sig_marker:<12}")
    
    print("\n" + "-" * 80)
    print("\nDETAILED STATISTICAL TEST RESULTS:")
    print("-" * 80)
    
    for metric, results in comparison_results["metrics"].items():
        print(f"\n{metric}:")
        print(f"  File 1 Mean: {results['file1_mean']:.3f}")
        print(f"  File 2 Mean: {results['file2_mean']:.3f}")
        print(f"  Difference: {results['difference']:.3f}")
        print(f"  Test Type: {results['test_type']}")
        print(f"  t-statistic: {results['statistic']:.4f}")
        print(f"  p-value: {results['p_value']:.6f}")
        print(f"  Significant (α=0.05): {'Yes' if results['is_significant'] else 'No'}")
    
    print("\n" + "=" * 80)
    print("Note: *** indicates statistically significant difference (p < 0.05)")
    print("=" * 80)


def save_comparison(comparison_results, output_path):
    """Save comparison results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nComparison results saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Compare two evaluation files
    eval_results_dir = Path("eval_results5")
    
    file1 = eval_results_dir / "CL3.json"
    file2 = eval_results_dir / "FL.json"
    
    if file1.exists() and file2.exists():
        comparison = compare_evaluations(file1, file2)
        print_comparison(comparison)
        
        # Save results
        output_file = eval_results_dir / "CL3VsFL.json"
        save_comparison(comparison, output_file)
    else:
        print(f"Files not found. Please check:")
        print(f"  File 1: {file1}")
        print(f"  File 2: {file2}")