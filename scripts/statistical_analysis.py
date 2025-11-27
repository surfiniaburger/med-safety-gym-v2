"""
Statistical Analysis Utilities for Benchmark Evaluation

Provides functions for calculating confidence intervals, significance tests,
and summary statistics for LLM safety benchmarks.
"""
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

def calculate_confidence_intervals(
    data: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        data: List of metric values
        confidence_level: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (mean, lower_bound, upper_bound)

def perform_significance_test(
    data1: List[float],
    data2: List[float],
    test_type: str = "mannwhitneyu"
) -> Dict[str, float]:
    """
    Perform statistical significance test between two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        test_type: Type of test ("mannwhitneyu" or "ttest")
        
    Returns:
        Dict with 'statistic' and 'p_value'
    """
    if len(data1) == 0 or len(data2) == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    
    if test_type == "mannwhitneyu":
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    elif test_type == "ttest":
        statistic, p_value = stats.ttest_ind(data1, data2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        "statistic": float(statistic),
        "p_value": float(p_value)
    }

def calculate_effect_size(data1: List[float], data2: List[float]) -> float:
    """
    Calculate Cohen's d effect size between two datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        Cohen's d value
    """
    if len(data1) == 0 or len(data2) == 0:
        return 0.0
    
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(data1), len(data2)
    
    if n1 + n2 < 3:
        return 0.0
        
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohen_d = (mean1 - mean2) / pooled_std
    return float(cohen_d)

def generate_summary_statistics(data: List[float]) -> Dict[str, float]:
    """
    Generate comprehensive summary statistics for a dataset.
    
    Args:
        data: List of values
        
    Returns:
        Dict with mean, median, std, min, max, q1, q3, iqr
    """
    if len(data) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "count": 0
        }
    
    data_array = np.array(data)
    
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    
    # Handle single element case for std dev
    if len(data) < 2:
        std_val = 0.0
    else:
        std_val = float(np.std(data_array, ddof=1))
    
    return {
        "mean": float(np.mean(data_array)),
        "median": float(np.median(data_array)),
        "std": std_val,
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "count": len(data)
    }

def calculate_metric_confidence_intervals(
    metrics_dict: Dict[str, List[float]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Calculate confidence intervals for multiple metrics.
    
    Args:
        metrics_dict: Dict mapping metric names to lists of values
        confidence_level: Confidence level
        
    Returns:
        Dict mapping metric names to {mean, lower, upper}
    """
    results = {}
    
    for metric_name, values in metrics_dict.items():
        mean, lower, upper = calculate_confidence_intervals(values, confidence_level)
        results[metric_name] = {
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "margin": upper - mean
        }
    
    return results
