"""
Visualization Utilities for Benchmark Evaluation

Provides functions to create publication-quality visualizations for
LLM safety benchmark results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def create_radar_chart(
    metrics: Dict[str, float],
    ci_data: Optional[Dict[str, Dict[str, float]]] = None,
    title: str = "Safety Metrics Profile",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a radar chart for multi-dimensional safety metrics.
    
    Args:
        metrics: Dict mapping metric names to values (0-1 scale)
        ci_data: Optional confidence interval data
        title: Chart title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    values += values[:1]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2, label='Model Performance', color='#2E86AB')
    ax.fill(angles, values, alpha=0.25, color='#2E86AB')
    
    # Fix axis to go in the right order
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=8)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add title
    plt.title(title, size=14, weight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def create_metrics_bar_chart(
    metrics: Dict[str, float],
    ci_data: Dict[str, Dict[str, float]],
    title: str = "Safety Metrics Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart with confidence intervals for metrics.
    
    Args:
        metrics: Dict mapping metric names to mean values
        ci_data: Dict with confidence interval data {metric: {mean, lower, upper}}
        title: Chart title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(metrics.keys())
    means = [metrics[label] for label in labels]
    errors = [[ci_data[label]['mean'] - ci_data[label]['lower'] for label in labels],
              [ci_data[label]['upper'] - ci_data[label]['mean'] for label in labels]]
    
    x_pos = np.arange(len(labels))
    
    # Create bars
    bars = ax.bar(x_pos, means, yerr=errors, capsize=5, 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Customize
    ax.set_xlabel('Metrics', fontsize=12, weight='bold')
    ax.set_ylabel('Rate', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1%}',
                ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def create_reward_distribution(
    rewards: List[float],
    title: str = "Reward Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create histogram and box plot for reward distribution.
    
    Args:
        rewards: List of reward values
        title: Chart title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(rewards, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax1.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax1.set_xlabel('Reward', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('Histogram', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot(rewards, vert=True, patch_artist=True,
                      boxprops=dict(facecolor='#C73E1D', alpha=0.7),
                      medianprops=dict(color='black', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))
    ax2.set_ylabel('Reward', fontsize=11, weight='bold')
    ax2.set_title('Box Plot', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(rewards):.2f}\nMedian: {np.median(rewards):.2f}\nStd: {np.std(rewards):.2f}"
    ax2.text(1.15, np.mean(rewards), stats_text, fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def create_comparison_plot(
    models_data: Dict[str, Dict[str, float]],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grouped bar chart comparing multiple models.
    
    Args:
        models_data: Dict mapping model names to metric dicts
        title: Chart title
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get metric names from first model
    metrics = list(next(iter(models_data.values())).keys())
    models = list(models_data.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    colors = sns.color_palette("husl", len(models))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [models_data[model][metric] for metric in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model, color=color, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, weight='bold')
    ax.set_ylabel('Rate', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def save_all_visualizations(
    results: Dict,
    output_dir: str,
    model_name: str = "model"
) -> List[str]:
    """
    Generate and save all visualizations for a benchmark run.
    
    Args:
        results: Benchmark results dict
        output_dir: Directory to save figures
        model_name: Name of the model being evaluated
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # 1. Radar chart
    metrics_for_radar = {
        "Refusal Rate": results.get('refusal_rate', 0),
        "Safe Response": results.get('safe_response_rate', 0),
        "Hallucination\n(inverted)": 1 - results.get('medical_hallucination_rate', 0),
        "Consistency": results.get('reasoning_consistency_rate', 0),
    }
    radar_path = str(output_path / f"{model_name}_radar.png")
    create_radar_chart(metrics_for_radar, title=f"{model_name} Safety Profile", save_path=radar_path)
    saved_files.append(radar_path)
    plt.close()
    
    # 2. Bar chart (if CI data available)
    if 'ci_data' in results:
        bar_path = str(output_path / f"{model_name}_metrics_bars.png")
        metrics_dict = {
            "Refusal Rate": results['refusal_rate'],
            "Safe Response": results['safe_response_rate'],
            "Hallucination": results['medical_hallucination_rate'],
            "Consistency": results['reasoning_consistency_rate'],
        }
        create_metrics_bar_chart(metrics_dict, results['ci_data'], 
                                  title=f"{model_name} Metrics with 95% CI", 
                                  save_path=bar_path)
        saved_files.append(bar_path)
        plt.close()
    
    # 3. Reward distribution
    if 'rewards' in results:
        dist_path = str(output_path / f"{model_name}_reward_distribution.png")
        create_reward_distribution(results['rewards'], 
                                    title=f"{model_name} Reward Distribution",
                                    save_path=dist_path)
        saved_files.append(dist_path)
        plt.close()
    
    return saved_files
