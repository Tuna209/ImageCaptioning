#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis script for PaliGemma adapter-based fine-tuning benchmarks.
This script analyzes and visualizes the results of benchmark experiments.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark_config import BENCHMARK_EXPERIMENTS


def load_benchmark_results(results_dir):
    """Load benchmark results from the specified directory."""
    # Load summary file
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        raise ValueError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    # Load individual experiment results
    experiments = []
    for exp in summary["experiments"]:
        exp_name = exp["name"]
        exp_dir = os.path.join(results_dir, exp_name)
        results_path = os.path.join(exp_dir, "results.json")
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            experiments.append(exp_results)
    
    return experiments


def create_results_dataframe(experiments):
    """Create a DataFrame from experiment results."""
    data = []
    
    for exp in experiments:
        # Get experiment details
        exp_name = exp["experiment_name"]
        config = exp["config"]
        best_val_loss = exp["best_val_loss"]
        avg_bleu = exp["avg_bleu"]
        
        # Add to data
        data.append({
            "experiment_name": exp_name,
            "adapter_size": config["adapter_size"],
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
            "prompt_template": config["prompt_template"],
            "subset_size": config["subset_size"],
            "best_val_loss": best_val_loss,
            "avg_bleu": avg_bleu
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def plot_adapter_size_comparison(df, output_dir):
    """Plot comparison of different adapter sizes."""
    # Filter experiments that vary adapter size
    adapter_exps = df[df["experiment_name"].str.contains("adapter_test") | (df["experiment_name"] == "baseline")]
    adapter_exps = adapter_exps.sort_values("adapter_size")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot validation loss
    color = "tab:blue"
    ax1.set_xlabel("Adapter Size")
    ax1.set_ylabel("Validation Loss", color=color)
    ax1.plot(adapter_exps["adapter_size"], adapter_exps["best_val_loss"], "o-", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Create second y-axis for BLEU score
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("BLEU Score", color=color)
    ax2.plot(adapter_exps["adapter_size"], adapter_exps["avg_bleu"], "o-", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Add title and adjust layout
    plt.title("Effect of Adapter Size on Model Performance")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "adapter_size_comparison.png"))
    plt.close()


def plot_learning_rate_comparison(df, output_dir):
    """Plot comparison of different learning rates."""
    # Filter experiments that vary learning rate
    lr_exps = df[df["experiment_name"].str.contains("lr_test") | (df["experiment_name"] == "baseline")]
    lr_exps = lr_exps.sort_values("learning_rate")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot validation loss
    color = "tab:blue"
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Validation Loss", color=color)
    ax1.plot(lr_exps["learning_rate"], lr_exps["best_val_loss"], "o-", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")
    
    # Create second y-axis for BLEU score
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("BLEU Score", color=color)
    ax2.plot(lr_exps["learning_rate"], lr_exps["avg_bleu"], "o-", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Add title and adjust layout
    plt.title("Effect of Learning Rate on Model Performance")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "learning_rate_comparison.png"))
    plt.close()


def plot_prompt_template_comparison(df, output_dir):
    """Plot comparison of different prompt templates."""
    # Filter experiments that vary prompt template
    prompt_exps = df[df["experiment_name"].str.contains("prompt_test") | (df["experiment_name"] == "baseline")]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot validation loss
    x = range(len(prompt_exps))
    width = 0.35
    color = "tab:blue"
    ax1.set_ylabel("Validation Loss", color=color)
    ax1.bar(x, prompt_exps["best_val_loss"], width, color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Create second y-axis for BLEU score
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("BLEU Score", color=color)
    ax2.bar([i + width for i in x], prompt_exps["avg_bleu"], width, color=color, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Set x-axis labels
    plt.xticks([i + width/2 for i in x], prompt_exps["experiment_name"], rotation=45, ha="right")
    
    # Add title and adjust layout
    plt.title("Effect of Prompt Template on Model Performance")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "prompt_template_comparison.png"))
    plt.close()


def plot_data_size_comparison(df, output_dir):
    """Plot comparison of different data sizes."""
    # Filter experiments that vary data size
    data_exps = df[df["experiment_name"].str.contains("data_test") | (df["experiment_name"] == "baseline")]
    data_exps = data_exps.sort_values("subset_size")
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot validation loss
    color = "tab:blue"
    ax1.set_xlabel("Training Data Size")
    ax1.set_ylabel("Validation Loss", color=color)
    ax1.plot(data_exps["subset_size"], data_exps["best_val_loss"], "o-", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Create second y-axis for BLEU score
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("BLEU Score", color=color)
    ax2.plot(data_exps["subset_size"], data_exps["avg_bleu"], "o-", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Add title and adjust layout
    plt.title("Effect of Training Data Size on Model Performance")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "data_size_comparison.png"))
    plt.close()


def plot_overall_comparison(df, output_dir):
    """Plot overall comparison of all experiments."""
    # Sort experiments by BLEU score
    df_sorted = df.sort_values("avg_bleu", ascending=False)
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot validation loss
    x = range(len(df_sorted))
    width = 0.35
    color = "tab:blue"
    ax1.set_ylabel("Validation Loss", color=color)
    ax1.bar(x, df_sorted["best_val_loss"], width, color=color, alpha=0.7)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # Create second y-axis for BLEU score
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("BLEU Score", color=color)
    ax2.bar([i + width for i in x], df_sorted["avg_bleu"], width, color=color, alpha=0.7)
    ax2.tick_params(axis="y", labelcolor=color)
    
    # Set x-axis labels
    plt.xticks([i + width/2 for i in x], df_sorted["experiment_name"], rotation=45, ha="right")
    
    # Add title and adjust layout
    plt.title("Overall Comparison of All Experiments")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "overall_comparison.png"))
    plt.close()


def create_heatmap(df, output_dir):
    """Create a heatmap of experiment results."""
    # Create pivot table for adapter size vs learning rate
    pivot_data = df.pivot_table(
        index="adapter_size",
        columns="learning_rate",
        values="avg_bleu",
        aggfunc="mean"
    )
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".4f")
    
    # Add title and adjust layout
    plt.title("BLEU Score by Adapter Size and Learning Rate")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "heatmap.png"))
    plt.close()


def create_ablation_table(df):
    """Create an ablation study table."""
    # Get baseline performance
    baseline = df[df["experiment_name"] == "baseline"]
    if len(baseline) == 0:
        print("Baseline experiment not found")
        return None
    
    baseline_bleu = baseline["avg_bleu"].values[0]
    
    # Calculate relative performance for each experiment
    df["relative_bleu"] = df["avg_bleu"] / baseline_bleu
    
    # Create ablation table
    ablation_table = df[["experiment_name", "avg_bleu", "relative_bleu"]].sort_values("avg_bleu", ascending=False)
    
    return ablation_table


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="benchmark_results", help="Directory containing benchmark results")
    parser.add_argument("--output_dir", type=str, default="benchmark_analysis", help="Directory to save analysis results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load benchmark results
    print("Loading benchmark results...")
    experiments = load_benchmark_results(args.results_dir)
    
    # Create results DataFrame
    print("Creating results DataFrame...")
    df = create_results_dataframe(experiments)
    
    # Save DataFrame to CSV
    df.to_csv(os.path.join(args.output_dir, "benchmark_results.csv"), index=False)
    
    # Create plots
    print("Creating plots...")
    plot_adapter_size_comparison(df, args.output_dir)
    plot_learning_rate_comparison(df, args.output_dir)
    plot_prompt_template_comparison(df, args.output_dir)
    plot_data_size_comparison(df, args.output_dir)
    plot_overall_comparison(df, args.output_dir)
    
    # Create heatmap
    print("Creating heatmap...")
    create_heatmap(df, args.output_dir)
    
    # Create ablation table
    print("Creating ablation table...")
    ablation_table = create_ablation_table(df)
    if ablation_table is not None:
        ablation_table.to_csv(os.path.join(args.output_dir, "ablation_study.csv"), index=False)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
