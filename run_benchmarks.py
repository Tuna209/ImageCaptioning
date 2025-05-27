#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarking script for PaliGemma adapter fine-tuning.

This script runs multiple experiments with different hyperparameters to find the optimal
configuration for adapter-based fine-tuning of the PaliGemma model on the RISC dataset.
"""

import os
import argparse
import json
import subprocess
import time
import wandb
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

def run_experiment(experiment, use_wandb, data_dir, subset_size=200, val_subset_size=50):
    """Run a single experiment.

    Args:
        experiment (dict): Experiment configuration.
        use_wandb (bool): Whether to use wandb for logging.
        data_dir (str): Directory containing the dataset.
        subset_size (int): Number of examples to use for training.
        val_subset_size (int): Number of examples to use for validation.

    Returns:
        str: Output directory.
    """
    # Create command
    cmd = [
        "python", "adapter_code/train.py",
        "--token", os.getenv("HF_TOKEN"),
        "--adapter_size", str(experiment["adapter_size"]),
        "--learning_rate", str(experiment["learning_rate"]),
        "--batch_size", str(experiment["batch_size"]),
        "--gradient_accumulation_steps", str(experiment["gradient_accumulation_steps"]),
        "--num_epochs", str(experiment["num_epochs"]),
        "--subset_size", str(subset_size),
        "--val_subset_size", str(val_subset_size),
        "--prompt_template", "Describe the remote sensing image in detail."
    ]

    # Add additional parameters if present in the experiment
    if "warmup_ratio" in experiment:
        cmd.extend(["--warmup_ratio", str(experiment["warmup_ratio"])])

    if "weight_decay" in experiment:
        cmd.extend(["--weight_decay", str(experiment["weight_decay"])])

    if "max_grad_norm" in experiment:
        cmd.extend(["--max_grad_norm", str(experiment["max_grad_norm"])])

    # Add optional arguments
    if use_wandb:
        cmd.append("--use_wandb")

    if data_dir:
        cmd.extend(["--data_dir", data_dir])

    # Set output directory
    experiment_name = experiment.get("name", f"adapter-{experiment['adapter_size']}-lr{experiment['learning_rate']:.0e}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"adapter_code/outputs/benchmark/{experiment_name}-{timestamp}"
    cmd.extend(["--output_dir", output_dir])

    # Run command
    print(f"Running experiment: {' '.join(cmd)}")

    # Run the process with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )

    # Capture output in real-time
    stdout_lines = []
    stderr_lines = []

    # Function to read from a pipe and print in real-time
    def read_pipe(pipe, lines_list):
        for line in iter(pipe.readline, ''):
            print(line, end='')  # Print in real-time
            lines_list.append(line)

    # Create threads to read stdout and stderr
    import threading
    stdout_thread = threading.Thread(target=read_pipe, args=(process.stdout, stdout_lines))
    stderr_thread = threading.Thread(target=read_pipe, args=(process.stderr, stderr_lines))

    # Start threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    return_code = process.wait()

    # Wait for threads to complete
    stdout_thread.join()
    stderr_thread.join()

    # Combine output
    stdout = ''.join(stdout_lines)
    stderr = ''.join(stderr_lines)

    # Check if the process was successful
    if return_code != 0:
        print(f"Error running experiment (return code {return_code}): {stderr}")

    return output_dir, stdout, stderr

def extract_metrics(stdout):
    """Extract metrics from the stdout of the training process.

    Args:
        stdout (str): Stdout of the training process.

    Returns:
        dict: Metrics.
    """
    metrics = {
        "train_loss": [],
        "train_perplexity": [],
        "val_loss": [],
        "val_perplexity": []
    }

    for line in stdout.split("\n"):
        if "Epoch" in line and "Train Loss" in line:
            try:
                # Extract metrics from the line
                parts = line.split(":")
                epoch_part = parts[0].strip()
                metrics_part = parts[1].strip()

                # Extract epoch number (for debugging if needed)
                _ = int(epoch_part.split()[1].split("/")[0])

                # Extract train metrics
                train_metrics = metrics_part.split(";")[0].strip()
                train_loss = float(train_metrics.split("=")[1].split(",")[0])
                train_ppl = float(train_metrics.split("=")[2])

                # Extract val metrics
                val_metrics = metrics_part.split(";")[1].strip()
                val_loss = float(val_metrics.split("=")[1].split(",")[0])
                val_ppl = float(val_metrics.split("=")[2])

                # Add to metrics
                metrics["train_loss"].append(train_loss)
                metrics["train_perplexity"].append(train_ppl)
                metrics["val_loss"].append(val_loss)
                metrics["val_perplexity"].append(val_ppl)
            except Exception as e:
                print(f"Error extracting metrics: {e}")

    # Calculate final metrics (average of last epoch)
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = values[-1]

    return final_metrics

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark_config.json", help="Path to benchmark config file")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for tracking")
    parser.add_argument("--data_dir", type=str, default="processed_dataset", help="Data directory")
    parser.add_argument("--subset_size", type=int, default=200, help="Number of examples to use for training")
    parser.add_argument("--val_subset_size", type=int, default=50, help="Number of examples to use for validation")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ImageCaptioning"),
            entity=os.getenv("WANDB_ENTITY", "tuna-ozturk1283"),
            name=f"benchmarking-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"experiments": config["experiments"]}
        )

    # Create output directory
    os.makedirs("adapter_code/outputs/benchmark", exist_ok=True)

    # Run experiments
    results = []
    for i, experiment in enumerate(config["experiments"]):
        print(f"\nRunning experiment {i+1}/{len(config['experiments'])}")
        print(f"Configuration: {experiment}")

        # Run experiment
        start_time = time.time()
        output_dir, stdout, _ = run_experiment(
            experiment,
            args.use_wandb,
            args.data_dir,
            args.subset_size,
            args.val_subset_size
        )
        end_time = time.time()

        # Extract metrics
        metrics = extract_metrics(stdout)

        # Record results
        experiment_result = {
            "experiment_id": i,
            "adapter_size": experiment["adapter_size"],
            "learning_rate": experiment["learning_rate"],
            "batch_size": experiment["batch_size"],
            "gradient_accumulation_steps": experiment["gradient_accumulation_steps"],
            "num_epochs": experiment["num_epochs"],
            "output_dir": output_dir,
            "runtime_seconds": end_time - start_time,
            **metrics
        }
        results.append(experiment_result)

        # Log to wandb
        if args.use_wandb:
            wandb.log(experiment_result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"adapter_code/outputs/benchmark/results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)

    # Print results
    print("\nBenchmarking results:")
    print(results_df)

    # Find best configuration
    if "val_loss" in results_df.columns:
        best_idx = results_df["val_loss"].idxmin()
        best_config = results_df.iloc[best_idx]
        print("\nBest configuration:")
        print(f"  Adapter size: {best_config['adapter_size']}")
        print(f"  Learning rate: {best_config['learning_rate']}")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Gradient accumulation steps: {best_config['gradient_accumulation_steps']}")
        print(f"  Number of epochs: {best_config['num_epochs']}")
        print(f"  Validation loss: {best_config['val_loss']:.4f}")
        print(f"  Validation perplexity: {best_config['val_perplexity']:.2f}")

    print(f"\nResults saved to {results_path}")
    print("\nBenchmarking complete!")

if __name__ == "__main__":
    main()
