#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weights & Biases sweep for hyperparameter tuning of PaliGemma adapter fine-tuning.
"""

import os
import sys
import argparse
import subprocess
import wandb
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define sweep configuration
sweep_config = {
    "method": "bayes",  # Bayesian optimization
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "adapter_size": {
            "values": [4, 8, 16, 32]
        },
        "learning_rate": {
            "values": [5e-6, 1e-5, 5e-5]
        },
        "batch_size": {
            "values": [4]
        },
        "gradient_accumulation_steps": {
            "values": [8]
        },
        "warmup_ratio": {
            "values": [0.1, 0.2]
        },
        "weight_decay": {
            "values": [0.01, 0.05]
        },
        "max_grad_norm": {
            "values": [1.0]
        },
        "num_epochs": {
            "values": [3]
        }
    },
    "name": "paligemma-adapter-sweep"
}

# Define training function
def train():
    """Training function to be called by wandb sweep agent."""
    # Initialize wandb run
    wandb.init()

    # Get hyperparameters from wandb
    config = wandb.config

    # Create a unique run name following the required format
    run_name = f"paligemma-adapter-{config.adapter_size}-lr{config.learning_rate:.0e}"

    # Add optimization details
    optimization_details = []

    if hasattr(config, "warmup_ratio") and config.warmup_ratio != 0.1:
        optimization_details.append(f"warmup{config.warmup_ratio}")

    if hasattr(config, "weight_decay") and config.weight_decay != 0.01:
        optimization_details.append(f"wd{config.weight_decay}")

    if hasattr(config, "gradient_accumulation_steps") and config.gradient_accumulation_steps != 8:
        optimization_details.append(f"grad{config.gradient_accumulation_steps}")

    # Add "optimized" suffix if there are any optimization details
    if optimization_details:
        run_name += "-" + "-".join(optimization_details) + "-optimized"
    else:
        run_name += "-optimized"

    # Create output directory
    output_dir = f"adapter_code/outputs/sweep/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Print configuration
    print(f"Starting training with configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")

    # Construct command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "adapter_code/train.py",
        "--token", os.getenv("HF_TOKEN"),
        "--adapter_size", str(config.adapter_size),
        "--learning_rate", str(config.learning_rate),
        "--batch_size", str(config.batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--warmup_ratio", str(config.warmup_ratio),
        "--weight_decay", str(config.weight_decay),
        "--max_grad_norm", str(config.max_grad_norm),
        "--num_epochs", str(config.num_epochs),
        "--subset_size", "1000",
        "--val_subset_size", "100",
        "--data_dir", "processed_dataset",
        "--output_dir", output_dir,
        "--use_wandb",
        "--prompt_template", "Describe the remote sensing image in detail.",
        "--verbose"
    ]

    # Print command
    print(f"Running command: {' '.join(cmd)}")

    # Run command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Capture output in real-time
    stdout_lines = []
    stderr_lines = []

    # Function to read from a pipe and print in real-time
    def read_pipe(pipe, lines_list, is_stderr=False):
        for line in iter(pipe.readline, ''):
            print(line, end='')  # Print in real-time
            lines_list.append(line)

            # Extract metrics from output and log to wandb
            if not is_stderr and "Train Loss=" in line and "Val Loss=" in line:
                try:
                    # Extract metrics
                    parts = line.split(":")
                    metrics_part = parts[1].strip()

                    # Extract train metrics
                    train_metrics = metrics_part.split(";")[0].strip()
                    train_loss = float(train_metrics.split("=")[1].split(",")[0])
                    train_ppl = float(train_metrics.split("=")[2])

                    # Extract val metrics
                    val_metrics = metrics_part.split(";")[1].strip()
                    val_loss = float(val_metrics.split("=")[1].split(",")[0])
                    val_ppl = float(val_metrics.split("=")[2])

                    # Log to wandb
                    wandb.log({
                        "train_loss": train_loss,
                        "train_perplexity": train_ppl,
                        "val_loss": val_loss,
                        "val_perplexity": val_ppl
                    })
                except Exception as e:
                    print(f"Error extracting metrics: {e}")

    # Create threads to read stdout and stderr
    import threading
    stdout_thread = threading.Thread(target=read_pipe, args=(process.stdout, stdout_lines))
    stderr_thread = threading.Thread(target=read_pipe, args=(process.stderr, stderr_lines, True))

    # Start threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    return_code = process.wait()

    # Wait for threads to complete
    stdout_thread.join()
    stderr_thread.join()

    # Check if the process was successful
    if return_code != 0:
        print(f"Error running experiment (return code {return_code})")
        stderr = ''.join(stderr_lines)
        print(f"Error output: {stderr}")

    # Log final metrics
    wandb.log({"completed": True, "return_code": return_code})

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--project", type=str, default="ImageCaptioning", help="Wandb project name")
    parser.add_argument("--entity", type=str, default="tuna-ozturk1283", help="Wandb entity name")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing sweep ID to use")
    args = parser.parse_args()

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be slow on CPU.")
        print("Consider installing PyTorch with CUDA support.")

    # Create output directory
    os.makedirs("adapter_code/outputs/sweep", exist_ok=True)

    # Initialize sweep or use existing one
    if args.sweep_id is None:
        print(f"Creating new sweep with configuration:")
        for key, value in sweep_config.items():
            if key != "parameters":
                print(f"  {key}: {value}")
            else:
                print("  parameters:")
                for param, param_config in value.items():
                    print(f"    {param}: {param_config}")

        sweep_id = wandb.sweep(
            sweep_config,
            project=args.project,
            entity=args.entity
        )
        print(f"Created sweep with ID: {sweep_id}")
    else:
        sweep_id = args.sweep_id
        print(f"Using existing sweep with ID: {sweep_id}")

    # Start sweep agent
    print(f"Starting sweep agent to run {args.count} experiments...")
    wandb.agent(sweep_id, train, count=args.count, project=args.project, entity=args.entity)

if __name__ == "__main__":
    main()
