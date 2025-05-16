#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the entire pipeline: preprocessing, training, and evaluation.
"""

import os
import argparse
import subprocess
import time


def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        # Wait for process to complete
        process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode != 0:
            print(f"\nCommand failed with return code {process.returncode}")
            return False
        else:
            print(f"\nCommand completed successfully in {duration:.2f} seconds ({duration/60:.2f} minutes)")
            return True
    except Exception as e:
        print(f"\nError running command: {e}")
        return False


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--data_dir", type=str, default="RISCM", help="Directory containing the dataset")
    parser.add_argument("--adapter_size", type=int, default=8, help="Size of the adapter (LoRA rank)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of examples to use for training")
    parser.add_argument("--val_subset_size", type=int, default=100, help="Number of examples to use for validation")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to use for evaluation")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip dataset analysis")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("paligemma_adapter/outputs", exist_ok=True)
    os.makedirs("paligemma_adapter/evaluation_results", exist_ok=True)
    
    # Step 1: Analyze the dataset
    if not args.skip_analysis:
        analyze_cmd = [
            ".venvic\\Scripts\\python", 
            "paligemma_adapter\\analyze_dataset.py", 
            "--data_dir", args.data_dir
        ]
        
        if not run_command(analyze_cmd, "Step 1: Analyzing the dataset"):
            print("Dataset analysis failed. Exiting.")
            return
    
    # Step 2: Preprocess the dataset
    if not args.skip_preprocessing:
        preprocess_cmd = [
            ".venvic\\Scripts\\python", 
            "paligemma_adapter\\preprocess_dataset.py", 
            "--data_dir", args.data_dir,
            "--output_dir", args.data_dir
        ]
        
        if not run_command(preprocess_cmd, "Step 2: Preprocessing the dataset"):
            print("Dataset preprocessing failed. Exiting.")
            return
    
    # Step 3: Train the model
    if not args.skip_training:
        train_cmd = [
            ".venvic\\Scripts\\python", 
            "paligemma_adapter\\train.py", 
            "--token", args.token,
            "--adapter_size", str(args.adapter_size),
            "--learning_rate", str(args.learning_rate),
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs),
            "--subset_size", str(args.subset_size),
            "--val_subset_size", str(args.val_subset_size),
            "--data_dir", args.data_dir,
            "--output_dir", f"paligemma_adapter/outputs/adapter-{args.adapter_size}-lr{args.learning_rate}"
        ]
        
        if args.use_wandb:
            train_cmd.append("--use_wandb")
        
        if not run_command(train_cmd, "Step 3: Training the model"):
            print("Model training failed. Exiting.")
            return
    
    # Step 4: Evaluate the model
    if not args.skip_evaluation:
        adapter_path = f"paligemma_adapter/outputs/adapter-{args.adapter_size}-lr{args.learning_rate}/adapter-{args.adapter_size}-best"
        
        evaluate_cmd = [
            ".venvic\\Scripts\\python", 
            "paligemma_adapter\\evaluate.py", 
            "--token", args.token,
            "--adapter_path", adapter_path,
            "--num_examples", str(args.num_examples),
            "--data_dir", args.data_dir
        ]
        
        if args.use_wandb:
            evaluate_cmd.append("--use_wandb")
        
        if not run_command(evaluate_cmd, "Step 4: Evaluating the model"):
            print("Model evaluation failed. Exiting.")
            return
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
