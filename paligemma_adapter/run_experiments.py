#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run experiments with PaliGemma model using different LoRA adapter sizes and learning rates.
"""

import os
import argparse
import subprocess
import time
import json
import pandas as pd


def run_experiment(token, adapter_size, learning_rate, batch_size, gradient_accumulation_steps, num_epochs, subset_size, val_subset_size, use_wandb=True):
    """Run an experiment with the given parameters."""
    # Build command
    cmd = [
        ".venvic\\Scripts\\python", 
        "paligemma_adapter\\train.py", 
        "--token", token,
        "--adapter_size", str(adapter_size),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--num_epochs", str(num_epochs),
        "--subset_size", str(subset_size),
        "--val_subset_size", str(val_subset_size),
        "--output_dir", f"paligemma_adapter/outputs/adapter-{adapter_size}-lr{learning_rate}"
    ]
    
    if use_wandb:
        cmd.append("--use_wandb")
        
    # Print command
    print(f"Running command: {' '.join(cmd)}")
    
    # Run command
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
        
        # Check return code
        if process.returncode != 0:
            print(f"Experiment failed with return code {process.returncode}")
            return False, None
        else:
            print(f"Experiment completed successfully")
            # Return the path to the best model
            adapter_path = f"paligemma_adapter/outputs/adapter-{adapter_size}-lr{learning_rate}/adapter-{adapter_size}-best"
            return True, adapter_path
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False, None


def evaluate_model(token, adapter_path, num_examples=10, use_wandb=True):
    """Evaluate the model with the given adapter."""
    # Build command
    cmd = [
        ".venvic\\Scripts\\python", 
        "paligemma_adapter\\evaluate.py", 
        "--token", token,
        "--adapter_path", adapter_path,
        "--num_examples", str(num_examples)
    ]
    
    if use_wandb:
        cmd.append("--use_wandb")
        
    # Print command
    print(f"Running command: {' '.join(cmd)}")
    
    # Run command
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
        
        # Check return code
        if process.returncode != 0:
            print(f"Evaluation failed with return code {process.returncode}")
            return False, None
        else:
            print(f"Evaluation completed successfully")
            # Get the BLEU score from the CSV file
            results_path = os.path.join("paligemma_adapter/evaluation_results", f"results_{os.path.basename(adapter_path)}.csv")
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                avg_bleu = results_df["avg_bleu_score"].iloc[0]
                return True, avg_bleu
            else:
                print(f"Results file not found: {results_path}")
                return True, None
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return False, None


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--experiments", type=str, default="all", help="Comma-separated list of experiments to run (small, medium, large, all)")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of examples to use for training")
    parser.add_argument("--val_subset_size", type=int, default=100, help="Number of examples to use for validation")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to use for evaluation")
    args = parser.parse_args()
    
    # Define experiments
    experiments = {
        "small": {
            "adapter_size": 8,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "description": "Small adapter (rank=8) with higher learning rate"
        },
        "medium": {
            "adapter_size": 16,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "description": "Medium adapter (rank=16) with medium learning rate"
        },
        "large": {
            "adapter_size": 32,
            "learning_rate": 1e-5,
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "num_epochs": 3,
            "description": "Large adapter (rank=32) with lower learning rate"
        }
    }
    
    # Determine which experiments to run
    if args.experiments.lower() == "all":
        experiments_to_run = list(experiments.keys())
    else:
        experiments_to_run = [exp.strip() for exp in args.experiments.split(",")]
        
    # Validate experiments
    for exp in experiments_to_run:
        if exp not in experiments:
            print(f"Unknown experiment: {exp}")
            print(f"Available experiments: {', '.join(experiments.keys())}")
            return
    
    # Create results directory
    os.makedirs("paligemma_adapter/experiment_results", exist_ok=True)
    
    # Run experiments
    results = {}
    for exp in experiments_to_run:
        print(f"\n{'='*80}")
        print(f"Running experiment: {exp} - {experiments[exp]['description']}")
        print(f"{'='*80}\n")
        
        # Record start time
        start_time = time.time()
        
        # Run experiment
        training_success, adapter_path = run_experiment(
            token=args.token,
            adapter_size=experiments[exp]["adapter_size"],
            learning_rate=experiments[exp]["learning_rate"],
            batch_size=experiments[exp]["batch_size"],
            gradient_accumulation_steps=experiments[exp]["gradient_accumulation_steps"],
            num_epochs=experiments[exp]["num_epochs"],
            subset_size=args.subset_size,
            val_subset_size=args.val_subset_size,
            use_wandb=args.use_wandb
        )
        
        # Record end time
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Initialize results
        results[exp] = {
            "training": {
                "success": training_success,
                "duration": training_duration,
                "adapter_path": adapter_path
            },
            "evaluation": {
                "success": None,
                "duration": None,
                "bleu_score": None
            },
            "description": experiments[exp]["description"],
            "parameters": {
                "adapter_size": experiments[exp]["adapter_size"],
                "learning_rate": experiments[exp]["learning_rate"],
                "batch_size": experiments[exp]["batch_size"],
                "gradient_accumulation_steps": experiments[exp]["gradient_accumulation_steps"],
                "num_epochs": experiments[exp]["num_epochs"],
                "subset_size": args.subset_size,
                "val_subset_size": args.val_subset_size
            }
        }
        
        print(f"\nExperiment {exp} training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
        
        # Evaluate the model if training was successful
        if training_success and adapter_path:
            print(f"\n{'='*80}")
            print(f"Evaluating model: {exp} - {adapter_path}")
            print(f"{'='*80}\n")
            
            # Record start time
            start_time = time.time()
            
            # Evaluate model
            evaluation_success, bleu_score = evaluate_model(
                token=args.token,
                adapter_path=adapter_path,
                num_examples=args.num_examples,
                use_wandb=args.use_wandb
            )
            
            # Record end time
            end_time = time.time()
            evaluation_duration = end_time - start_time
            
            # Update results
            results[exp]["evaluation"]["success"] = evaluation_success
            results[exp]["evaluation"]["duration"] = evaluation_duration
            results[exp]["evaluation"]["bleu_score"] = bleu_score
            
            print(f"\nExperiment {exp} evaluation completed in {evaluation_duration:.2f} seconds ({evaluation_duration/60:.2f} minutes)")
    
    # Print summary
    print("\n\n")
    print(f"{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}")
    
    for exp, result in results.items():
        training_status = "SUCCESS" if result["training"]["success"] else "FAILED"
        evaluation_status = "SUCCESS" if result["evaluation"]["success"] else "FAILED" if result["evaluation"]["success"] is not None else "SKIPPED"
        
        print(f"{exp}: {result['description']}")
        print(f"  Training: {training_status} - Duration: {result['training']['duration']/60:.2f} minutes")
        print(f"  Evaluation: {evaluation_status}")
        if result["evaluation"]["bleu_score"]:
            print(f"  BLEU Score: {result['evaluation']['bleu_score']:.4f}")
        print()
    
    # Save results to file
    with open("paligemma_adapter/experiment_results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to paligemma_adapter/experiment_results/summary.json")


if __name__ == "__main__":
    main()
