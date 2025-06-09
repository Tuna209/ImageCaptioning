#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main evaluation script for PaliGemma models.
"""

import os
import argparse
import json
from dotenv import load_dotenv
import wandb
from PIL import Image

from evaluator import PaliGemmaEvaluator
from metrics import calculate_comprehensive_metrics, print_metrics_summary

# Load environment variables
load_dotenv()

def log_evaluation_to_wandb(results, metrics, images_dir, run_name, project_name="ImageCaptioning"):
    """Log evaluation results to WANDB with images and captions."""

    # Initialize WANDB
    wandb.init(
        project=project_name,
        name=run_name,
        job_type="evaluation",
        tags=["evaluation", "paligemma"]
    )

    # Log metrics
    wandb.log(metrics)

    # Prepare image data for WANDB
    wandb_images = []

    for i, result in enumerate(results[:20]):  # Limit to first 20 for display
        image_path = os.path.join(images_dir, result['image'])

        try:
            # Load image
            image = Image.open(image_path)

            # Create caption with comparison
            caption = f"""
Ground Truth: {result['ground_truth']}
Generated: {result['generated_caption']}
BLEU-1: {metrics['individual_bleu_1'][i]:.3f}
ROUGE-L: {metrics['individual_rouge_l'][i]:.3f}
Domain Relevance: {metrics['individual_domain_relevance'][i]:.3f}
            """.strip()

            # Add to WANDB images
            wandb_images.append(wandb.Image(
                image,
                caption=caption
            ))

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

    # Log images
    wandb.log({"evaluation_samples": wandb_images})

    # Log summary table
    table_data = []
    for i, result in enumerate(results):
        table_data.append([
            result['image'],
            result['ground_truth'],
            result['generated_caption'],
            metrics['individual_bleu_1'][i],
            metrics['individual_rouge_l'][i],
            metrics['individual_domain_relevance'][i]
        ])

    table = wandb.Table(
        columns=["Image", "Ground Truth", "Generated", "BLEU-1", "ROUGE-L", "Domain Relevance"],
        data=table_data
    )
    wandb.log({"evaluation_results": table})

    print(f" Evaluation results logged to WANDB: {run_name}")
    wandb.finish()

def run_baseline_evaluation(args):
    """Run baseline model evaluation."""
    print(" Starting baseline evaluation...")
    
    evaluator = PaliGemmaEvaluator()
    
    # Load baseline model
    processor, model = evaluator.load_baseline_model(token=args.token)
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set(
        test_csv=args.test_csv,
        images_dir=args.images_dir,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples
    )
    
    # Calculate basic metrics
    basic_metrics = evaluator.calculate_basic_metrics(results)
    
    # Calculate comprehensive metrics
    references = [r['ground_truth'] for r in results]
    hypotheses = [r['generated_caption'] for r in results]
    comprehensive_metrics = calculate_comprehensive_metrics(references, hypotheses)
    
    # Combine metrics
    all_metrics = {**basic_metrics, **comprehensive_metrics}
    
    # Print results
    evaluator.print_results(basic_metrics, results)
    print_metrics_summary(comprehensive_metrics)
    
    # Save results
    output_file = os.path.join(args.output_dir, "baseline_evaluation.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator.save_results(results, all_metrics, output_file, "Baseline PaliGemma")
    
    print(f"\n Baseline evaluation complete!")
    return results, all_metrics

def run_finetuned_evaluation(args):
    """Run fine-tuned model evaluation."""
    print(" Starting fine-tuned model evaluation...")
    
    evaluator = PaliGemmaEvaluator()
    
    # Load fine-tuned model
    processor, model = evaluator.load_finetuned_model(
        adapter_path=args.adapter_path,
        token=args.token
    )
    
    # Evaluate on test set
    results = evaluator.evaluate_on_test_set(
        test_csv_path=args.test_csv,
        images_dir=args.images_dir,
        prompt_template=args.prompt_template,
        max_samples=args.max_samples
    )
    
    # Calculate basic metrics
    basic_metrics = evaluator.calculate_basic_metrics(results)
    
    # Calculate comprehensive metrics
    references = [r['ground_truth'] for r in results]
    hypotheses = [r['generated_caption'] for r in results]
    comprehensive_metrics = calculate_comprehensive_metrics(references, hypotheses)
    
    # Combine metrics
    all_metrics = {**basic_metrics, **comprehensive_metrics}
    
    # Print results
    evaluator.print_results(basic_metrics, results)
    print_metrics_summary(comprehensive_metrics)
    
    # Save results
    output_file = os.path.join(args.output_dir, "finetuned_evaluation.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluator.save_results(results, all_metrics, output_file, "Fine-tuned PaliGemma")

    # Log to WANDB if requested
    if hasattr(args, 'use_wandb') and args.use_wandb:
        run_name = getattr(args, 'wandb_run_name', 'paligemma-adapter-8-lr1e-05-evaluation')
        log_evaluation_to_wandb(results, all_metrics, args.images_dir, run_name)

    print(f"\n Fine-tuned evaluation complete!")
    return results, all_metrics

def compare_models(baseline_metrics_file, finetuned_metrics_file, output_dir):
    """Compare baseline and fine-tuned model performance."""
    
    print(f"\n COMPARING BASELINE vs FINE-TUNED MODELS")
    print("=" * 60)
    
    # Load metrics
    with open(baseline_metrics_file, 'r') as f:
        baseline_metrics = json.load(f)
    
    with open(finetuned_metrics_file, 'r') as f:
        finetuned_metrics = json.load(f)
    
    # Calculate improvements
    improvements = {}
    for metric in ['mean_bleu_1', 'mean_rouge_l', 'mean_domain_relevance']:
        if metric in baseline_metrics and metric in finetuned_metrics:
            baseline_val = baseline_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            improvement = ((finetuned_val - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0
            improvements[f"{metric}_improvement"] = improvement
            
            print(f" {metric.replace('mean_', '').replace('_', ' ').title()}:")
            print(f"   Baseline: {baseline_val:.4f}")
            print(f"   Fine-tuned: {finetuned_val:.4f}")
            print(f"   Improvement: {improvement:+.1f}%")
            print()
    
    # Save comparison
    comparison_file = os.path.join(output_dir, "model_comparison.json")
    comparison_data = {
        'baseline_metrics': baseline_metrics,
        'finetuned_metrics': finetuned_metrics,
        'improvements': improvements
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f" Comparison saved to: {comparison_file}")
    print("=" * 60)

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma models")
    parser.add_argument("--mode", type=str, choices=['baseline', 'finetuned', 'compare'], 
                       required=True, help="Evaluation mode")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter (for finetuned mode)")
    parser.add_argument("--test_csv", type=str, default="processed_dataset/test.csv", 
                       help="Test CSV file")
    parser.add_argument("--images_dir", type=str, default="processed_dataset/images", 
                       help="Images directory")
    parser.add_argument("--prompt_template", type=str, 
                       default="Describe the remote sensing image in detail.", 
                       help="Prompt template")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Maximum samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Log results to WANDB")
    parser.add_argument("--wandb_run_name", type=str,
                       default="paligemma-evaluation",
                       help="WANDB run name")

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "baseline":
        run_baseline_evaluation(args)
    
    elif args.mode == "finetuned":
        if not args.adapter_path:
            raise ValueError("--adapter_path is required for finetuned mode")
        run_finetuned_evaluation(args)
    
    elif args.mode == "compare":
        # Look for existing evaluation files
        baseline_file = os.path.join(args.output_dir, "baseline_evaluation_metrics.json")
        finetuned_file = os.path.join(args.output_dir, "finetuned_evaluation_metrics.json")
        
        if os.path.exists(baseline_file) and os.path.exists(finetuned_file):
            compare_models(baseline_file, finetuned_file, args.output_dir)
        else:
            print(" Missing evaluation files. Run baseline and finetuned evaluations first.")

if __name__ == "__main__":
    main()
