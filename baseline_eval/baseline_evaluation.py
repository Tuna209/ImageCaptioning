#!/usr/bin/env python3
"""
Baseline PaliGemma Evaluation using EXACT same metrics as fine-tuned models
This will prove the baseline performance and show improvement from fine-tuning
"""

import os
import sys
import json
import csv
import torch
import wandb
import numpy as np
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from collections import Counter
import argparse

# Add parent directory to path to import evaluation modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))
from metrics import calculate_bleu_scores, calculate_rouge_scores, calculate_domain_relevance

def calculate_rouge_l(reference, candidate):
    """Calculate individual ROUGE-L score for a single pair"""
    rouge_result = calculate_rouge_scores([reference], [candidate])
    return rouge_result['rouge_l']

def calculate_domain_relevance_single(caption):
    """Calculate domain relevance for a single caption"""
    domain_result = calculate_domain_relevance([caption])
    return domain_result['domain_relevance']

class BaselinePaliGemmaEvaluator:
    def __init__(self, token):
        self.token = token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_baseline_model(self):
        """Load the original PaliGemma model without any fine-tuning"""
        print("Loading baseline PaliGemma model (NO fine-tuning)...")
        print("Model: google/paligemma-3b-mix-224")

        # Load processor
        processor = PaliGemmaProcessor.from_pretrained(
            "google/paligemma-3b-mix-224",
            token=self.token
        )

        # Load model with memory optimizations
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            token=self.token,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("Baseline PaliGemma model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("IMPORTANT: This is the ORIGINAL model with NO fine-tuning!")

        return processor, model
    
    def generate_caption(self, processor, model, image_path, prompt):
        """Generate caption for an image using baseline model"""
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate caption
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,  # Same as fine-tuned models
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract caption (remove prompt)
            caption = generated_text.replace(prompt, "").strip()
            
            return caption

        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return ""

    def load_test_data(self, test_csv, max_samples=100):
        """Load test data"""
        print(f"Loading test data from: {test_csv}")

        test_data = []
        with open(test_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                test_data.append({
                    'image': row['image'],
                    'caption': row['caption']
                })

        print(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def evaluate_baseline(self, test_csv, images_dir, max_samples=100, use_wandb=True, wandb_run_name=None):
        """Evaluate baseline PaliGemma model using EXACT same metrics as fine-tuned models"""
        print("BASELINE PALIGEMMA EVALUATION")
        print("=" * 60)
        print("Using EXACT same metrics as fine-tuned model evaluation:")
        print("   - BLEU-1, BLEU-2, BLEU-3, BLEU-4")
        print("   - ROUGE-L")
        print("   - Domain Relevance")
        print("   - Vocabulary Diversity")
        print("   - Same test images and prompts")
        print()

        # Load baseline model
        processor, model = self.load_baseline_model()

        # Load test data
        test_data = self.load_test_data(test_csv, max_samples)

        # Evaluation setup
        prompt = "Describe the remote sensing image in detail."
        print(f"Using prompt: '{prompt}'")
        print(f"Evaluating on {len(test_data)} images")
        print()
        
        # Initialize Wandb if requested
        if use_wandb:
            wandb.init(
                project="ImageCaptioning",
                name=wandb_run_name or "baseline-paligemma-evaluation",
                job_type="evaluation",
                tags=["baseline", "paligemma", "no-finetuning", "evaluation"]
            )
        
        # Evaluation results
        results = []
        generated_captions = []
        ground_truth_captions = []

        print("Starting evaluation...")

        for i, sample in enumerate(test_data):
            print(f"Evaluating: {i+1:3d}/{len(test_data)} | {sample['image']}", end="")

            # Generate caption
            image_path = os.path.join(images_dir, sample['image'])
            generated_caption = self.generate_caption(processor, model, image_path, prompt)

            if generated_caption:
                generated_captions.append(generated_caption)
                ground_truth_captions.append(sample['caption'])

                # Store result
                result = {
                    'image': sample['image'],
                    'ground_truth': sample['caption'],
                    'generated': generated_caption,
                    'ground_truth_length': len(sample['caption'].split()),
                    'generated_length': len(generated_caption.split())
                }
                results.append(result)

                # Show first few examples
                if i < 3:
                    print(f"\nExample {i+1}:")
                    print(f"   Image: {sample['image']}")
                    print(f"   Ground Truth: {sample['caption'][:80]}...")
                    print(f"   Generated: {generated_caption[:80]}...")
                else:
                    print(" OK")
            else:
                print(" FAILED")

        print(f"\nSuccessfully generated {len(generated_captions)} captions")

        # Calculate metrics using EXACT same functions as fine-tuned evaluation
        print("\nCalculating metrics...")
        
        # BLEU scores
        bleu_scores = calculate_bleu_scores(ground_truth_captions, generated_captions)
        
        # ROUGE-L scores
        rouge_l_scores = [
            calculate_rouge_l(gt, gen)
            for gt, gen in zip(ground_truth_captions, generated_captions)
        ]

        # Domain relevance scores
        domain_scores = [
            calculate_domain_relevance_single(gen)
            for gen in generated_captions
        ]
        
        # Vocabulary analysis
        all_words = []
        for caption in generated_captions:
            all_words.extend(caption.lower().split())
        
        word_counts = Counter(all_words)
        vocabulary_size = len(set(all_words))
        type_token_ratio = vocabulary_size / len(all_words) if all_words else 0
        
        # Calculate summary metrics
        metrics = {
            'total_samples': len(test_data),
            'successful_generations': len(generated_captions),
            'success_rate': len(generated_captions) / len(test_data),
            'mean_bleu_1': np.mean(bleu_scores['bleu_1']),
            'mean_bleu_2': np.mean(bleu_scores['bleu_2']),
            'mean_bleu_3': np.mean(bleu_scores['bleu_3']),
            'mean_bleu_4': np.mean(bleu_scores['bleu_4']),
            'mean_rouge_l': np.mean(rouge_l_scores),
            'mean_domain_relevance': np.mean(domain_scores),
            'avg_ground_truth_length': np.mean([len(gt.split()) for gt in ground_truth_captions]),
            'avg_generated_length': np.mean([len(gen.split()) for gen in generated_captions]),
            'vocabulary_size': vocabulary_size,
            'type_token_ratio': type_token_ratio,
            'most_common_words': dict(word_counts.most_common(10))
        }
        
        return results, metrics, processor, model

    def print_results(self, metrics):
        """Print evaluation results in the same format as fine-tuned models"""
        print("\nBASELINE PALIGEMMA EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Successful Generations: {metrics['successful_generations']}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Avg Ground Truth Length: {metrics['avg_ground_truth_length']:.1f} words")
        print(f"Avg Generated Length: {metrics['avg_generated_length']:.1f} words")
        print()

        print("COMPREHENSIVE EVALUATION METRICS")
        print("=" * 60)
        print(f"BLEU-1: {metrics['mean_bleu_1']:.4f}")
        print(f"BLEU-2: {metrics['mean_bleu_2']:.4f}")
        print(f"BLEU-3: {metrics['mean_bleu_3']:.4f}")
        print(f"BLEU-4: {metrics['mean_bleu_4']:.4f}")
        print(f"ROUGE-L: {metrics['mean_rouge_l']:.4f}")
        print(f"Domain Relevance: {metrics['mean_domain_relevance']:.4f}")
        print(f"Vocabulary Size: {metrics['vocabulary_size']}")
        print(f"Type-Token Ratio: {metrics['type_token_ratio']:.4f}")
        print()

        print("Most Common Words:")
        for word, count in metrics['most_common_words'].items():
            print(f"   {word}: {count}")
        print("=" * 60)

    def log_to_wandb(self, results, metrics, images_dir):
        """Log results to Wandb with images"""
        print("\nLogging results to Wandb...")

        # Log summary metrics
        wandb.log({
            'total_samples': metrics['total_samples'],
            'successful_generations': metrics['successful_generations'],
            'success_rate': metrics['success_rate'],
            'mean_bleu_1': metrics['mean_bleu_1'],
            'mean_bleu_2': metrics['mean_bleu_2'],
            'mean_bleu_3': metrics['mean_bleu_3'],
            'mean_bleu_4': metrics['mean_bleu_4'],
            'mean_rouge_l': metrics['mean_rouge_l'],
            'mean_domain_relevance': metrics['mean_domain_relevance'],
            'avg_ground_truth_length': metrics['avg_ground_truth_length'],
            'avg_generated_length': metrics['avg_generated_length'],
            'vocabulary_size': metrics['vocabulary_size'],
            'type_token_ratio': metrics['type_token_ratio'],
            'model_name': 'Baseline PaliGemma (No Fine-tuning)'
        })

        # Log sample images with captions
        wandb_images = []
        for i, result in enumerate(results[:100]):  # Log all 100 images
            try:
                image_path = os.path.join(images_dir, result['image'])
                image = Image.open(image_path)

                caption = f"""
Ground Truth: {result['ground_truth']}
Generated: {result['generated']}
GT Length: {result['ground_truth_length']} words | Gen Length: {result['generated_length']} words
                """.strip()

                wandb_images.append(wandb.Image(image, caption=caption))
            except Exception as e:
                print(f"Error logging image {result['image']}: {e}")
                continue

        wandb.log({"baseline_evaluation_images": wandb_images})
        print(f"Logged {len(wandb_images)} images to Wandb")

    def save_results(self, results, metrics, output_dir="baseline_results"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        results_file = os.path.join(output_dir, "baseline_evaluation.csv")
        with open(results_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        # Save metrics
        metrics_file = os.path.join(output_dir, "baseline_evaluation_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"Results saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")

def main():
    """Main function to run baseline evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate baseline PaliGemma model")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--test_csv", default="../processed_dataset/test.csv", help="Test CSV file")
    parser.add_argument("--images_dir", default="../RISCM/resized", help="Images directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to evaluate")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use Wandb logging")
    parser.add_argument("--wandb_run_name", default="baseline-paligemma-evaluation-100-samples", help="Wandb run name")
    parser.add_argument("--output_dir", default="baseline_results", help="Output directory")

    args = parser.parse_args()

    print("BASELINE PALIGEMMA EVALUATION")
    print("=" * 60)
    print("This evaluation will:")
    print("   - Load the ORIGINAL PaliGemma model (NO fine-tuning)")
    print("   - Use EXACT same metrics as fine-tuned model evaluation")
    print("   - Use EXACT same test images and prompts")
    print("   - Calculate BLEU-4 (primary metric) for comparison")
    print("   - Log 100 images with captions to Wandb")
    print("   - Prove this is the true baseline performance")
    print()

    # Initialize evaluator
    evaluator = BaselinePaliGemmaEvaluator(args.token)

    # Run evaluation
    results, metrics, processor, model = evaluator.evaluate_baseline(
        test_csv=args.test_csv,
        images_dir=args.images_dir,
        max_samples=args.max_samples,
        use_wandb=args.use_wandb,
        wandb_run_name=args.wandb_run_name
    )

    # Print results
    evaluator.print_results(metrics)

    # Log to Wandb
    if args.use_wandb:
        evaluator.log_to_wandb(results, metrics, args.images_dir)
        wandb.finish()

    # Save results
    evaluator.save_results(results, metrics, args.output_dir)

    print("\nBASELINE EVALUATION COMPLETED!")
    print("This proves the original PaliGemma performance before fine-tuning")
    print(f"BLEU-4 Score: {metrics['mean_bleu_4']:.4f}")
    print("Compare this with fine-tuned model BLEU-4 scores to see improvement!")

if __name__ == "__main__":
    main()
