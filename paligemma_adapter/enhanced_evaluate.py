#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced evaluation script for PaliGemma model with LoRA adapter-based fine-tuning.
Includes detailed metrics, example visualization, and support for external features.
"""

import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import wandb
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from adapter_config import get_default_config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test_data(csv_path, image_dir, subset_size=None):
    """Load test data from CSV file."""
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        image_path = os.path.join(image_dir, row["image_name"])
        caption = row["caption"]
        data.append((image_path, caption))
    
    # Create a subset if specified
    if subset_size and len(data) > subset_size:
        data = random.sample(data, subset_size)
    
    return data


def calculate_bleu(reference, candidate):
    """Calculate BLEU score."""
    # Tokenize
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    # Calculate BLEU score
    smoothie = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    except:
        bleu = 0.0
    
    return bleu


def calculate_rouge(reference, candidate):
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def calculate_semantic_similarity(reference, candidate, model_name="all-MiniLM-L6-v2"):
    """Calculate semantic similarity using sentence embeddings."""
    try:
        model = SentenceTransformer(model_name)
        ref_embedding = model.encode([reference])
        cand_embedding = model.encode([candidate])
        similarity = cosine_similarity(ref_embedding, cand_embedding)[0][0]
        return similarity
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0


def evaluate_model(model, processor, test_data, device, num_examples=10, generate_max_length=100, num_beams=4):
    """Evaluate the model on test data."""
    model.eval()
    results = []
    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_semantic_sim = 0.0
    
    # Process test data
    progress_bar = tqdm(test_data[:num_examples], desc="Evaluating")
    
    for image_path, reference_caption in progress_bar:
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Process image and text
            text = "<image> Describe the remote sensing image in detail."
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generate_max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode caption
            generated_caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate metrics
            bleu_score = calculate_bleu(reference_caption, generated_caption)
            rouge_scores = calculate_rouge(reference_caption, generated_caption)
            semantic_sim = calculate_semantic_similarity(reference_caption, generated_caption)
            
            # Update totals
            total_bleu += bleu_score
            total_rouge1 += rouge_scores['rouge1']
            total_rouge2 += rouge_scores['rouge2']
            total_rougeL += rouge_scores['rougeL']
            total_semantic_sim += semantic_sim
            
            # Add to results
            results.append({
                "image_path": image_path,
                "reference_caption": reference_caption,
                "generated_caption": generated_caption,
                "bleu_score": bleu_score,
                "rouge1_score": rouge_scores['rouge1'],
                "rouge2_score": rouge_scores['rouge2'],
                "rougeL_score": rouge_scores['rougeL'],
                "semantic_similarity": semantic_sim
            })
            
            # Clear CUDA cache to free up memory
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate average metrics
    num_results = len(results)
    if num_results > 0:
        avg_metrics = {
            "bleu_score": total_bleu / num_results,
            "rouge1_score": total_rouge1 / num_results,
            "rouge2_score": total_rouge2 / num_results,
            "rougeL_score": total_rougeL / num_results,
            "semantic_similarity": total_semantic_sim / num_results
        }
    else:
        avg_metrics = {
            "bleu_score": 0.0,
            "rouge1_score": 0.0,
            "rouge2_score": 0.0,
            "rougeL_score": 0.0,
            "semantic_similarity": 0.0
        }
    
    return results, avg_metrics


def visualize_results(results, output_dir):
    """Visualize evaluation results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for each example
    for i, result in enumerate(results[:5]):  # Visualize first 5 examples
        try:
            # Load image
            image = Image.open(result["image_path"])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.axis('off')
            
            # Add captions
            plt.figtext(0.5, 0.01, f"Reference: {result['reference_caption']}", wrap=True, 
                        horizontalalignment='center', fontsize=10)
            plt.figtext(0.5, 0.05, f"Generated: {result['generated_caption']}", wrap=True, 
                        horizontalalignment='center', fontsize=10)
            plt.figtext(0.5, 0.09, f"BLEU: {result['bleu_score']:.4f}, ROUGE-L: {result['rougeL_score']:.4f}, Semantic Sim: {result['semantic_similarity']:.4f}", 
                        horizontalalignment='center', fontsize=10)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"example_{i+1}.png"), bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing example {i+1}: {e}")
            continue
    
    # Create metrics summary plot
    try:
        metrics = ['bleu_score', 'rouge1_score', 'rouge2_score', 'rougeL_score', 'semantic_similarity']
        values = [[result[metric] for result in results] for metric in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(values, labels=[m.replace('_score', '').upper() for m in metrics])
        ax.set_title('Evaluation Metrics Distribution')
        ax.set_ylabel('Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_summary.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics summary plot: {e}")


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter model")
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-224", help="Path to the base model")
    parser.add_argument("--data_dir", type=str, default="RISCM", help="Directory containing the dataset")
    parser.add_argument("--prompt_template", type=str, default="Describe the remote sensing image in detail.", help="Prompt template")
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to evaluate")
    parser.add_argument("--generate_max_length", type=int, default=100, help="Maximum length for generation")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for generation")
    parser.add_argument("--output_dir", type=str, default="paligemma_adapter/evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("Using CPU")
    
    print(f"Using device: {device}")
    
    # Configuration
    config = get_default_config()
    config.update({
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "prompt_template": args.prompt_template,
        "use_wandb": args.use_wandb,
        "adapter_path": args.adapter_path,
        "num_examples": args.num_examples,
        "generate_max_length": args.generate_max_length,
        "num_beams": args.num_beams,
        "output_dir": args.output_dir,
        "seed": args.seed
    })
    
    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project="ImageCaptioning",
            name=f"eval-enhanced-{os.path.basename(args.adapter_path)}",
            config=config
        )
    
    # Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, token=args.token)
    
    # Load model with memory optimizations
    print("Loading model with memory optimizations...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_path,
        token=args.token,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Move model to device
    model = model.to(device)

    # Load adapter
    print(f"Loading adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Adapter loaded successfully!")

    # Load test data
    print("Loading test data...")
    csv_path = os.path.join(args.data_dir, "test.csv")
    image_dir = os.path.join(args.data_dir, "images")
    test_data = load_test_data(csv_path, image_dir)
    
    # Evaluate model
    print(f"Evaluating model on {args.num_examples} examples...")
    results, avg_metrics = evaluate_model(
        model=model,
        processor=processor,
        test_data=test_data,
        device=device,
        num_examples=args.num_examples,
        generate_max_length=args.generate_max_length,
        num_beams=args.num_beams
    )
    
    # Print average metrics
    print("\nAverage Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print examples
    print("\nExamples:")
    for i, result in enumerate(results[:5]):
        print(f"Example {i+1}:")
        print(f"Reference: {result['reference_caption']}")
        print(f"Generated: {result['generated_caption']}")
        print(f"BLEU: {result['bleu_score']:.4f}, ROUGE-L: {result['rougeL_score']:.4f}, Semantic Sim: {result['semantic_similarity']:.4f}")
        print()
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(results, os.path.join(args.output_dir, os.path.basename(args.adapter_path)))
    
    # Log to wandb
    if args.use_wandb:
        # Log metrics
        wandb.log(avg_metrics)
        
        # Log examples
        for i, result in enumerate(results[:10]):
            try:
                image = Image.open(result["image_path"])
                wandb.log({
                    f"example_{i+1}": wandb.Image(
                        image,
                        caption=f"Reference: {result['reference_caption']}\nGenerated: {result['generated_caption']}\nBLEU: {result['bleu_score']:.4f}, ROUGE-L: {result['rougeL_score']:.4f}"
                    )
                })
            except Exception as e:
                print(f"Error logging image {result['image_path']}: {e}")
        
        # Log visualization plots
        for i in range(min(5, len(results))):
            try:
                plot_path = os.path.join(args.output_dir, os.path.basename(args.adapter_path), f"example_{i+1}.png")
                if os.path.exists(plot_path):
                    wandb.log({f"visualization_{i+1}": wandb.Image(plot_path)})
            except Exception as e:
                print(f"Error logging visualization {i+1}: {e}")
        
        # Log metrics summary plot
        metrics_plot_path = os.path.join(args.output_dir, os.path.basename(args.adapter_path), "metrics_summary.png")
        if os.path.exists(metrics_plot_path):
            wandb.log({"metrics_summary": wandb.Image(metrics_plot_path)})
        
        wandb.finish()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df["adapter_path"] = args.adapter_path
    for metric, value in avg_metrics.items():
        results_df[f"avg_{metric}"] = value
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    output_path = os.path.join(args.output_dir, f"enhanced_results_{os.path.basename(args.adapter_path)}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
