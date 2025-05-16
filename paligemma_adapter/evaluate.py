#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for PaliGemma model with LoRA adapter-based fine-tuning.
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


def generate_caption(model, processor, image_path, device, prompt_template="Describe the remote sensing image in detail."):
    """Generate a caption for an image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process image and text
    text = f"<image> {prompt_template}"
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
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
    
    # Decode caption
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return caption


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
        "seed": args.seed
    })
    
    # Initialize wandb
    if config["use_wandb"]:
        wandb.init(
            project="ImageCaptioning",
            name=f"eval-{os.path.basename(args.adapter_path)}",
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
    data = load_test_data(csv_path, image_dir, args.num_examples)
    
    # Generate captions
    print("Generating captions...")
    results = []
    total_bleu = 0.0
    
    for image_path, reference_caption in tqdm(data, desc="Generating captions"):
        # Generate caption
        generated_caption = generate_caption(
            model=model,
            processor=processor,
            image_path=image_path,
            device=device,
            prompt_template=args.prompt_template
        )
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(reference_caption, generated_caption)
        total_bleu += bleu_score
        
        # Add to results
        results.append({
            "image_path": image_path,
            "reference_caption": reference_caption,
            "generated_caption": generated_caption,
            "bleu_score": bleu_score
        })
    
    # Calculate average BLEU score
    avg_bleu = total_bleu / len(data)
    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    
    # Print examples
    print("\nExamples:")
    for i, result in enumerate(results[:5]):
        print(f"Example {i+1}:")
        print(f"Reference: {result['reference_caption']}")
        print(f"Generated: {result['generated_caption']}")
        print(f"BLEU score: {result['bleu_score']:.4f}")
        print()
    
    # Log to wandb
    if args.use_wandb:
        # Log metrics
        wandb.log({"bleu_score": avg_bleu})
        
        # Log examples
        for i, result in enumerate(results[:10]):
            try:
                image = Image.open(result["image_path"])
                wandb.log({
                    f"example_{i+1}": wandb.Image(
                        image,
                        caption=f"Reference: {result['reference_caption']}\nGenerated: {result['generated_caption']}\nBLEU: {result['bleu_score']:.4f}"
                    )
                })
            except Exception as e:
                print(f"Error logging image {result['image_path']}: {e}")
        
        wandb.finish()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df["adapter_path"] = args.adapter_path
    results_df["avg_bleu_score"] = avg_bleu
    
    # Create output directory
    os.makedirs("paligemma_adapter/evaluation_results", exist_ok=True)
    
    # Save results
    output_path = os.path.join("paligemma_adapter/evaluation_results", f"results_{os.path.basename(args.adapter_path)}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
