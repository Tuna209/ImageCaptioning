#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarking script for PaliGemma adapter-based fine-tuning.
This script runs experiments with different hyperparameters and configurations.
"""

import os
import argparse
import json
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    get_linear_schedule_with_warmup
)
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Import from local modules
from benchmark_config import BENCHMARK_EXPERIMENTS, get_experiment_config


# Set random seed for reproducibility
def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Dataset class for remote sensing image captioning
class BenchmarkDataset(Dataset):
    """Dataset for remote sensing image captioning with benchmark configurations."""

    def __init__(self, data_dir, split, processor, max_length=128, prompt_template=""):
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        self.max_length = max_length
        self.prompt_template = prompt_template

        # Load the data
        self.data = self._load_data()

    def _load_data(self):
        """Load the data from the data directory."""
        # Load the data from the CSV file
        csv_path = os.path.join(self.data_dir, f"{self.split}.csv")
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Create a list of (image_path, caption) pairs
        data = []
        for _, row in df.iterrows():
            image_path = os.path.join(self.data_dir, "images", row["image_name"])
            caption = row["caption"]
            data.append((image_path, caption))

        return data

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # Get the image path and caption
        image_path, caption = self.data[idx]

        # Load the image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a random item instead
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # Process the image and text with <image> token
        text = f"<image> {self.prompt_template}"

        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get the actual length of the input_ids
        input_length = inputs["input_ids"].shape[1]
        
        # Process the caption as the target with the same length as input_ids
        labels = self.processor.tokenizer(
            caption,
            padding="max_length",
            max_length=input_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        # Set the labels
        inputs["labels"] = labels
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in inputs.items()}

        return inputs


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config):
    """Train the model for one epoch with mixed precision for speed."""
    model.train()
    total_loss = 0
    step_count = 0

    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        try:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / config["gradient_accumulation_steps"]

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Update weights if gradient accumulation steps reached
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                # Unscale gradients for clipping
                if config["max_grad_norm"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                
                # Update weights with scaler
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()
                step_count += 1

                # Clear CUDA cache to free up memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Update progress bar
            loss_value = loss.item() * config["gradient_accumulation_steps"]
            total_loss += loss_value
            progress_bar.set_postfix({"loss": loss_value})

            # Log to wandb
            if config["use_wandb"] and step % 5 == 0:
                wandb.log({
                    "train_loss": loss_value,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": step
                })

        except RuntimeError as e:
            print(f"Error in batch {step}: {e}")
            # Skip this batch
            continue

    # Calculate average loss
    avg_loss = total_loss / max(step_count, 1)  # Avoid division by zero

    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    eval_steps = min(20, len(dataloader))
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= eval_steps:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

    # Calculate average loss
    avg_loss = total_loss / eval_steps

    return avg_loss


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


def evaluate_generation(model, processor, test_data, device, num_examples=5):
    """Evaluate the model's generation capabilities."""
    model.eval()
    bleu_scores = []
    examples = []
    
    # Select random examples
    indices = random.sample(range(len(test_data)), min(num_examples, len(test_data)))
    
    for idx in indices:
        image_path, reference_caption = test_data[idx]
        
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
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            # Decode caption
            generated_caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate BLEU score
            bleu = calculate_bleu(reference_caption, generated_caption)
            bleu_scores.append(bleu)
            
            # Store example
            examples.append({
                "image_path": image_path,
                "reference_caption": reference_caption,
                "generated_caption": generated_caption,
                "bleu_score": bleu
            })
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    # Calculate average BLEU score
    avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    
    return avg_bleu, examples


def run_experiment(experiment_name, args):
    """Run a single experiment with the specified configuration."""
    # Get experiment configuration
    config = get_experiment_config(experiment_name)
    config.update({
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "output_dir": os.path.join(args.output_dir, experiment_name),
        "use_wandb": args.use_wandb,
        "token": args.token,
        "seed": args.seed
    })
    
    # Set random seed
    set_seed(config["seed"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config["use_wandb"]:
        run_name = f"benchmark-{experiment_name}"
        wandb.init(
            project="ImageCaptioning",
            name=run_name,
            config=config,
            group="benchmarking"
        )
    
    # Load model, tokenizer, and processor
    print("Loading model, tokenizer, and processor...")
    processor = AutoProcessor.from_pretrained(config["model_path"], token=config["token"])
    
    # Load model with memory optimizations
    print("Loading model with memory optimizations...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config["model_path"],
        token=config["token"],
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Move model to device
    model = model.to(device)
    
    # Apply LoRA adapters to the model
    print("Applying LoRA adapters to the model...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["adapter_size"],
        lora_alpha=config["adapter_size"] * 2,
        lora_dropout=config["adapter_dropout"],
        init_lora_weights="gaussian",
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset_full = BenchmarkDataset(
        data_dir=config["data_dir"],
        split="train",
        processor=processor,
        max_length=config["max_length"],
        prompt_template=config["prompt_template"]
    )
    val_dataset_full = BenchmarkDataset(
        data_dir=config["data_dir"],
        split="val",
        processor=processor,
        max_length=config["max_length"],
        prompt_template=config["prompt_template"]
    )
    test_dataset_full = BenchmarkDataset(
        data_dir=config["data_dir"],
        split="test",
        processor=processor,
        max_length=config["max_length"],
        prompt_template=config["prompt_template"]
    )
    
    # Create subsets for faster training
    train_indices = random.sample(range(len(train_dataset_full)), min(config["subset_size"], len(train_dataset_full)))
    val_indices = random.sample(range(len(val_dataset_full)), min(config["val_subset_size"], len(val_dataset_full)))
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    print(f"Using {len(train_dataset)} train examples (out of {len(train_dataset_full)})")
    print(f"Using {len(val_dataset)} val examples (out of {len(val_dataset_full)})")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Create scheduler
    total_steps = len(train_dataloader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Train the model
    print(f"Starting training for experiment: {experiment_name}...")
    best_val_loss = float("inf")
    results = {
        "experiment_name": experiment_name,
        "config": config,
        "epochs": []
    }
    
    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            config=config
        )
        
        # Evaluate
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device
        )
        
        # Print results
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Log to wandb
        if config["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save the model
            output_dir = os.path.join(config["output_dir"], "best")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
        
        # Store epoch results
        results["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # Save the final model
    print("Saving final model...")
    output_dir = os.path.join(config["output_dir"], "final")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Evaluate generation on test set
    print("Evaluating generation on test set...")
    avg_bleu, examples = evaluate_generation(
        model=model,
        processor=processor,
        test_data=test_dataset_full.data,
        device=device,
        num_examples=5
    )
    
    print(f"Average BLEU score: {avg_bleu:.4f}")
    
    # Log to wandb
    if config["use_wandb"]:
        wandb.log({"avg_bleu": avg_bleu})
        
        # Log examples
        for i, example in enumerate(examples):
            try:
                image = Image.open(example["image_path"])
                wandb.log({
                    f"example_{i+1}": wandb.Image(
                        image,
                        caption=f"Reference: {example['reference_caption']}\nGenerated: {example['generated_caption']}\nBLEU: {example['bleu_score']:.4f}"
                    )
                })
            except Exception as e:
                print(f"Error logging image {example['image_path']}: {e}")
    
    # Store final results
    results["best_val_loss"] = best_val_loss
    results["avg_bleu"] = avg_bleu
    results["examples"] = examples
    
    # Save results to file
    results_path = os.path.join(config["output_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Finish wandb
    if config["use_wandb"]:
        wandb.finish()
    
    print(f"Experiment {experiment_name} complete!")
    
    return results


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--data_dir", type=str, default="RISCM_processed", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-224", help="Path to the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiments", type=str, nargs="+", help="Specific experiments to run")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get experiments to run
    if args.experiments:
        experiments = args.experiments
    else:
        experiments = [exp["name"] for exp in BENCHMARK_EXPERIMENTS]
    
    # Run experiments
    results = []
    for experiment_name in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*50}\n")
        
        try:
            result = run_experiment(experiment_name, args)
            results.append(result)
        except Exception as e:
            print(f"Error running experiment {experiment_name}: {e}")
            continue
    
    # Summarize results
    print("\n\nBenchmark Results Summary:")
    print(f"{'='*50}")
    print(f"{'Experiment':<30} {'Best Val Loss':<15} {'BLEU Score':<15}")
    print(f"{'-'*30} {'-'*15} {'-'*15}")
    
    for result in results:
        print(f"{result['experiment_name']:<30} {result['best_val_loss']:<15.4f} {result['avg_bleu']:<15.4f}")
    
    # Save summary to file
    summary = {
        "experiments": [
            {
                "name": result["experiment_name"],
                "best_val_loss": result["best_val_loss"],
                "avg_bleu": result["avg_bleu"]
            }
            for result in results
        ]
    }
    
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBenchmark summary saved to {summary_path}")


if __name__ == "__main__":
    main()
