#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for PaliGemma model with LoRA adapter-based fine-tuning.
"""

import os
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
import wandb
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from dataset import create_dataloaders
from adapter_config import get_lora_config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Training")

    print(f"Starting training epoch with {len(dataloader)} batches")
    print(f"Gradient accumulation steps: {config['gradient_accumulation_steps']}")

    for step, batch in enumerate(progress_bar):
        try:
            # Print batch information
            if step == 0:
                print(f"Batch keys: {batch.keys()}")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Get loss
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / config["gradient_accumulation_steps"]

            # Backward pass
            loss.backward()

            # Update progress bar
            loss_value = loss.item() * config["gradient_accumulation_steps"]
            total_loss += loss_value
            progress_bar.set_postfix({"loss": loss_value})

            # Log to wandb
            if config["use_wandb"] and step % 10 == 0:
                wandb.log({
                    "train_loss": loss_value,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })

            # Update weights
            if (step + 1) % config["gradient_accumulation_steps"] == 0 or step == len(dataloader) - 1:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                # Update weights
                optimizer.step()
                scheduler.step()

                # Zero gradients
                optimizer.zero_grad()

                # Print GPU memory usage
                if torch.cuda.is_available() and (step + 1) % (config["gradient_accumulation_steps"] * 10) == 0:
                    print(f"Step {step + 1}/{len(dataloader)}, GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        except Exception as e:
            print(f"Error in training step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0

    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Get loss
            loss = outputs.loss

            # Update progress bar
            loss_value = loss.item()
            total_loss += loss_value
            progress_bar.set_postfix({"loss": loss_value})

    return total_loss / len(dataloader)

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional, will use HF_TOKEN from .env if not provided)")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")
    parser.add_argument("--adapter_size", type=int, default=8, help="Size of the adapter (LoRA rank)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--data_dir", type=str, default="processed_dataset", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save the model")
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-224", help="Path to the model")
    parser.add_argument("--prompt_template", type=str, default="Describe the remote sensing image in detail.", help="Prompt template")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the input")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of examples to use for training")
    parser.add_argument("--val_subset_size", type=int, default=100, help="Number of examples to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Get HF token from environment if not provided as argument
    if args.token is None:
        args.token = os.getenv("HF_TOKEN")
        if args.token is None:
            raise ValueError("HF_TOKEN must be provided either as --token argument or in .env file")

    # Print all arguments (but hide the token for security)
    print("Arguments:")
    for arg in vars(args):
        if arg == "token":
            print(f"  {arg}: {'*' * 20}")  # Hide token for security
        else:
            print(f"  {arg}: {getattr(args, arg)}")

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
    config = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "max_length": args.max_length,
        "prompt_template": args.prompt_template,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_grad_norm": args.max_grad_norm,
        "use_wandb": args.use_wandb,
        "adapter_size": args.adapter_size,
        "adapter_dropout": 0.05,
        "adapter_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        "output_dir": args.output_dir,
        "subset_size": args.subset_size,
        "val_subset_size": args.val_subset_size,
        "seed": args.seed,
        "verbose": args.verbose
    }

    # Initialize wandb
    if config["use_wandb"]:
        # Use custom run name if provided, otherwise generate one
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        else:
            run_name = f"paligemma-adapter-{config['adapter_size']}-lr{config['learning_rate']:.0e}-optimized"

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ImageCaptioning"),
            entity=os.getenv("WANDB_ENTITY", "tuna-ozturk1283"),
            name=run_name,
            config=config
        )

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load model and processor
    print("Loading model, tokenizer, and processor...")
    processor = AutoProcessor.from_pretrained(config["model_path"], token=args.token)

    # Load model with memory optimizations
    print("Loading model with memory optimizations...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        config["model_path"],
        token=args.token,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    # Move model to device
    model = model.to(device)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Apply LoRA adapters to the model
    print("Applying LoRA adapters to the model...")
    lora_config = get_lora_config(
        adapter_size=config["adapter_size"],
        adapter_dropout=config["adapter_dropout"],
        target_modules=config["adapter_target_modules"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataloaders
    print("Creating datasets...")
    train_dataloader, val_dataloader, _ = create_dataloaders(
        data_dir=config["data_dir"],
        processor=processor,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        prompt_template=config["prompt_template"],
        subset_size=config["subset_size"],
        val_subset_size=config["val_subset_size"]
    )

    print(f"Using {config['subset_size']} train examples")
    print(f"Using {config['val_subset_size']} val examples")

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

    # Train the model
    print("Starting training...")
    print(f"Training on {len(train_dataloader)} batches with batch size {config['batch_size']}")
    print(f"Validation on {len(val_dataloader)} batches with batch size {config['batch_size']}")
    print(f"Total training steps per epoch: {len(train_dataloader)}")
    print(f"Total validation steps per epoch: {len(val_dataloader)}")
    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch + 1}/{config['num_epochs']} starting...")

        # Train
        print(f"Starting training for epoch {epoch + 1}...")
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config
        )

        # Calculate perplexity
        train_perplexity = torch.exp(torch.tensor(train_loss)).item()
        print(f"Epoch {epoch + 1} training completed. Train Loss={train_loss:.4f}, Train PPL={train_perplexity:.2f}")

        # Evaluate
        print(f"Starting evaluation for epoch {epoch + 1}...")
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            device=device
        )

        # Calculate perplexity
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()

        # Print results
        print(f"Epoch {epoch + 1} completed: Train Loss={train_loss:.4f}, Train PPL={train_perplexity:.2f}; "
              f"Val Loss={val_loss:.4f}, Val PPL={val_perplexity:.2f}")

        if torch.cuda.is_available():
            print(f"GPU Memory after epoch {epoch + 1}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB / {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        # Log to wandb
        if config["use_wandb"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity
            })

        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")

            # Save the model
            output_dir = os.path.join(config["output_dir"], f"adapter-{config['adapter_size']}-best")
            model.save_pretrained(output_dir)

    # Save the final model
    print("Saving final model...")
    output_dir = os.path.join(config["output_dir"], f"adapter-{config['adapter_size']}-final")
    model.save_pretrained(output_dir)

    # Finish wandb
    if config["use_wandb"]:
        wandb.finish()

    print("Training complete!")

if __name__ == "__main__":
    main()
