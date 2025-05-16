#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick test script for PaliGemma model with LoRA adapter-based fine-tuning.
This script performs a minimal test to verify that the implementation works.
"""

import os
import argparse
import torch
import random
import numpy as np
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model
import time

from adapter_config import get_lora_config


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-224", help="Path to the model")
    parser.add_argument("--data_dir", type=str, default="RISCM_processed", help="Directory containing the dataset")
    parser.add_argument("--adapter_size", type=int, default=8, help="Size of the adapter (LoRA rank)")
    parser.add_argument("--output_dir", type=str, default="paligemma_adapter/outputs/test", help="Directory to save the model")
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and processor
    print("Loading model, tokenizer, and processor...")
    start_time = time.time()
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

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Apply LoRA adapters to the model
    print("Applying LoRA adapters to the model...")
    lora_config = get_lora_config(adapter_size=args.adapter_size)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    # Find a test image
    print("Finding a test image...")
    image_dir = os.path.join(args.data_dir, "images")
    image_files = os.listdir(image_dir)
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    test_image_path = os.path.join(image_dir, image_files[0])
    print(f"Using test image: {test_image_path}")

    # Load the image
    image = Image.open(test_image_path).convert("RGB")

    # Create a simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Perform a few training steps
    print("Performing a few training steps...")
    num_steps = 5

    for step in range(num_steps):
        start_time = time.time()

        # Process the image and text
        text = "<image> Describe the remote sensing image in detail."
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Add dummy labels for training
        inputs["labels"] = inputs["input_ids"].clone()

        # Forward pass
        outputs = model(**inputs)

        # Backward pass
        outputs.loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - start_time
        print(f"Step {step+1}/{num_steps} completed in {step_time:.2f} seconds, Loss: {outputs.loss.item()}")

    # Test generation
    print("Testing generation...")
    start_time = time.time()

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(
            **{k: v for k, v in inputs.items() if k != "labels"},
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )

    # Decode caption
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Generated caption: {caption}")

    # Save the model
    print("Saving model...")
    start_time = time.time()

    model.save_pretrained(args.output_dir)

    save_time = time.time() - start_time
    print(f"Model saved in {save_time:.2f} seconds")

    print("Quick test completed successfully!")


if __name__ == "__main__":
    main()
