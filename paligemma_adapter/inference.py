#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for PaliGemma model with LoRA adapter-based fine-tuning.
"""

import os
import argparse
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


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


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="Hugging Face token")
    parser.add_argument("--adapter_path", type=str, help="Path to the adapter model")
    parser.add_argument("--model_path", type=str, default="google/paligemma-3b-mix-224", help="Path to the base model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--prompt_template", type=str, default="Describe the remote sensing image in detail.", help="Prompt template")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("Using CPU")
    
    print(f"Using device: {device}")
    
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

    # Load adapter if specified
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print("Adapter loaded successfully!")

    # Generate caption
    print("Generating caption...")
    caption = generate_caption(
        model=model,
        processor=processor,
        image_path=args.image_path,
        device=device,
        prompt_template=args.prompt_template
    )
    
    # Print caption
    print("\nGenerated caption:")
    print(caption)


if __name__ == "__main__":
    main()
