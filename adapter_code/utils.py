#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for PaliGemma adapter fine-tuning.
"""

import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    # We'll just return the loss for now
    # In a real scenario, you'd compute BLEU, ROUGE, etc.
    return {}

def generate_and_print_sample(model, processor, dataset, num_samples=3, max_length=128, min_length=20, num_beams=4):
    """Generate and print sample captions."""
    # Set model to evaluation mode
    model.eval()

    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        # Get sample
        sample = dataset[idx]

        # Get image
        pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        # Decode caption
        generated_caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Get ground truth caption
        ground_truth = processor.batch_decode(sample["labels"].unsqueeze(0), skip_special_tokens=True)[0]

        # Print results
        print(f"Generated: {generated_caption}")
        print(f"Ground truth: {ground_truth}")
        print()



def visualize_image_with_caption(image_path, caption):
    """Visualize image with caption."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption, fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()
