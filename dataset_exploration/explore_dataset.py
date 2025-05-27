#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to explore and analyze the RISC dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('punkt', quiet=True)

def analyze_splits(data_dir):
    """Analyze the dataset splits."""
    # Load the data
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Print statistics
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} examples")

    # Check for duplicates
    train_images = set(train_df["image_name"])
    val_images = set(val_df["image_name"])
    test_images = set(test_df["image_name"])

    train_val_overlap = train_images.intersection(val_images)
    train_test_overlap = train_images.intersection(test_images)
    val_test_overlap = val_images.intersection(test_images)

    print(f"Train-val overlap: {len(train_val_overlap)} images")
    print(f"Train-test overlap: {len(train_test_overlap)} images")
    print(f"Val-test overlap: {len(val_test_overlap)} images")

    return train_df, val_df, test_df

def analyze_captions(train_df, val_df, test_df):
    """Analyze the captions."""
    # Combine all captions
    all_captions = pd.concat([train_df["caption"], val_df["caption"], test_df["caption"]])

    # Calculate caption lengths
    caption_lengths = all_captions.str.split().apply(len)

    # Print statistics
    print(f"Total captions: {len(all_captions)}")
    print(f"Average caption length: {caption_lengths.mean():.2f} words")
    print(f"Minimum caption length: {caption_lengths.min()} words")
    print(f"Maximum caption length: {caption_lengths.max()} words")

    # Plot caption length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(caption_lengths, bins=30, alpha=0.7)
    plt.title("Caption Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("dataset_exploration/caption_length_distribution.png")
    plt.close()

    # Analyze vocabulary
    all_words = []
    for caption in all_captions:
        # Simple word tokenization by splitting on whitespace
        all_words.extend(caption.lower().split())

    vocab = Counter(all_words)

    print(f"Vocabulary size: {len(vocab)} unique words")
    print(f"Most common words: {vocab.most_common(10)}")

    return caption_lengths, vocab

def analyze_images(data_dir, train_df, sample_size=100):
    """Analyze the images."""
    # Get a sample of images
    sample_images = train_df["image_name"].sample(sample_size).tolist()

    # Analyze image dimensions and file sizes
    dimensions = []
    file_sizes = []

    for image_name in tqdm(sample_images, desc="Analyzing images"):
        image_path = os.path.join(data_dir, "images", image_name)

        try:
            # Get file size
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_sizes.append(file_size)

            # Get dimensions
            with Image.open(image_path) as img:
                dimensions.append(img.size)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    # Convert to numpy arrays
    dimensions = np.array(dimensions)
    file_sizes = np.array(file_sizes)

    # Print statistics
    print(f"Average image dimensions: {dimensions[:, 0].mean():.2f} x {dimensions[:, 1].mean():.2f} pixels")
    print(f"Average file size: {file_sizes.mean():.2f} KB")

    # Plot file size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(file_sizes, bins=30, alpha=0.7)
    plt.title("Image File Size Distribution")
    plt.xlabel("File Size (KB)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("dataset_exploration/image_size_distribution.png")
    plt.close()

    return dimensions, file_sizes

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="RISCM", help="Directory containing the dataset")
    args = parser.parse_args()

    # Create output directory
    os.makedirs("dataset_exploration", exist_ok=True)

    # Analyze dataset splits
    print("Analyzing dataset splits...")
    train_df, val_df, test_df = analyze_splits(args.data_dir)

    # Analyze captions
    print("\nAnalyzing captions...")
    caption_lengths, vocab = analyze_captions(train_df, val_df, test_df)

    # Analyze images
    print("\nAnalyzing images...")
    dimensions, file_sizes = analyze_images(args.data_dir, train_df)

    print("\nAnalysis complete. Results saved to dataset_exploration/")

if __name__ == "__main__":
    main()
