#!/usr/bin/env python3
"""
Preprocessing script for the RISC dataset.

This script processes the RISC dataset for image captioning:
1. Checks for missing images
2. Selects one random caption per image
3. Creates train/val/test splits based on the original split
4. Saves the processed data to CSV files
"""

import os
import pandas as pd
import random
import argparse
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess the RISC dataset for image captioning")

    # Data arguments
    parser.add_argument("--captions_path", type=str, default="RISCM/captions.csv", help="Path to captions.csv")
    parser.add_argument("--images_path", type=str, default="RISCM/resized", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="processed_dataset", help="Output directory")

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    # Parse arguments
    args = parse_args()

    # Create output directory for CSV files
    os.makedirs(args.output_dir, exist_ok=True)

    # Load captions
    print("Loading captions...")
    captions_df = pd.read_csv(args.captions_path)

    # Check for missing images
    print("Checking for missing images...")
    all_images = set(os.listdir(args.images_path))
    missing_images = []

    for image_name in tqdm(captions_df['image'].unique()):
        if image_name not in all_images:
            missing_images.append(image_name)

    print(f"Found {len(missing_images)} missing images")

    # Remove rows with missing images
    if missing_images:
        captions_df = captions_df[~captions_df['image'].isin(missing_images)]
        print(f"Removed {len(missing_images)} rows with missing images")

    # Process the dataframe to select one random caption per image
    print("Processing dataframe to select one random caption per image...")
    processed_rows = []

    for _, row in tqdm(captions_df.iterrows(), total=len(captions_df)):
        image_name = row['image']
        source = row['source']
        split = row['split']

        # Create full absolute image path to ensure it works from anywhere
        image_path = os.path.abspath(os.path.join(args.images_path, image_name))

        # Collect all valid captions for this image
        valid_captions = []
        for i in range(1, 6):
            caption_key = f'caption_{i}'
            if caption_key in row and pd.notna(row[caption_key]) and row[caption_key].strip():
                valid_captions.append(row[caption_key].strip())

        # If there are valid captions, randomly select one
        if valid_captions:
            # Randomly select one caption
            selected_caption = random.choice(valid_captions)

            # Add the row with the selected caption
            processed_rows.append({
                'source': source,
                'split': split,
                'image': image_name,
                'image_path': image_path,  # Add full image path
                'caption': selected_caption
            })

    processed_df = pd.DataFrame(processed_rows)
    print(f"Processed {len(processed_df)} images, each with one randomly selected caption")

    # Split the data
    train_df = processed_df[processed_df['split'] == 'train']
    val_df = processed_df[processed_df['split'] == 'val']
    test_df = processed_df[processed_df['split'] == 'test']

    # Shuffle the data
    print("Shuffling the data...")
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Train: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")

    # Save the split dataframes with image paths
    print("Saving CSV files with image paths...")
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    # Save the combined dataframe for reference
    processed_df.to_csv(os.path.join(args.output_dir, "combined.csv"), index=False)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(processed_df['image'].unique())}")
    print(f"Total captions: {len(processed_df)}")
    print(f"Train split: {len(train_df)} captions, {len(train_df['image'].unique())} images")
    print(f"Validation split: {len(val_df)} captions, {len(val_df['image'].unique())} images")
    print(f"Test split: {len(test_df)} captions, {len(test_df['image'].unique())} images")

    print("\nData preprocessing complete!")
    print(f"CSV files saved to {args.output_dir}")


if __name__ == "__main__":
    main()