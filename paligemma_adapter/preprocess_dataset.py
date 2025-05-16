#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to preprocess the RISCM dataset.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import shutil
from PIL import Image


def check_images(data_dir, df):
    """Check if all images exist and are valid."""
    missing_images = []
    corrupt_images = []

    for image_name in tqdm(df["image_name"], desc="Checking images"):
        image_path = os.path.join(data_dir, "images", image_name)

        # Check if image exists
        if not os.path.exists(image_path):
            missing_images.append(image_name)
            continue

        # Check if image is valid
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            corrupt_images.append(image_name)

    return missing_images, corrupt_images


def clean_captions(df):
    """Clean the captions."""
    # Remove leading/trailing whitespace
    df["caption"] = df["caption"].str.strip()

    # Remove duplicate captions for the same image
    df = df.drop_duplicates(subset=["image_name", "caption"])

    # Remove rows with empty captions
    df = df[df["caption"].str.len() > 0]

    return df


def split_dataset(df, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split the dataset into train, validation, and test sets."""
    # Check if the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Get unique image names
    unique_images = df["image_name"].unique()

    # Shuffle the images
    import numpy as np
    np.random.seed(seed)
    np.random.shuffle(unique_images)

    # Split the images
    train_size = int(len(unique_images) * train_ratio)
    val_size = int(len(unique_images) * val_ratio)

    train_images = unique_images[:train_size]
    val_images = unique_images[train_size:train_size + val_size]
    test_images = unique_images[train_size + val_size:]

    # Create the splits
    train_df = df[df["image_name"].isin(train_images)]
    val_df = df[df["image_name"].isin(val_images)]
    test_df = df[df["image_name"].isin(test_images)]

    # Save the splits
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Train set: {len(train_df)} examples ({len(train_images)} unique images)")
    print(f"Validation set: {len(val_df)} examples ({len(val_images)} unique images)")
    print(f"Test set: {len(test_df)} examples ({len(test_images)} unique images)")

    return train_df, val_df, test_df


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="RISCM", help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="RISCM_processed", help="Directory to save the processed dataset")
    parser.add_argument("--captions_file", type=str, default="captions.csv", help="Name of the captions file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check if train.csv, val.csv, and test.csv already exist
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    test_path = os.path.join(args.data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Found existing train, validation, and test splits.")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        print(f"Train set: {len(train_df)} examples")
        print(f"Validation set: {len(val_df)} examples")
        print(f"Test set: {len(test_df)} examples")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Copy the CSV files
        shutil.copy2(train_path, os.path.join(args.output_dir, "train.csv"))
        shutil.copy2(val_path, os.path.join(args.output_dir, "val.csv"))
        shutil.copy2(test_path, os.path.join(args.output_dir, "test.csv"))

        # Create images directory and copy images
        images_dir = os.path.join(args.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Get all unique image names
        all_images = set(train_df["image_name"]).union(set(val_df["image_name"])).union(set(test_df["image_name"]))

        # Copy images
        print("Copying images...")
        for image_name in tqdm(all_images):
            src_path = os.path.join(args.data_dir, "images", image_name)
            dst_path = os.path.join(images_dir, image_name)

            if os.path.exists(src_path) and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

        print(f"Preprocessing complete. Processed dataset saved to {args.output_dir}")
        return

    # If no existing splits, load the captions
    captions_path = os.path.join(args.data_dir, args.captions_file)
    df = pd.read_csv(captions_path)

    print(f"Loaded {len(df)} captions for {df['image_name'].nunique()} unique images")

    # Clean the captions
    print("Cleaning captions...")
    df = clean_captions(df)

    # Check images
    print("Checking images...")
    missing_images, corrupt_images = check_images(args.data_dir, df)

    print(f"Found {len(missing_images)} missing images and {len(corrupt_images)} corrupt images")

    # Remove rows with missing or corrupt images
    df = df[~df["image_name"].isin(missing_images + corrupt_images)]

    # Split the dataset
    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df, args.output_dir, seed=args.seed)

    # Create images directory
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Copy images
    print("Copying images...")
    all_images = set(train_df["image_name"]).union(set(val_df["image_name"])).union(set(test_df["image_name"]))

    for image_name in tqdm(all_images):
        src_path = os.path.join(args.data_dir, "images", image_name)
        dst_path = os.path.join(images_dir, image_name)

        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

    print(f"Preprocessing complete. Processed dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()
