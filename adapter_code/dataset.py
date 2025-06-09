#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset class for the RISCM dataset.
"""

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import random


class RISCMDataset(Dataset):
    """Dataset class for the RISCM dataset."""

    def __init__(self, data_dir, split, processor, max_length=128, prompt_template=""):
        """Initialize the dataset.

        Args:
            data_dir (str): Path to the RISCM dataset directory.
            split (str): Split to use ('train', 'val', or 'test').
            processor: Processor for the PaliGemma model.
            max_length (int): Maximum length of the input sequence.
            prompt_template (str): Template for the prompt.
        """
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
            # Handle different column names (image vs image_name)
            if 'image' in row:
                image_name = row['image']
            elif 'image_name' in row:
                image_name = row['image_name']
            else:
                raise ValueError("Neither 'image' nor 'image_name' column found in dataset")

            # Images are in RISCM/resized folder
            image_path = os.path.join("RISCM", "resized", image_name)
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

        # Process inputs - disable truncation to avoid token mismatch
        inputs = self.processor(
            text=text,
            images=image,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt"
        )

        # Get the actual length of the input_ids
        input_length = inputs["input_ids"].shape[1]

        # Process the caption as the target with the same length as input_ids
        labels = self.processor.tokenizer(
            caption,
            padding="max_length",
            max_length=input_length,  # Use the same length as input_ids
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        # Manually mask out the pad tokens in labels (for future compatibility)
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        # Set the labels
        inputs["labels"] = labels

        # Remove batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in inputs.items()}

        return inputs

def create_dataloaders(data_dir, processor, batch_size, max_length=128, prompt_template="",
                      subset_size=None, val_subset_size=None, num_workers=0):
    """Create dataloaders for the RISCM dataset.

    Args:
        data_dir (str): Path to the RISCM dataset directory.
        processor: Processor for the PaliGemma model.
        batch_size (int): Batch size.
        max_length (int): Maximum length of the input sequence.
        prompt_template (str): Template for the prompt.
        subset_size (int): Number of examples to use for training.
        val_subset_size (int): Number of examples to use for validation.
        num_workers (int): Number of workers for the dataloader.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset_full = RISCMDataset(
        data_dir=data_dir,
        split="train",
        processor=processor,
        max_length=max_length,
        prompt_template=prompt_template
    )

    val_dataset_full = RISCMDataset(
        data_dir=data_dir,
        split="val",
        processor=processor,
        max_length=max_length,
        prompt_template=prompt_template
    )

    test_dataset = RISCMDataset(
        data_dir=data_dir,
        split="test",
        processor=processor,
        max_length=max_length,
        prompt_template=prompt_template
    )

    # Create subsets for faster training if specified
    if subset_size is not None and subset_size < len(train_dataset_full):
        train_indices = random.sample(range(len(train_dataset_full)), subset_size)
        train_dataset = Subset(train_dataset_full, train_indices)
    else:
        train_dataset = train_dataset_full

    if val_subset_size is not None and val_subset_size < len(val_dataset_full):
        val_indices = random.sample(range(len(val_dataset_full)), val_subset_size)
        val_dataset = Subset(val_dataset_full, val_indices)
    else:
        val_dataset = val_dataset_full

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader
