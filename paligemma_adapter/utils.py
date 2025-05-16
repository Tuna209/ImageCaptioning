#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for PaliGemma model.
"""

import os
import torch
from typing import Tuple
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    AutoTokenizer
)
from huggingface_hub import login


def load_model(model_path: str, device: str, token: str = None, use_flash_attention: bool = False) -> Tuple[PaliGemmaForConditionalGeneration, AutoProcessor]:
    """Load the PaliGemma model and processor.
    
    Args:
        model_path (str): Path to the model or model ID on Hugging Face.
        device (str): Device to load the model on ('cpu', 'cuda', 'cuda:0', etc.).
        token (str, optional): Hugging Face token for accessing gated models.
        use_flash_attention (bool, optional): Whether to use flash attention for faster inference.
        
    Returns:
        tuple: (model, processor)
    """
    # Login with token if provided
    if token:
        login(token=token)
    
    print(f"Loading model and processor from {model_path}...")
    
    # Load the processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        token=token
    )
    
    # Load the model with memory optimizations
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        token=token,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attention_2=use_flash_attention and device != "cpu"
    )
    
    # Move model to device
    model = model.to(device)
    
    return model, processor


def get_device() -> str:
    """Get the device to use for training/inference.
    
    Returns:
        str: Device name ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def save_json(data, file_path):
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_path (str): Path to the JSON file.
    """
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save data to JSON file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(file_path):
    """Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Data from the JSON file.
    """
    import json
    
    # Load data from JSON file
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data


def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
