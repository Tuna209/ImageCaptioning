#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for adapter-based fine-tuning of PaliGemma.

This module provides configuration utilities for parameter-efficient fine-tuning
of the PaliGemma model using LoRA (Low-Rank Adaptation) adapters. LoRA is a technique
that adds small trainable matrices to the model's attention layers, allowing for
efficient adaptation to new tasks with minimal additional parameters.

Design subjects to improve:
1. Add support for other adapter methods like IA³, Prefix Tuning, or Prompt Tuning
2. Implement configuration validation to ensure parameters are within valid ranges
3. Add presets for different hardware configurations (high-end GPU, low-end GPU, CPU)
4. Support for mixed precision training configuration
"""

from peft import LoraConfig, TaskType


def get_lora_config(adapter_size=8, adapter_dropout=0.05, target_modules=None):
    """Get the LoRA configuration for adapter-based fine-tuning.

    Args:
        adapter_size (int): Size of the adapter (LoRA rank).
        adapter_dropout (float): Dropout probability for the adapter.
        target_modules (list): List of modules to apply LoRA to.

    Returns:
        LoraConfig: Configuration for LoRA.
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=adapter_size,
        lora_alpha=adapter_size * 2,
        lora_dropout=adapter_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    return lora_config


def get_default_config():
    """Get the default configuration for adapter-based fine-tuning.

    Returns:
        dict: Default configuration.
    """
    config = {
        "model_path": "google/paligemma-3b-mix-224",
        "data_dir": "RISCM",
        "max_length": 128,
        "prompt_template": "Describe the remote sensing image in detail.",
        "batch_size": 4,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "adapter_size": 8,
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
        "output_dir": "paligemma_adapter/outputs",
        "subset_size": 1000,
        "val_subset_size": 100,
        "seed": 42
    }

    return config
