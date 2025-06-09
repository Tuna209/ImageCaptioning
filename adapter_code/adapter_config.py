#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA adapter configuration for PaliGemma fine-tuning.
"""

from peft import LoraConfig

def get_lora_config(adapter_size=8, adapter_dropout=0.05, target_modules=None):
    """
    Get LoRA configuration for PaliGemma model.
    
    Args:
        adapter_size (int): Rank of the adapter (default: 8)
        adapter_dropout (float): Dropout rate for adapter layers (default: 0.05)
        target_modules (list): List of target modules to apply LoRA to
    
    Returns:
        LoraConfig: LoRA configuration object
    """
    
    if target_modules is None:
        # Default target modules for PaliGemma
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
        r=adapter_size,
        lora_alpha=adapter_size * 2,  # Common practice: alpha = 2 * rank
        target_modules=target_modules,
        lora_dropout=adapter_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return lora_config
