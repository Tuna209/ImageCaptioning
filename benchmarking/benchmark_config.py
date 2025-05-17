#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for benchmarking PaliGemma adapter-based fine-tuning.
This file defines the hyperparameter space for benchmarking experiments.
"""

# Adapter architecture configurations
ADAPTER_CONFIGS = [
    {
        "name": "tiny_adapter",
        "adapter_size": 4,
        "adapter_dropout": 0.1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    },
    {
        "name": "small_adapter",
        "adapter_size": 8,
        "adapter_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj"
        ]
    },
    {
        "name": "medium_adapter",
        "adapter_size": 16,
        "adapter_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    },
    {
        "name": "large_adapter",
        "adapter_size": 32,
        "adapter_dropout": 0.1,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    }
]

# Training dynamics configurations
TRAINING_CONFIGS = [
    {
        "name": "conservative",
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    },
    {
        "name": "balanced",
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    },
    {
        "name": "aggressive",
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0
    },
    {
        "name": "extended",
        "learning_rate": 5e-5,
        "num_epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    }
]

# Prompt template variations
PROMPT_TEMPLATES = [
    {
        "name": "basic",
        "template": "Describe the remote sensing image in detail."
    },
    {
        "name": "detailed",
        "template": "Provide a detailed description of this remote sensing image, including visible features, land use, and geographical elements."
    },
    {
        "name": "analytical",
        "template": "Analyze this remote sensing image and describe the key features, patterns, and potential applications."
    },
    {
        "name": "structured",
        "template": "Describe this remote sensing image in terms of: 1) Land cover types, 2) Human structures, 3) Natural features, 4) Notable patterns."
    }
]

# Data processing configurations
DATA_CONFIGS = [
    {
        "name": "baseline",
        "subset_size": 1000,
        "val_subset_size": 100,
        "max_length": 128,
        "batch_size": 1,
        "gradient_accumulation_steps": 16
    },
    {
        "name": "small_data",
        "subset_size": 500,
        "val_subset_size": 50,
        "max_length": 96,
        "batch_size": 1,
        "gradient_accumulation_steps": 8
    },
    {
        "name": "large_data",
        "subset_size": 2000,
        "val_subset_size": 200,
        "max_length": 128,
        "batch_size": 1,
        "gradient_accumulation_steps": 16
    }
]

# Benchmark experiment combinations
# These are the specific combinations we want to test
BENCHMARK_EXPERIMENTS = [
    # Baseline experiment (reference point)
    {
        "name": "baseline",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    
    # Adapter size variations
    {
        "name": "tiny_adapter_test",
        "adapter_config": "tiny_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    {
        "name": "medium_adapter_test",
        "adapter_config": "medium_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    {
        "name": "large_adapter_test",
        "adapter_config": "large_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    
    # Learning rate variations
    {
        "name": "conservative_lr_test",
        "adapter_config": "small_adapter",
        "training_config": "conservative",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    {
        "name": "aggressive_lr_test",
        "adapter_config": "small_adapter",
        "training_config": "aggressive",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    
    # Training duration test
    {
        "name": "extended_training_test",
        "adapter_config": "small_adapter",
        "training_config": "extended",
        "prompt_template": "basic",
        "data_config": "baseline"
    },
    
    # Prompt template variations
    {
        "name": "detailed_prompt_test",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "detailed",
        "data_config": "baseline"
    },
    {
        "name": "analytical_prompt_test",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "analytical",
        "data_config": "baseline"
    },
    {
        "name": "structured_prompt_test",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "structured",
        "data_config": "baseline"
    },
    
    # Data size variations
    {
        "name": "small_data_test",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "small_data"
    },
    {
        "name": "large_data_test",
        "adapter_config": "small_adapter",
        "training_config": "balanced",
        "prompt_template": "basic",
        "data_config": "large_data"
    },
    
    # Combined optimizations (based on expected best performers)
    {
        "name": "optimized_combination_1",
        "adapter_config": "medium_adapter",
        "training_config": "balanced",
        "prompt_template": "detailed",
        "data_config": "baseline"
    },
    {
        "name": "optimized_combination_2",
        "adapter_config": "small_adapter",
        "training_config": "extended",
        "prompt_template": "analytical",
        "data_config": "baseline"
    }
]

def get_experiment_config(experiment_name):
    """Get the full configuration for a named experiment."""
    # Find the experiment
    experiment = None
    for exp in BENCHMARK_EXPERIMENTS:
        if exp["name"] == experiment_name:
            experiment = exp
            break
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Get the individual configurations
    adapter_config = None
    for config in ADAPTER_CONFIGS:
        if config["name"] == experiment["adapter_config"]:
            adapter_config = config
            break
    
    training_config = None
    for config in TRAINING_CONFIGS:
        if config["name"] == experiment["training_config"]:
            training_config = config
            break
    
    prompt_template = None
    for config in PROMPT_TEMPLATES:
        if config["name"] == experiment["prompt_template"]:
            prompt_template = config
            break
    
    data_config = None
    for config in DATA_CONFIGS:
        if config["name"] == experiment["data_config"]:
            data_config = config
            break
    
    # Combine all configurations
    full_config = {
        "experiment_name": experiment["name"],
        "adapter_size": adapter_config["adapter_size"],
        "adapter_dropout": adapter_config["adapter_dropout"],
        "target_modules": adapter_config["target_modules"],
        "learning_rate": training_config["learning_rate"],
        "num_epochs": training_config["num_epochs"],
        "weight_decay": training_config["weight_decay"],
        "warmup_ratio": training_config["warmup_ratio"],
        "max_grad_norm": training_config["max_grad_norm"],
        "prompt_template": prompt_template["template"],
        "subset_size": data_config["subset_size"],
        "val_subset_size": data_config["val_subset_size"],
        "max_length": data_config["max_length"],
        "batch_size": data_config["batch_size"],
        "gradient_accumulation_steps": data_config["gradient_accumulation_steps"]
    }
    
    return full_config
