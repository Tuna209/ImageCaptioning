#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for PaliGemma adapter fine-tuning.
"""

import os
import argparse
import yaml
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapter_code.dataset import RISCDataset
import torch
import evaluate

def evaluate_model(model, processor, dataset, metrics_list=None, max_length=128, min_length=20, num_beams=4):
    """Evaluate model on dataset."""
    # Set model to evaluation mode
    model.eval()

    # Initialize metrics
    if metrics_list is None:
        metrics_list = ["bleu", "rouge", "meteor"]

    metrics = {}
    for metric_name in metrics_list:
        metrics[metric_name] = evaluate.load(metric_name)

    # Generate predictions
    predictions = []
    references = []

    for idx in range(len(dataset)):
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

        # Add to predictions and references
        predictions.append(generated_caption)
        references.append(ground_truth)

    # Compute metrics
    results = {}
    for metric_name, metric in metrics.items():
        if metric_name == "bleu":
            results[metric_name] = metric.compute(predictions=predictions, references=[[r] for r in references])
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=references)

    return results

def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--data_dir", type=str, default="processed_dataset", help="Path to data directory")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load processor and model
    print(f"Loading model from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # Check if we have a base model or a full model
    if os.path.exists(os.path.join(args.model_dir, "adapter_model.bin")):
        print("Loading base model and adapter")
        base_model = AutoModelForVision2Seq.from_pretrained(
            config["model"]["model_id"],
            token=args.token,
            torch_dtype=torch.bfloat16 if config["training"]["mixed_precision"] else torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    else:
        print("Loading full model")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16 if config["training"]["mixed_precision"] else torch.float32,
        )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load test dataset
    print("Loading test dataset")
    test_dataset = RISCDataset(
        csv_file=os.path.join(args.data_dir, "test.csv"),
        img_dir=os.path.join(args.data_dir, "images"),
        processor=processor,
        max_samples=args.max_samples,
        input_prompt=config["prompt"]["input_prompt"]
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate model
    print("Evaluating model")
    results = evaluate_model(
        model=model,
        processor=processor,
        dataset=test_dataset,
        metrics_list=["bleu", "rouge", "meteor"],
        max_length=config["evaluation"]["generate_max_length"],
        min_length=config["evaluation"]["min_length"],
        num_beams=config["evaluation"]["num_beams"],
    )

    # Print results
    print("\nEvaluation results:")
    for metric_name, result in results.items():
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"{metric_name}_{k}: {v:.4f}")
        else:
            print(f"{metric_name}: {result:.4f}")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
