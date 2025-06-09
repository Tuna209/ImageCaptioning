#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation utilities for PaliGemma image captioning.
"""

import os
import torch
import pandas as pd
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from tqdm import tqdm
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PaliGemmaEvaluator:
    """Evaluator for PaliGemma models."""
    
    def __init__(self, model_path="google/paligemma-3b-mix-224"):
        self.model_path = model_path
        self.processor = None
        self.model = None
    
    def load_baseline_model(self, token):
        """Load the baseline PaliGemma model (zero-shot)."""
        print(f"🔄 Loading baseline model: {self.model_path}")
        
        # Load processor and model
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_path,
            token=token
        )
        
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        
        print(f"✅ Baseline model loaded successfully")
        return self.processor, self.model
    
    def load_finetuned_model(self, adapter_path, token):
        """Load the fine-tuned PaliGemma model with LoRA adapters."""
        print(f"🔄 Loading base model: {self.model_path}")
        
        # Load base model
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_path,
            token=token
        )
        
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        
        print(f"🔄 Loading LoRA adapter: {adapter_path}")
        
        # Load and apply LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print(f"✅ Fine-tuned model loaded successfully")
        return self.processor, self.model
    
    def evaluate_on_test_set(self, test_csv_path, images_dir, prompt_template, max_samples=100):
        """Evaluate the model on test set."""
        
        print(f"📊 Evaluating model on test set...")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        
        # Limit samples for faster evaluation
        if len(test_df) > max_samples:
            test_df = test_df.head(max_samples)
            print(f"📝 Using {max_samples} samples for evaluation")
        
        results = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            try:
                # Load image (handle different column names)
                image_name = row.get('image', row.get('image_name', ''))
                image_path = os.path.join(images_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                
                # Prepare input
                prompt = prompt_template
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
                
                # Generate caption
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                # Decode generated text
                generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                if prompt in generated_text:
                    generated_caption = generated_text.replace(prompt, "").strip()
                else:
                    generated_caption = generated_text.strip()
                
                result = {
                    'image': image_name,
                    'ground_truth': row['caption'],
                    'generated_caption': generated_caption,
                    'prompt_used': prompt
                }
                
                results.append(result)
                
                # Print first few examples
                if idx < 3:
                    print(f"\n📸 Example {idx + 1}:")
                    print(f"   Image: {image_name}")
                    print(f"   Ground Truth: {row['caption'][:100]}...")
                    print(f"   Generated: {generated_caption[:100]}...")
            
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")
                continue
        
        return results
    
    def calculate_basic_metrics(self, results):
        """Calculate basic evaluation metrics."""
        
        print(f"\n📈 Calculating metrics...")
        
        # Basic statistics
        total_samples = len(results)
        successful_generations = len([r for r in results if r['generated_caption']])
        
        # Average caption lengths
        gt_lengths = [len(r['ground_truth'].split()) for r in results]
        gen_lengths = [len(r['generated_caption'].split()) for r in results if r['generated_caption']]
        
        avg_gt_length = sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0
        avg_gen_length = sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0
        
        metrics = {
            'total_samples': total_samples,
            'successful_generations': successful_generations,
            'success_rate': successful_generations / total_samples if total_samples > 0 else 0,
            'avg_ground_truth_length': avg_gt_length,
            'avg_generated_length': avg_gen_length
        }
        
        return metrics
    
    def save_results(self, results, metrics, output_file, model_name="PaliGemma"):
        """Save evaluation results."""
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Save metrics summary
        metrics_file = output_file.replace('.csv', '_metrics.json')
        metrics['model_name'] = model_name
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        print(f"💾 Metrics saved to: {metrics_file}")
        
        return metrics_file
    
    def print_results(self, metrics, results):
        """Print comprehensive results."""
        
        print(f"\n🎯 EVALUATION RESULTS")
        print("=" * 50)
        print(f"📊 Total Samples: {metrics['total_samples']}")
        print(f"✅ Successful Generations: {metrics['successful_generations']}")
        print(f"📈 Success Rate: {metrics['success_rate']:.2%}")
        print(f"📝 Avg Ground Truth Length: {metrics['avg_ground_truth_length']:.1f} words")
        print(f"📝 Avg Generated Length: {metrics['avg_generated_length']:.1f} words")
        
        print(f"\n📋 Sample Predictions:")
        print("-" * 50)
        
        for i, result in enumerate(results[:3]):
            print(f"\n🖼️  Example {i+1}:")
            print(f"   Image: {result['image']}")
            print(f"   Ground Truth: {result['ground_truth'][:150]}...")
            print(f"   Generated: {result['generated_caption'][:150]}...")
