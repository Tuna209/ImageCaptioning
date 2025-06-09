#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for image captioning.
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from collections import Counter
import re

# Download NLTK data
nltk.download('punkt', quiet=True)

def calculate_bleu_scores(references, hypotheses):
    """
    Calculate BLEU scores for a list of reference and hypothesis pairs.
    
    Args:
        references (list): List of reference captions
        hypotheses (list): List of generated captions
    
    Returns:
        dict: Dictionary containing BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    smoothing = SmoothingFunction().method1
    
    for ref, hyp in zip(references, hypotheses):
        # Tokenize
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        
        # Calculate BLEU scores
        bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        
        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)
    
    return {
        'bleu_1': np.mean(bleu_1_scores),
        'bleu_2': np.mean(bleu_2_scores),
        'bleu_3': np.mean(bleu_3_scores),
        'bleu_4': np.mean(bleu_4_scores),
        'bleu_1_scores': bleu_1_scores,
        'bleu_2_scores': bleu_2_scores,
        'bleu_3_scores': bleu_3_scores,
        'bleu_4_scores': bleu_4_scores
    }

def calculate_rouge_scores(references, hypotheses):
    """
    Calculate ROUGE scores for a list of reference and hypothesis pairs.
    
    Args:
        references (list): List of reference captions
        hypotheses (list): List of generated captions
    
    Returns:
        dict: Dictionary containing ROUGE-L scores
    """
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge_l': np.mean(rouge_l_scores),
        'rouge_l_scores': rouge_l_scores
    }

def calculate_domain_relevance(hypotheses, domain_keywords=None):
    """
    Calculate domain relevance score based on remote sensing keywords.
    
    Args:
        hypotheses (list): List of generated captions
        domain_keywords (list): List of domain-specific keywords
    
    Returns:
        dict: Dictionary containing domain relevance scores
    """
    
    if domain_keywords is None:
        # Default remote sensing keywords
        domain_keywords = [
            'building', 'buildings', 'road', 'roads', 'tree', 'trees', 'forest',
            'water', 'river', 'lake', 'field', 'fields', 'agricultural', 'urban',
            'residential', 'commercial', 'industrial', 'vegetation', 'land',
            'area', 'region', 'landscape', 'terrain', 'satellite', 'aerial',
            'remote', 'sensing', 'infrastructure', 'development', 'green',
            'dense', 'sparse', 'scattered', 'clustered', 'transportation',
            'highway', 'street', 'avenue', 'bridge', 'parking', 'lot',
            'farmland', 'cropland', 'pasture', 'meadow', 'grassland'
        ]
    
    domain_scores = []
    
    for hyp in hypotheses:
        # Tokenize and clean
        tokens = re.findall(r'\b\w+\b', hyp.lower())
        
        # Count domain keywords
        domain_count = sum(1 for token in tokens if token in domain_keywords)
        
        # Calculate relevance score (percentage of domain keywords)
        relevance_score = domain_count / len(tokens) if len(tokens) > 0 else 0
        domain_scores.append(relevance_score)
    
    return {
        'domain_relevance': np.mean(domain_scores),
        'domain_scores': domain_scores
    }

def calculate_vocabulary_diversity(hypotheses):
    """
    Calculate vocabulary diversity metrics.
    
    Args:
        hypotheses (list): List of generated captions
    
    Returns:
        dict: Dictionary containing vocabulary diversity metrics
    """
    
    all_tokens = []
    for hyp in hypotheses:
        tokens = re.findall(r'\b\w+\b', hyp.lower())
        all_tokens.extend(tokens)
    
    unique_tokens = set(all_tokens)
    total_tokens = len(all_tokens)
    
    # Type-Token Ratio (TTR)
    ttr = len(unique_tokens) / total_tokens if total_tokens > 0 else 0
    
    # Most common words
    word_counts = Counter(all_tokens)
    most_common = word_counts.most_common(10)
    
    return {
        'vocabulary_size': len(unique_tokens),
        'total_tokens': total_tokens,
        'type_token_ratio': ttr,
        'most_common_words': most_common
    }

def calculate_comprehensive_metrics(references, hypotheses):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        references (list): List of reference captions
        hypotheses (list): List of generated captions
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    
    print("📊 Calculating BLEU scores...")
    bleu_metrics = calculate_bleu_scores(references, hypotheses)
    
    print("📊 Calculating ROUGE scores...")
    rouge_metrics = calculate_rouge_scores(references, hypotheses)
    
    print("📊 Calculating domain relevance...")
    domain_metrics = calculate_domain_relevance(hypotheses)
    
    print("📊 Calculating vocabulary diversity...")
    vocab_metrics = calculate_vocabulary_diversity(hypotheses)
    
    # Combine all metrics
    comprehensive_metrics = {
        'mean_bleu_1': bleu_metrics['bleu_1'],
        'mean_bleu_2': bleu_metrics['bleu_2'],
        'mean_bleu_3': bleu_metrics['bleu_3'],
        'mean_bleu_4': bleu_metrics['bleu_4'],
        'mean_rouge_l': rouge_metrics['rouge_l'],
        'mean_domain_relevance': domain_metrics['domain_relevance'],
        'vocabulary_size': vocab_metrics['vocabulary_size'],
        'type_token_ratio': vocab_metrics['type_token_ratio'],
        'total_samples': len(references)
    }
    
    # Add individual scores for detailed analysis
    comprehensive_metrics.update({
        'individual_bleu_1': bleu_metrics['bleu_1_scores'],
        'individual_rouge_l': rouge_metrics['rouge_l_scores'],
        'individual_domain_relevance': domain_metrics['domain_scores'],
        'most_common_words': vocab_metrics['most_common_words']
    })
    
    return comprehensive_metrics

def print_metrics_summary(metrics):
    """Print a summary of evaluation metrics."""
    
    print("\n🎯 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 60)
    print(f"📊 Total Samples: {metrics['total_samples']}")
    print(f"📈 BLEU-1: {metrics['mean_bleu_1']:.4f}")
    print(f"📈 BLEU-2: {metrics['mean_bleu_2']:.4f}")
    print(f"📈 BLEU-3: {metrics['mean_bleu_3']:.4f}")
    print(f"📈 BLEU-4: {metrics['mean_bleu_4']:.4f}")
    print(f"📈 ROUGE-L: {metrics['mean_rouge_l']:.4f}")
    print(f"🎯 Domain Relevance: {metrics['mean_domain_relevance']:.4f}")
    print(f"📝 Vocabulary Size: {metrics['vocabulary_size']}")
    print(f"📝 Type-Token Ratio: {metrics['type_token_ratio']:.4f}")
    
    print(f"\n📋 Most Common Words:")
    for word, count in metrics['most_common_words'][:5]:
        print(f"   {word}: {count}")
    
    print("=" * 60)
