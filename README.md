# PaliGemma Fine-tuning for Remote Sensing Image Captioning

## Overview

This project demonstrates systematic fine-tuning of Google's PaliGemma vision-language model for remote sensing image captioning using the RISCM dataset. Through comprehensive hyperparameter optimization, we achieved **374% improvement** over the baseline model.

## Key Results

- **Best BLEU-4 Score**: 0.0289 (vs 0.0061 baseline)
- **Optimal Configuration**: LoRA Rank 8, Learning Rate 5e-5, 1000 samples
- **Reproducibility**: 90%+ consistency across different random seeds
- **Systematic Analysis**: 13+ configurations tested across multiple dimensions

## Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (8GB+ memory recommended)
- **HuggingFace account** - [Get token here](https://huggingface.co/settings/tokens)
- **Wandb account** - [Get API key here](https://wandb.ai/authorize) (optional but recommended)

## Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to project
cd your-project-directory

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file with your HuggingFace token and Wandb credentials

```

### 2. Data Preprocessing
```bash
# Preprocess the RISCM dataset
python preprocessing.py

# This creates:
# - processed_dataset/train.csv 
# - processed_dataset/val.csv 
# - processed_dataset/test.csv 
```

### 3. Train Optimal Model
```bash
# Train the best performing configuration
# Without HF_TOKEN in .env
python adapter_code/train.py --adapter_size 8 --learning_rate 5e-05 --batch_size 16 --gradient_accumulation_steps 4 --num_epochs 3 --data_dir processed_dataset --output_dir outputs/adapter-8-best --subset_size 1000 --val_subset_size 100 --verbose

# With HF_TOKEN in .env
python adapter_code/train.py --token $HF_TOKEN --adapter_size 8 --learning_rate 5e-05 --batch_size 16 --gradient_accumulation_steps 4 --num_epochs 3 --data_dir processed_dataset --output_dir outputs/adapter-8-best --use_wandb --subset_size 1000 --val_subset_size 100 --verbose
```

### 4. Evaluate Models
```bash
# Evaluate baseline (original PaliGemma)
python baseline_eval/baseline_evaluation.py \
    --token $HF_TOKEN \
    --test_csv processed_dataset/test.csv \
    --images_dir RISCM/resized \
    --max_samples 100 \
    --use_wandb

# Evaluate fine-tuned model
python evaluation/run_evaluation.py --mode finetuned --adapter_path outputs/adapter-8-best/adapter-8-best --test_csv processed_dataset/test.csv --images_dir RISCM/resized --max_samples 100
```

## Reproduce Full Systematic Study (Optional)

To reproduce our complete systematic study with all 13+ configurations:

```bash
# Train different LoRA ranks with optimal LR
python adapter_code/train.py --adapter_size 8 --learning_rate 5e-05 --subset_size 1000
python adapter_code/train.py --adapter_size 16 --learning_rate 5e-05 --subset_size 1000
python adapter_code/train.py --adapter_size 32 --learning_rate 5e-05 --subset_size 1000

# Train different learning rates with optimal rank
python adapter_code/train.py --adapter_size 8 --learning_rate 2e-05 --subset_size 1000
python adapter_code/train.py --adapter_size 8 --learning_rate 1e-04 --subset_size 1000
python adapter_code/train.py --adapter_size 8 --learning_rate 1e-05 --subset_size 1000

# Test dataset size impact (shows overfitting)
python adapter_code/train.py --adapter_size 8 --learning_rate 5e-05 --subset_size 3000

# Test reproducibility with different seeds
python adapter_code/train.py --adapter_size 8 --learning_rate 5e-05 --subset_size 1000 --seed 123

# Evaluate each trained model
python evaluation/run_evaluation.py --mode finetuned --adapter_path outputs/[model_path]
```

## Upload to HuggingFace (Optional)

You can manually upload your trained models to HuggingFace Hub for sharing:

1. Install HuggingFace Hub: `pip install huggingface_hub`
2. Login: `huggingface-cli login`
3. Upload your adapter files from `outputs/adapter-8-best/`

## Project Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git ignore rules
├── preprocessing.py                   # Dataset preprocessing
├── adapter_code/                     # Training code
│   ├── train.py                      # Main training script
│   ├── dataset.py                    # Dataset handling
│   └── adapter_config.py             # LoRA configuration
├── evaluation/                       # Evaluation framework
│   ├── run_evaluation.py             # Main evaluation script
│   ├── metrics.py                    # BLEU, ROUGE, domain metrics
│   └── evaluator.py                  # Evaluation logic
└── baseline_eval/                    # Baseline comparison
    └── baseline_evaluation.py        # Baseline model evaluation
```

**Note**: You'll need to add your own dataset in `RISCM/` directory with:
- `RISCM/captions.csv` (original captions)
- `RISCM/resized/` (resized images 224x224)

## Key Features

- **Systematic Hyperparameter Optimization**: Manual testing across LoRA ranks, learning rates, dataset sizes
- **Reproducible Results**: Seed-based reproducibility validation (90%+ consistency)
- **Baseline Comparison**: Direct comparison with original PaliGemma (374% improvement)
- **Professional Evaluation**: Industry-standard metrics (BLEU-4, ROUGE-L, domain relevance)
- **Experiment Tracking**: Full Wandb integration with 100 sample visualizations
- **Model Sharing**: HuggingFace integration for model distribution

## Expected Results

### Optimal Configuration (LoRA Rank 8, LR 5e-5, 1K samples)
- **BLEU-4**: 0.0289 (374% improvement over baseline)
- **BLEU-1**: 0.2296
- **ROUGE-L**: 0.2165
- **Domain Relevance**: 0.4210

### Baseline Performance
- **BLEU-4**: 0.0061
- **BLEU-1**: 0.0363
- **ROUGE-L**: 0.0568
- **Domain Relevance**: 0.0369

## Configuration Options

### Training Parameters
- `--adapter_size`: LoRA rank (8, 16, 32) - **8 recommended**
- `--learning_rate`: Learning rate (1e-6 to 1e-4) - **5e-5 recommended**
- `--subset_size`: Training samples (1000 recommended, avoid 3000+)
- `--batch_size`: Batch size (adjust based on GPU memory)
- `--num_epochs`: Training epochs (3 recommended)

### Evaluation Parameters
- `--max_samples`: Test samples for evaluation (100 recommended)
- `--use_wandb`: Enable Wandb logging (recommended)

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce `batch_size` or increase `gradient_accumulation_steps`
- **Missing images**: Ensure `RISCM/resized/` directory exists with images
- **Token errors**: Configure .env file with valid HuggingFace token
- **Poor performance**: Use recommended hyperparameters (Rank 8, LR 5e-5, 1K samples)
- **Missing dataset**: Download RISCM dataset and place in `RISCM/` directory

### Performance Tips
- **GPU Memory**: 8GB+ recommended for optimal batch sizes
- **Training Time**: ~30 minutes for optimal configuration on modern GPU
- **Dataset Size**: 1000 samples optimal, avoid 3000+ (causes overfitting)

## Scientific Reproducibility

This project emphasizes reproducible research:
1. **Fixed random seeds** for consistent results
2. **Systematic hyperparameter testing** across multiple dimensions
3. **Baseline comparison** with original model
4. **Multiple evaluation metrics** for comprehensive assessment
5. **Public model sharing** via HuggingFace
