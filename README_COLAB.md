# Running PaLI-Gemma Adapter Training in Google Colab

This README provides instructions for running the `file_to_use_colab.ipynb` notebook in Google Colab to train and evaluate PaLI-Gemma adapters for remote sensing image captioning.

## Prerequisites

1. **Google Account**: You need a Google account to access Google Colab and Google Drive.
2. **Hugging Face Account**: You need a Hugging Face account and API token to download the PaLI-Gemma model.
3. **GPU Runtime**: The notebook requires a GPU runtime, preferably an A100 GPU.

## Required Files and Directories

Before running the notebook, you need to upload the following files and directories to your Google Drive:

### Main Project Directories

- `paligemma_adapter/` - Core code for adapter-based fine-tuning
- `RISCM/` - Dataset (images and CSV files)
- `benchmarking/` - Benchmarking code (optional)
- `zero_shot_evaluation/` - Zero-shot evaluation scripts (optional)

### Core Python Scripts

The `paligemma_adapter/` directory should contain:

- `preprocess_dataset.py` - For preprocessing the dataset
- `analyze_dataset.py` - For analyzing the dataset
- `train.py` - For training the model with adapters
- `evaluate.py` - For evaluating the fine-tuned model
- `inference.py` - For running inference with the model

### Dataset Files

The `RISCM/` directory should contain:

- `images/` - Directory containing the image files
- `train.csv` - Training data annotations
- `val.csv` - Validation data annotations
- `test.csv` - Test data annotations

## Setup Instructions

1. **Upload Files to Google Drive**:
   - Create a folder in your Google Drive at the path: `/content/drive/My Drive/1 METU/7/DI725 Transformer/project/New folder/`
   - Upload all the required files and directories to this location

2. **Open the Notebook in Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `file_to_use_colab.ipynb` or open it from your Google Drive

3. **Select GPU Runtime**:
   - Click on "Runtime" in the menu
   - Select "Change runtime type"
   - Choose "GPU" as the hardware accelerator (preferably A100)
   - Click "Save"

## Running the Notebook

The notebook contains several cells that perform different tasks. Run them in order:

1. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Navigate to Project Directory**:
   ```python
   %cd "/content/drive/My Drive/1 METU/7/DI725 Transformer/project/New folder"
   ```

3. **Install Required Packages**:
   ```python
   !pip install transformers peft torch pandas numpy pillow tqdm nltk wandb
   !pip install accelerate bitsandbytes
   ```

4. **Preprocess Dataset**:
   ```python
   !python paligemma_adapter/preprocess_dataset.py --data_dir RISCM --output_dir RISCM_processed
   ```

5. **Analyze Dataset**:
   ```python
   !python paligemma_adapter/analyze_dataset.py --data_dir RISCM
   ```

6. **Train Model**:
   ```python
   # Get your Hugging Face token
   import getpass
   hf_token = getpass.getpass("Enter your Hugging Face token: ")

   # Run the training script with a small subset for testing
   !python paligemma_adapter/train.py \
     --token "$hf_token" \
     --adapter_size 8 \
     --learning_rate 1e-4 \
     --batch_size 1 \
     --gradient_accumulation_steps 16 \
     --num_epochs 1 \
     --subset_size 100 \
     --val_subset_size 20 \
     --data_dir RISCM_processed \
     --output_dir paligemma_adapter/outputs/adapter-8-test
   ```

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce `batch_size` and increase `gradient_accumulation_steps`
   - Reduce `subset_size` and `val_subset_size`
   - Use a smaller adapter size (e.g., 4 instead of 8)

2. **NLTK Resources Missing**:
   - If you encounter NLTK errors, run:
     ```python
     import nltk
     nltk.download('punkt')
     ```

3. **Slow Image Processing**:
   - The preprocessing step can be slow due to image copying. Be patient or reduce the dataset size.

4. **Hugging Face Token Issues**:
   - Ensure your token has the correct permissions
   - Check that the token is entered correctly

## Output Directories

The notebook will create several output directories:

- `RISCM_processed/` - Preprocessed dataset
- `paligemma_adapter/outputs/` - Trained adapter models
- `zero_shot_evaluation/results/` - Zero-shot evaluation results (if applicable)

## Weights & Biases Integration

The notebook includes Weights & Biases (wandb) integration for experiment tracking. If you use the `--use_wandb` flag, you can view training progress and results on your wandb dashboard.

## Additional Notes

- The notebook is configured to use a small subset of the data for quick testing. For full training, increase the `subset_size` and `num_epochs` parameters.
- Training with the full dataset will take several hours, even on an A100 GPU.
- The adapter approach significantly reduces the number of trainable parameters, making fine-tuning more efficient.

## References

- PaLI-Gemma: [Hugging Face Model Card](https://huggingface.co/google/paligemma-3b-mix-224)
- PEFT Library: [GitHub Repository](https://github.com/huggingface/peft)
- Weights & Biases: [Documentation](https://docs.wandb.ai/)
