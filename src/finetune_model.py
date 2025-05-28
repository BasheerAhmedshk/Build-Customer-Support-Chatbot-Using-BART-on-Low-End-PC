"""
Script to fine-tune the Facebook BART-base model on the Bitext customer support dataset.
This script handles model training, optimization for CPU, and saving the fine-tuned model.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset as HFDataset
import logging
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'bart-customer-support')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 512
BATCH_SIZE = 8  # Smaller batch size for CPU
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3  # Reduced for CPU training
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 4  # Increase for CPU training
FP16 = False  # Set to False for CPU training

class CustomerSupportDataset(Dataset):
    """Custom dataset for the customer support data."""
    
    def __init__(self, data_path, tokenizer, max_input_length, max_target_length):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input_text']
        target_text = self.data.iloc[idx]['target_text']
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_and_prepare_datasets(tokenizer):
    """Load and prepare datasets for training and evaluation."""
    logger.info("Loading and preparing datasets...")
    
    # Load datasets
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    
    # For quick testing, use the sample dataset
    sample_path = os.path.join(PROCESSED_DATA_DIR, 'sample.csv')
    use_sample = os.path.exists(sample_path) and os.environ.get('USE_SAMPLE', 'false').lower() == 'true'
    
    if use_sample:
        logger.info("Using sample dataset for quick testing")
        train_path = sample_path
    
    # Convert to HuggingFace datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Function to tokenize the data
    def tokenize_function(examples):
        inputs = tokenizer(
            examples['input_text'],
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        targets = tokenizer(
            examples['target_text'],
            max_length=MAX_TARGET_LENGTH,
            padding='max_length',
            truncation=True
        )
        
        # Replace padding token id with -100 so it's ignored in loss calculation
        labels = targets['input_ids'].copy()
        for i in range(len(labels)):
            labels[i] = [label if label != tokenizer.pad_token_id else -100 for label in labels[i]]
        
        inputs['labels'] = labels
        return inputs
    
    # Convert to HuggingFace datasets
    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing train dataset"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing validation dataset"
    )
    
    # Set format for pytorch
    train_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    val_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset, tokenizer):
    """Train the BART model on the customer support dataset."""
    logger.info(f"Loading model: {MODEL_NAME}")
    
    # Load model
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=2,
        fp16=FP16,  # Set to False for CPU training
        dataloader_num_workers=0,  # Set to 0 for CPU training
        disable_tqdm=False,
        report_to="none"  # Disable wandb, tensorboard, etc.
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training configuration
    config = {
        'model_name': MODEL_NAME,
        'max_input_length': MAX_INPUT_LENGTH,
        'max_target_length': MAX_TARGET_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_epochs': NUM_EPOCHS,
        'warmup_steps': WARMUP_STEPS,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'fp16': FP16,
        'training_time': training_time
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return model, trainer

def optimize_model_for_cpu(model_path):
    """Optimize the model for CPU inference."""
    logger.info("Optimizing model for CPU inference...")
    
    # Load the fine-tuned model
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    
    # Quantize the model for CPU
    # Note: Full quantization requires PyTorch 1.13+ and would use torch.quantization
    # For now, we'll use a simpler approach with torch.jit
    
    # Create a sample input for tracing
    sample_input = tokenizer(
        "How can I cancel my order?",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )
    
    # Trace the model
    try:
        logger.info("Tracing model with TorchScript...")
        traced_model = torch.jit.trace(
            model,
            [sample_input['input_ids'], sample_input['attention_mask']]
        )
        
        # Save the traced model
        optimized_model_path = os.path.join(MODEL_DIR, 'bart-customer-support-optimized')
        os.makedirs(optimized_model_path, exist_ok=True)
        traced_model.save(os.path.join(optimized_model_path, 'traced_model.pt'))
        tokenizer.save_pretrained(optimized_model_path)
        
        logger.info(f"Optimized model saved to {optimized_model_path}")
        return optimized_model_path
    except Exception as e:
        logger.warning(f"Error tracing model: {e}")
        logger.info("Falling back to standard model without optimization")
        return model_path

def test_model_inference(model_path, tokenizer_path=None):
    """Test model inference speed and performance."""
    logger.info("Testing model inference...")
    
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    # Check if the model is traced
    is_traced = os.path.exists(os.path.join(model_path, 'traced_model.pt'))
    
    # Load the model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    
    if is_traced:
        model = torch.jit.load(os.path.join(model_path, 'traced_model.pt'))
    else:
        model = BartForConditionalGeneration.from_pretrained(model_path)
    
    # Load test data
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    test_df = pd.read_csv(test_path)
    
    # Select a subset for testing
    num_test_samples = min(10, len(test_df))
    test_samples = test_df.sample(n=num_test_samples, random_state=42)
    
    # Test inference
    results = []
    total_time = 0
    
    for _, row in tqdm(test_samples.iterrows(), total=num_test_samples, desc="Testing inference"):
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            if is_traced:
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                generated_ids = outputs[0].argmax(dim=-1)
            else:
                generated_ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=MAX_TARGET_LENGTH,
                    num_beams=4,
                    early_stopping=True
                )
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        results.append({
            'input': input_text,
            'target': target_text,
            'generated': generated_text,
            'inference_time': inference_time
        })
    
    # Calculate average inference time
    avg_inference_time = total_time / num_test_samples
    logger.info(f"Average inference time: {avg_inference_time:.4f} seconds per sample")
    
    # Save results
    results_path = os.path.join(MODEL_DIR, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'avg_inference_time': avg_inference_time,
            'model_path': model_path,
            'is_traced': is_traced
        }, f, indent=2)
    
    logger.info(f"Inference results saved to {results_path}")
    
    return avg_inference_time, results

def main():
    """Main function to fine-tune and optimize the BART model."""
    logger.info("=== BART Fine-tuning for Customer Support Chatbot ===")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        logger.info("Using GPU for training")
    else:
        logger.info("Using CPU for training")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    
    # Prepare datasets
    train_dataset, val_dataset = load_and_prepare_datasets(tokenizer)
    
    # Train model
    model, trainer = train_model(train_dataset, val_dataset, tokenizer)
    
    # Optimize model for CPU
    optimized_model_path = optimize_model_for_cpu(OUTPUT_DIR)
    
    # Test model inference
    avg_inference_time, results = test_model_inference(optimized_model_path)
    
    logger.info("=== Fine-tuning and optimization completed successfully! ===")
    logger.info(f"Fine-tuned model saved to: {OUTPUT_DIR}")
    logger.info(f"Optimized model saved to: {optimized_model_path}")
    logger.info(f"Average inference time: {avg_inference_time:.4f} seconds per sample")

if __name__ == "__main__":
    main()
