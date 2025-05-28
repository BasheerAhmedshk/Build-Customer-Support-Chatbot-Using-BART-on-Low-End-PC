"""
Script to download and preprocess the Bitext customer support dataset from Hugging Face.
This script handles downloading, exploring, and preprocessing the dataset for fine-tuning.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_dataset():
    """
    Download the Bitext dataset from Hugging Face.
    Returns the dataset object.
    """
    print("Downloading Bitext customer support dataset from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        print("Dataset downloaded successfully!")
        return dataset
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def save_dataset_to_disk(dataset, split='train'):
    """
    Save the dataset to disk as CSV files.
    """
    try:
        if dataset is None:
            print("No dataset to save.")
            return False
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[split])
        
        # Save to CSV
        csv_path = os.path.join(RAW_DATA_DIR, f"bitext_customer_support_{split}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved to {csv_path}")
        
        return True
    
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return False

def explore_dataset(dataset, split='train'):
    """
    Perform exploratory data analysis on the dataset.
    """
    try:
        if dataset is None:
            print("No dataset to explore.")
            return
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame(dataset[split])
        
        # Basic information
        print("\n=== Dataset Exploration ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Display data types
        print("\nData types:")
        for col, dtype in df.dtypes.items():
            print(f"- {col}: {dtype}")
        
        # Sample data
        print("\nSample data (5 rows):")
        print(df.head())
        
        # Count unique values for categorical columns
        print("\nUnique values per column:")
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"- {col}: {unique_count} unique values")
            
            # For columns with few unique values, show distribution
            if col in ['category', 'intent'] or unique_count < 20:
                value_counts = df[col].value_counts().head(10)
                print(f"  Top values: {value_counts.to_dict()}")
        
        # Text length statistics
        if 'instruction' in df.columns:
            df['instruction_length'] = df['instruction'].apply(len)
            print("\nInstruction length statistics:")
            print(df['instruction_length'].describe())
        
        if 'response' in df.columns:
            df['response_length'] = df['response'].apply(len)
            print("\nResponse length statistics:")
            print(df['response_length'].describe())
        
        # Save exploration summary
        summary_path = os.path.join(DATA_DIR, 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Columns: {df.columns.tolist()}\n\n")
            f.write("Data types:\n")
            for col, dtype in df.dtypes.items():
                f.write(f"- {col}: {dtype}\n")
            
            f.write("\nUnique values per column:\n")
            for col in df.columns:
                unique_count = df[col].nunique()
                f.write(f"- {col}: {unique_count} unique values\n")
        
        print(f"Exploration summary saved to {summary_path}")
        
        # Create visualizations
        try:
            # Distribution of categories
            if 'category' in df.columns:
                plt.figure(figsize=(12, 6))
                df['category'].value_counts().plot(kind='bar')
                plt.title('Distribution of Categories')
                plt.xlabel('Category')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(DATA_DIR, 'category_distribution.png'))
                plt.close()
            
            # Distribution of intents
            if 'intent' in df.columns:
                plt.figure(figsize=(15, 8))
                df['intent'].value_counts().plot(kind='bar')
                plt.title('Distribution of Intents')
                plt.xlabel('Intent')
                plt.ylabel('Count')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(DATA_DIR, 'intent_distribution.png'))
                plt.close()
            
            # Text length distributions
            if 'instruction_length' in df.columns and 'response_length' in df.columns:
                plt.figure(figsize=(12, 6))
                plt.hist(df['instruction_length'], bins=50, alpha=0.5, label='Instructions')
                plt.hist(df['response_length'], bins=50, alpha=0.5, label='Responses')
                plt.title('Distribution of Text Lengths')
                plt.xlabel('Character Length')
                plt.ylabel('Count')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(DATA_DIR, 'text_length_distribution.png'))
                plt.close()
            
            print("Visualizations saved to data directory")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        # Extract sample questions for the UI
        if 'instruction' in df.columns:
            sample_questions = df['instruction'].sample(n=min(10, len(df))).tolist()
            sample_path = os.path.join(DATA_DIR, 'sample_questions.txt')
            with open(sample_path, 'w') as f:
                for i, q in enumerate(sample_questions, 1):
                    f.write(f"{i}. {q}\n")
            
            print(f"Sample questions saved to {sample_path}")
        
        return df
    
    except Exception as e:
        print(f"Error exploring dataset: {e}")
        return None

def preprocess_dataset(dataset, split='train', test_size=0.1, val_size=0.1):
    """
    Preprocess the dataset for fine-tuning.
    - Clean the data
    - Format for BART fine-tuning
    - Split into train/val/test sets
    """
    try:
        if dataset is None:
            print("No dataset to preprocess.")
            return None, None, None
        
        print("\n=== Preprocessing Dataset ===")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[split])
        
        # Basic cleaning
        print("Cleaning data...")
        
        # Remove any rows with missing values
        df = df.dropna(subset=['instruction', 'response'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['instruction', 'response'])
        
        # Format for BART fine-tuning
        print("Formatting for BART fine-tuning...")
        
        # Create a new DataFrame with the required columns
        bart_df = pd.DataFrame({
            'input_text': df['instruction'],
            'target_text': df['response']
        })
        
        # Add metadata columns if needed for analysis
        if 'category' in df.columns:
            bart_df['category'] = df['category']
        if 'intent' in df.columns:
            bart_df['intent'] = df['intent']
        
        # Split into train/val/test sets
        print(f"Splitting into train/val/test sets ({1-test_size-val_size:.1f}/{val_size:.1f}/{test_size:.1f})...")
        
        # First split off the test set
        train_val_df, test_df = train_test_split(bart_df, test_size=test_size, random_state=42)
        
        # Then split the remaining data into train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for the remaining data
        train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, random_state=42)
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        # Save processed datasets
        train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
        val_path = os.path.join(PROCESSED_DATA_DIR, 'val.csv')
        test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Processed datasets saved to {PROCESSED_DATA_DIR}")
        
        # Save metadata about the preprocessing
        metadata = {
            'original_size': len(df),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'columns': bart_df.columns.tolist(),
            'train_path': train_path,
            'val_path': val_path,
            'test_path': test_path
        }
        
        with open(os.path.join(PROCESSED_DATA_DIR, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_df, val_df, test_df
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None, None, None

def prepare_sample_for_model():
    """
    Prepare a small sample dataset for quick model testing.
    """
    try:
        # Load the processed training data
        train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
        if not os.path.exists(train_path):
            print("Training data not found. Run preprocessing first.")
            return
        
        train_df = pd.read_csv(train_path)
        
        # Create a small sample (100 examples)
        sample_size = min(100, len(train_df))
        sample_df = train_df.sample(n=sample_size, random_state=42)
        
        # Save the sample
        sample_path = os.path.join(PROCESSED_DATA_DIR, 'sample.csv')
        sample_df.to_csv(sample_path, index=False)
        
        print(f"Sample dataset with {sample_size} examples saved to {sample_path}")
    
    except Exception as e:
        print(f"Error preparing sample dataset: {e}")

def main():
    print("=== Bitext Customer Support Dataset Processor ===")
    
    # Download dataset
    dataset = download_dataset()
    
    if dataset is not None:
        # Save raw dataset
        save_dataset_to_disk(dataset)
        
        # Explore dataset
        explore_dataset(dataset)
        
        # Preprocess dataset
        train_df, val_df, test_df = preprocess_dataset(dataset)
        
        # Prepare sample for quick testing
        prepare_sample_for_model()
        
        print("\n=== Dataset processing completed successfully! ===")
    else:
        print("Dataset processing failed.")

if __name__ == "__main__":
    main()
