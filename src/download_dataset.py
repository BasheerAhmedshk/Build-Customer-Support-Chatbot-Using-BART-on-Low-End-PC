"""
Script to download the Bitext customer support dataset from Kaggle.
This script provides two methods to download the dataset:
1. Using the Kaggle API (requires API key setup)
2. Manual download instructions (if API method fails)
"""

import os
import sys
import subprocess
import zipfile
import pandas as pd

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def setup_kaggle_api():
    """
    Setup Kaggle API credentials.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if kaggle is installed
        subprocess.run(['pip', 'install', 'kaggle'], check=True)
        
        # Check if kaggle.json exists
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json_path):
            print("\n=== Kaggle API Setup Instructions ===")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll down to 'API' section and click 'Create New API Token'")
            print("3. This will download a kaggle.json file")
            print(f"4. Place this file in {kaggle_dir}")
            print("5. Run this script again\n")
            return False
        
        # Set permissions
        os.chmod(kaggle_json_path, 0o600)
        return True
    
    except Exception as e:
        print(f"Error setting up Kaggle API: {e}")
        return False

def download_dataset_api():
    """
    Download the Bitext dataset using Kaggle API.
    Returns True if successful, False otherwise.
    """
    try:
        dataset_name = "bitext/bitext-gen-ai-chatbot-customer-support-dataset"
        output_path = RAW_DATA_DIR
        
        print(f"Downloading dataset from Kaggle: {dataset_name}")
        subprocess.run(['kaggle', 'datasets', 'download', dataset_name, '-p', output_path, '--unzip'], check=True)
        
        # Verify download
        csv_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
        if csv_files:
            print(f"Dataset downloaded successfully. Files: {csv_files}")
            return True
        else:
            print("Download completed but no CSV files found.")
            return False
    
    except Exception as e:
        print(f"Error downloading dataset via API: {e}")
        return False

def manual_download_instructions():
    """
    Provide instructions for manual download.
    """
    print("\n=== Manual Download Instructions ===")
    print("1. Go to https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset")
    print("2. Click the 'Download' button (requires Kaggle account)")
    print("3. Download the ZIP file (approximately 3 MB)")
    print(f"4. Extract the contents to: {RAW_DATA_DIR}")
    print("5. Run this script again with --manual flag\n")

def process_manual_download():
    """
    Process manually downloaded dataset.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if files exist
        csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {RAW_DATA_DIR}")
            return False
        
        print(f"Found manually downloaded files: {csv_files}")
        return True
    
    except Exception as e:
        print(f"Error processing manual download: {e}")
        return False

def explore_dataset():
    """
    Perform initial exploration of the dataset.
    """
    try:
        # Find CSV files
        csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        if not csv_files:
            print("No CSV files found to explore.")
            return
        
        main_file = os.path.join(RAW_DATA_DIR, csv_files[0])
        print(f"\n=== Exploring dataset: {main_file} ===")
        
        # Load the dataset
        df = pd.read_csv(main_file)
        
        # Display basic information
        print("\nDataset shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nData types:\n", df.dtypes)
        print("\nSample data (5 rows):\n", df.head())
        
        # Count unique values for categorical columns
        print("\nUnique values per column:")
        for col in df.columns:
            print(f"- {col}: {df[col].nunique()} unique values")
        
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
                f.write(f"- {col}: {df[col].nunique()} unique values\n")
        
        print(f"\nSummary saved to {summary_path}")
        
        # Extract sample questions for the UI
        sample_questions = df['instruction'].sample(n=min(10, len(df))).tolist()
        sample_path = os.path.join(DATA_DIR, 'sample_questions.txt')
        with open(sample_path, 'w') as f:
            for i, q in enumerate(sample_questions, 1):
                f.write(f"{i}. {q}\n")
        
        print(f"Sample questions saved to {sample_path}")
        
    except Exception as e:
        print(f"Error exploring dataset: {e}")

def main():
    print("=== Bitext Customer Support Dataset Downloader ===")
    
    # Check for manual flag
    manual_mode = '--manual' in sys.argv
    
    if manual_mode:
        if process_manual_download():
            explore_dataset()
        else:
            manual_download_instructions()
    else:
        # Try API download first
        if setup_kaggle_api():
            if download_dataset_api():
                explore_dataset()
            else:
                manual_download_instructions()
        else:
            manual_download_instructions()

if __name__ == "__main__":
    main()
