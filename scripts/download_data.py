#!/usr/bin/env python3

import os
from huggingface_hub import snapshot_download

def download_datasets(repo_id="HuggingFaceH4/MATH-500", local_dir="data/math"):
    """
    Download all datasets from Hugging Face Hub to local directory.
    
    Args:
        repo_id (str): Hugging Face repository ID
        local_dir (str): Local directory to save datasets
    """
    print(f"Downloading datasets from {repo_id}...")
    
    # Create the data directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download the entire repository
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nDatasets successfully downloaded to {local_dir}/")
        
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_datasets()