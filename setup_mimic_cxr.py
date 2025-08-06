"""
MIMIC-CXR Dataset Setup

This script helps you download and prepare the MIMIC-CXR dataset for use with
the medical image report generation pipeline.

MIMIC-CXR is a large dataset of chest X-rays with corresponding radiology reports.
It requires PhysioNet credentialed access.
"""
import os
import json
import pandas as pd
from pathlib import Path
import gzip
import csv
from typing import Dict, List, Optional

# Dataset information
MIMIC_CXR_INFO = {
    "name": "MIMIC-CXR",
    "version": "2.0.0",
    "description": "Large dataset of chest X-rays with radiology reports",
    "homepage": "https://physionet.org/content/mimic-cxr/2.0.0/",
    "size": "~647 GB (images) + ~1.6 GB (reports)",
    "samples": "377,110 images, 227,835 reports"
}

def print_dataset_info():
    """Print information about the MIMIC-CXR dataset"""
    print("=" * 60)
    print("MIMIC-CXR Dataset Information")
    print("=" * 60)
    for key, value in MIMIC_CXR_INFO.items():
        print(f"{key.capitalize()}: {value}")
    print("\nRequirements:")
    print("- PhysioNet credentialed account")
    print("- Signed Data Use Agreement")
    print("- Training in human subjects research")
    print("=" * 60)

def check_prerequisites():
    """Check if necessary packages are installed"""
    required_packages = ['pandas', 'requests']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True

def create_download_instructions():
    """Create instructions for downloading MIMIC-CXR"""
    instructions = """
# MIMIC-CXR Download Instructions

## Step 1: Get PhysioNet Access
1. Create account at https://physionet.org/register/
2. Complete credentialing process (requires training completion)
3. Sign the MIMIC-CXR Data Use Agreement

## Step 2: Download Dataset
Option A - Use wget (recommended):
```bash
# Download reports and metadata
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/

# Or download specific files:
wget --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.csv.gz
wget --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv.gz
wget --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv.gz
```

Option B - Use PhysioNet's download script:
```bash
# Install physionet-data-loader
pip install physionet-data-loader

# Download with authentication
physionet-download -u YOUR_USERNAME -p PASSWORD mimic-cxr/2.0.0/
```

## Step 3: Extract Files
```bash
# Extract compressed files
gunzip mimic-cxr-reports.csv.gz
gunzip cxr-study-list.csv.gz
gunzip cxr-record-list.csv.gz
```

## Step 4: Run Setup Script
```bash
python setup_mimic_cxr.py --data-dir /path/to/mimic-cxr/
```
"""
    
    with open("MIMIC_CXR_DOWNLOAD.md", "w") as f:
        f.write(instructions)
    
    print("Created MIMIC_CXR_DOWNLOAD.md with detailed instructions")

def setup_mimic_structure(data_dir: str):
    """Set up directory structure for MIMIC-CXR"""
    data_path = Path(data_dir)
    
    # Create directory structure
    directories = [
        "files",
        "reports", 
        "processed",
        "processed/train",
        "processed/val",
        "processed/test"
    ]
    
    for dir_name in directories:
        (data_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {data_path}")

def process_mimic_reports(data_dir: str, output_dir: str = None):
    """Process MIMIC-CXR reports into our format"""
    if not output_dir:
        output_dir = os.path.join(data_dir, "processed")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # File paths
    reports_file = data_path / "mimic-cxr-reports.csv"
    study_list_file = data_path / "cxr-study-list.csv" 
    record_list_file = data_path / "cxr-record-list.csv"
    
    # Check if files exist
    missing_files = []
    for file_path in [reports_file, study_list_file, record_list_file]:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease download these files first using the instructions.")
        return False
    
    print("Processing MIMIC-CXR reports...")
    
    try:
        # Load data
        print("Loading reports...")
        reports_df = pd.read_csv(reports_file)
        
        print("Loading study list...")
        studies_df = pd.read_csv(study_list_file)
        
        print("Loading record list...")
        records_df = pd.read_csv(record_list_file)
        
        # Merge datasets
        print("Merging datasets...")
        # Merge reports with studies
        merged_df = reports_df.merge(studies_df, on='study_id', how='inner')
        
        # Merge with records to get image paths
        final_df = merged_df.merge(records_df, on='study_id', how='inner')
        
        print(f"Total merged records: {len(final_df)}")
        
        # Filter for valid reports (with findings and impression)
        print("Filtering valid reports...")
        valid_reports = final_df[
            (final_df['findings'].notna()) & 
            (final_df['impression'].notna()) &
            (final_df['findings'].str.len() > 10) &
            (final_df['impression'].str.len() > 10)
        ].copy()
        
        print(f"Valid reports after filtering: {len(valid_reports)}")
        
        # Convert to our format
        processed_data = []
        
        for idx, row in valid_reports.iterrows():
            # Create image path (MIMIC-CXR structure: files/p{patient_id}/p{patient_id}/s{study_id}/{dicom_id}.jpg)
            patient_id = f"p{row['subject_id']}"
            study_id = f"s{row['study_id']}"
            image_path = f"files/{patient_id}/{patient_id}/{study_id}/{row['dicom_id']}.jpg"
            
            record = {
                "id": f"mimic_cxr_{row['study_id']}_{row['dicom_id']}",
                "image": image_path,
                "findings": clean_text(str(row['findings'])),
                "impression": clean_text(str(row['impression'])),
                "study_id": row['study_id'],
                "subject_id": row['subject_id'],
                "split": row.get('split', 'train')  # Use split if available
            }
            processed_data.append(record)
        
        # Split data if no split column exists
        if 'split' not in valid_reports.columns:
            print("Creating train/val/test splits...")
            import random
            random.seed(42)
            random.shuffle(processed_data)
            
            n_total = len(processed_data)
            n_train = int(0.8 * n_total)
            n_val = int(0.1 * n_total)
            
            for i, record in enumerate(processed_data):
                if i < n_train:
                    record['split'] = 'train'
                elif i < n_train + n_val:
                    record['split'] = 'validate'
                else:
                    record['split'] = 'test'
        
        # Save splits
        splits = {'train': [], 'validate': [], 'test': []}
        for record in processed_data:
            split = record['split']
            if split == 'val':
                split = 'validate'  # Normalize naming
            if split in splits:
                # Remove split from individual records
                record_copy = {k: v for k, v in record.items() if k != 'split'}
                splits[split].append(record_copy)
        
        # Save JSON files
        for split_name, split_data in splits.items():
            if split_data:  # Only save if data exists
                output_file = output_path / f"{split_name}.json"
                with open(output_file, 'w') as f:
                    json.dump(split_data, f, indent=2)
                print(f"Saved {len(split_data)} {split_name} examples to {output_file}")
        
        # Save combined dataset info
        dataset_info = {
            "name": "MIMIC-CXR",
            "version": "2.0.0",
            "processed_date": str(pd.Timestamp.now()),
            "total_samples": len(processed_data),
            "splits": {split: len(data) for split, data in splits.items() if data},
            "sample_fields": ["id", "image", "findings", "impression", "study_id", "subject_id"]
        }
        
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nProcessing complete!")
        print(f"Total samples: {len(processed_data)}")
        print(f"Splits: {dataset_info['splits']}")
        print(f"Output directory: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing MIMIC-CXR data: {e}")
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text from reports"""
    if pd.isna(text) or text == 'nan':
        return ""
    
    # Basic cleaning
    text = str(text).strip()
    
    # Remove common artifacts
    text = text.replace('___', ' ')
    text = text.replace('  ', ' ')
    
    # Remove leading/trailing whitespace and periods
    text = text.strip(' .')
    
    return text

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "dataset": {
            "name": "MIMIC-CXR",
            "data_dir": "path/to/mimic-cxr-2.0.0",
            "processed_dir": "path/to/mimic-cxr-2.0.0/processed",
            "image_dir": "path/to/mimic-cxr-2.0.0/files"
        },
        "training": {
            "batch_size": 4,
            "max_length": 512,
            "image_size": [224, 224],
            "learning_rate": 1e-4
        },
        "template": {
            "type": "radiology",
            "image_token": "<image>",
            "system_prompt": "You are an expert radiologist analyzing chest X-rays."
        }
    }
    
    with open("mimic_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created sample configuration: mimic_config.json")

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up MIMIC-CXR dataset")
    parser.add_argument("--data-dir", type=str, help="Path to MIMIC-CXR data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed data")
    parser.add_argument("--info-only", action="store_true", help="Only show dataset information")
    parser.add_argument("--instructions", action="store_true", help="Create download instructions")
    
    args = parser.parse_args()
    
    if args.info_only:
        print_dataset_info()
        return
    
    if args.instructions:
        create_download_instructions()
        create_sample_config()
        return
    
    if not check_prerequisites():
        return
    
    print_dataset_info()
    
    if args.data_dir:
        if not os.path.exists(args.data_dir):
            print(f"Creating data directory: {args.data_dir}")
            os.makedirs(args.data_dir, exist_ok=True)
        
        setup_mimic_structure(args.data_dir)
        
        # Check if data files exist, if so process them
        data_path = Path(args.data_dir)
        if (data_path / "mimic-cxr-reports.csv").exists():
            print("\nFound MIMIC-CXR data files, processing...")
            success = process_mimic_reports(args.data_dir, args.output_dir)
            if success:
                print("\nSetup complete! You can now use the processed data with:")
                print("python demo.py --use-mimic --data-dir", args.data_dir)
        else:
            print(f"\nData directory created: {args.data_dir}")
            print("Please download MIMIC-CXR files to this directory first.")
            print("Run: python setup_mimic_cxr.py --instructions")
    else:
        print("\nTo set up MIMIC-CXR:")
        print("1. python setup_mimic_cxr.py --instructions  # Get download instructions")
        print("2. Download data to a directory")
        print("3. python setup_mimic_cxr.py --data-dir /path/to/data  # Process data")

if __name__ == "__main__":
    main()
