"""
MIMIC-CXR Subset Downloader

Downloads a small subset of MIMIC-CXR data for testing the pipeline
without needing the full 650GB dataset.
"""
import os
import subprocess
import json
from pathlib import Path

def download_mimic_subset(username, output_dir, num_patients=10):
    """
    Download a subset of MIMIC-CXR for testing
    
    Args:
        username: PhysioNet username
        output_dir: Directory to download data to
        num_patients: Number of patients to download (default: 10)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Downloading MIMIC-CXR subset to {output_path}")
    print(f"Username: {username}")
    print(f"Patients: {num_patients}")
    print()
    
    base_url = "https://physionet.org/files/mimic-cxr/2.0.0"
    
    # Step 1: Download metadata files
    print("1. Downloading metadata files...")
    metadata_files = [
        "mimic-cxr-reports.csv.gz",
        "cxr-study-list.csv.gz", 
        "cxr-record-list.csv.gz"
    ]
    
    for filename in metadata_files:
        print(f"   Downloading {filename}...")
        cmd = [
            "wget", "--user", username, "--ask-password",
            f"{base_url}/{filename}",
            "-P", str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✓ Downloaded {filename}")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to download {filename}: {e}")
            return False
    
    # Step 2: Extract metadata files
    print("\n2. Extracting metadata files...")
    for filename in metadata_files:
        gz_path = output_path / filename
        if gz_path.exists():
            print(f"   Extracting {filename}...")
            subprocess.run(["gunzip", str(gz_path)], check=True)
            print(f"   ✓ Extracted {filename}")
    
    # Step 3: Get patient list from records
    print("\n3. Finding patients to download...")
    import pandas as pd
    
    try:
        records_df = pd.read_csv(output_path / "cxr-record-list.csv")
        patient_ids = records_df['subject_id'].unique()[:num_patients]
        print(f"   Found {len(patient_ids)} patients: {list(patient_ids)}")
    except Exception as e:
        print(f"   Error reading records: {e}")
        # Fallback to known patient IDs
        patient_ids = [10000032 + i for i in range(num_patients)]
        print(f"   Using fallback patient IDs: {list(patient_ids)}")
    
    # Step 4: Download images for selected patients
    print(f"\n4. Downloading images for {len(patient_ids)} patients...")
    
    for i, patient_id in enumerate(patient_ids):
        patient_dir = f"p{patient_id:08d}"
        print(f"   [{i+1}/{len(patient_ids)}] Downloading {patient_dir}...")
        
        # Create files directory
        files_dir = output_path / "files"
        files_dir.mkdir(exist_ok=True)
        
        # Download patient directory
        cmd = [
            "wget", "-r", "-N", "-c", "-np",
            "--user", username, "--ask-password",
            f"{base_url}/files/{patient_dir}/",
            "-P", str(files_dir),
            "--cut-dirs=4"  # Remove extra directory levels
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"   ✓ Downloaded {patient_dir}")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to download {patient_dir}: {e}")
            # Continue with other patients
    
    print(f"\n✓ Subset download complete!")
    print(f"Data saved to: {output_path}")
    print(f"\nNext steps:")
    print(f"1. cd {output_path}")
    print(f"2. python ../setup_mimic_cxr.py --data-dir .")
    print(f"3. python ../demo.py --use-mimic --mimic-data-dir .")
    
    return True

def check_wget():
    """Check if wget is available"""
    try:
        subprocess.run(["wget", "--version"], 
                      check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MIMIC-CXR subset for testing")
    parser.add_argument("--username", required=True, help="PhysioNet username")
    parser.add_argument("--output-dir", default="mimic-cxr-subset", 
                       help="Output directory (default: mimic-cxr-subset)")
    parser.add_argument("--num-patients", type=int, default=10,
                       help="Number of patients to download (default: 10)")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_wget():
        print("Error: wget is required but not found")
        print("On Windows: Install wget or use WSL")
        print("On Mac: brew install wget")
        print("On Linux: apt-get install wget / yum install wget")
        return
    
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required")
        print("Install with: pip install pandas")
        return
    
    print("MIMIC-CXR Subset Downloader")
    print("=" * 40)
    print(f"This will download a subset of MIMIC-CXR for testing.")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of patients: {args.num_patients}")
    print(f"Estimated size: ~{args.num_patients * 50}MB (varies by patient)")
    print()
    
    confirm = input("Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        return
    
    success = download_mimic_subset(
        args.username, 
        args.output_dir, 
        args.num_patients
    )
    
    if success:
        print("\n🎉 Download completed successfully!")
        print("You can now test the pipeline with real MIMIC-CXR data.")
    else:
        print("\n❌ Download failed. Please check your credentials and network.")

if __name__ == "__main__":
    main()
