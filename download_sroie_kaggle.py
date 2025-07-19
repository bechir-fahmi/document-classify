#!/usr/bin/env python3
"""
Download and integrate SROIE dataset from Kaggle
to improve receipt detection performance
"""
import pandas as pd
import os
import config
import zipfile
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_kaggle_api():
    """Setup Kaggle API authentication"""
    print("=== SETTING UP KAGGLE API ===\n")
    
    try:
        # Check if kaggle.json exists
        kaggle_config_path = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_config_path.exists():
            print("❌ Kaggle API credentials not found!")
            print("\n📋 To setup Kaggle API:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Download kaggle.json")
            print("4. Place it in: ~/.kaggle/kaggle.json")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Test API connection
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print("✅ Kaggle API authenticated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Kaggle API: {str(e)}")
        print("❌ Failed to authenticate Kaggle API")
        print("Make sure you have kaggle installed: pip install kaggle")
        return False

def extract_sroie_dataset():
    """Extract SROIE dataset from existing zip file"""
    print("=== EXTRACTING SROIE DATASET FROM ZIP FILE ===\n")
    
    try:
        # Look for the zip file
        zip_path = os.path.join(config.DATA_DIR, "archive (1).zip")
        
        if not os.path.exists(zip_path):
            print(f"❌ Zip file not found at: {zip_path}")
            return None
        
        print(f"Found zip file: {zip_path}")
        
        # Setup extraction directory
        extract_dir = os.path.join(config.DATA_DIR, "kaggle", "sroie")
        os.makedirs(extract_dir, exist_ok=True)
        
        logger.info("Extracting SROIE dataset from zip file...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"✅ Dataset extracted to: {extract_dir}")
        
        # List extracted files and directories
        for root, dirs, files in os.walk(extract_dir):
            level = root.replace(extract_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files per directory
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
        
        return extract_dir
        
    except Exception as e:
        logger.error(f"Error extracting SROIE dataset: {str(e)}")
        print("❌ Failed to extract SROIE dataset")
        return None

def process_sroie_text_files(download_dir, max_samples=1500):
    """Process SROIE text files for receipt classification training"""
    print(f"\n=== PROCESSING SROIE TEXT FILES ===\n")
    
    if not download_dir or not os.path.exists(download_dir):
        return []
    
    try:
        receipt_samples = []
        
        # Look for text files in the dataset
        for root, dirs, files in os.walk(download_dir):
            text_files = [f for f in files if f.endswith('.txt')]
            
            if text_files:
                print(f"Found {len(text_files)} text files in {root}")
                
                for i, txt_file in enumerate(text_files[:max_samples]):
                    txt_path = os.path.join(root, txt_file)
                    
                    try:
                        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        
                        if content and len(content) > 10:
                            receipt_samples.append({
                                'text': content,
                                'label': 'receipt',
                                'source_file': txt_file
                            })
                            
                            if (i + 1) % 100 == 0:
                                print(f"Processed {i + 1} text files...")
                                
                    except Exception as e:
                        logger.warning(f"Error reading {txt_file}: {str(e)}")
                        continue
        
        # Also look for JSON files with text data
        for root, dirs, files in os.walk(download_dir):
            json_files = [f for f in files if f.endswith('.json')]
            
            if json_files:
                print(f"Found {len(json_files)} JSON files in {root}")
                
                for json_file in json_files[:200]:  # Limit JSON processing
                    json_path = os.path.join(root, json_file)
                    
                    try:
                        with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                            data = json.load(f)
                        
                        # Extract text from JSON structure
                        text_content = ""
                        if isinstance(data, dict):
                            # Look for common text fields
                            for key in ['text', 'content', 'ocr_text', 'extracted_text']:
                                if key in data and data[key]:
                                    text_content = str(data[key])
                                    break
                            
                            # If no direct text, try to extract from nested structures
                            if not text_content and 'annotations' in data:
                                annotations = data['annotations']
                                if isinstance(annotations, list):
                                    text_parts = []
                                    for ann in annotations:
                                        if isinstance(ann, dict) and 'text' in ann:
                                            text_parts.append(ann['text'])
                                    text_content = ' '.join(text_parts)
                        
                        if text_content and len(text_content) > 10:
                            receipt_samples.append({
                                'text': text_content,
                                'label': 'receipt',
                                'source_file': json_file
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error reading {json_file}: {str(e)}")
                        continue
        
        print(f"✅ Successfully processed {len(receipt_samples)} receipt samples")
        return receipt_samples
        
    except Exception as e:
        logger.error(f"Error processing SROIE files: {str(e)}")
        return []

def integrate_with_existing_data(receipt_samples):
    """Integrate SROIE receipt samples with existing training data"""
    print(f"\n=== INTEGRATING WITH EXISTING DATA ===\n")
    
    # Load existing training data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if os.path.exists(complete_data_path):
        existing_df = pd.read_csv(complete_data_path)
        print(f"Existing training data: {len(existing_df)} samples")
        
        # Check current receipt samples
        existing_receipts = len(existing_df[existing_df['label'] == 'receipt'])
        print(f"Current receipt samples: {existing_receipts}")
        
    else:
        print("❌ Existing training data not found!")
        return False
    
    # Create DataFrame from new receipt samples
    new_receipts_df = pd.DataFrame(receipt_samples)
    print(f"New receipt samples to add: {len(new_receipts_df)}")
    
    # Remove duplicates based on text content
    print("Removing duplicates...")
    combined_df = pd.concat([existing_df, new_receipts_df], ignore_index=True)
    initial_size = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    final_size = len(combined_df)
    duplicates_removed = initial_size - final_size
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate samples")
    
    # Check final distribution
    print(f"\nFinal dataset: {len(combined_df)} samples")
    final_receipts = len(combined_df[combined_df['label'] == 'receipt'])
    print(f"Total receipt samples: {final_receipts}")
    
    # Save updated dataset
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_sroie_kaggle.csv")
    existing_df.to_csv(backup_path, index=False)
    print(f"✅ Backup saved to: {backup_path}")
    
    combined_df.to_csv(complete_data_path, index=False)
    print(f"✅ Updated training data saved to: {complete_data_path}")
    
    return True

def create_sroie_sample_preview(receipt_samples, num_samples=5):
    """Create a preview of SROIE receipt samples"""
    print(f"\n=== SROIE RECEIPT SAMPLES PREVIEW ===\n")
    
    for i, sample in enumerate(receipt_samples[:num_samples]):
        print(f"📄 Receipt Sample {i+1} (from {sample['source_file']}):")
        print("-" * 60)
        text_preview = sample['text'][:400] + "..." if len(sample['text']) > 400 else sample['text']
        print(text_preview)
        print("-" * 60)
        print()

def show_class_distribution_comparison():
    """Show before/after class distribution"""
    print(f"\n=== CLASS DISTRIBUTION COMPARISON ===\n")
    
    # Load updated data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_sroie_kaggle.csv")
    
    if os.path.exists(backup_path) and os.path.exists(complete_data_path):
        # Before
        before_df = pd.read_csv(backup_path)
        before_receipts = len(before_df[before_df['label'] == 'receipt'])
        
        # After
        after_df = pd.read_csv(complete_data_path)
        after_receipts = len(after_df[after_df['label'] == 'receipt'])
        
        print("📊 RECEIPT SAMPLES:")
        print(f"   Before: {before_receipts} samples")
        print(f"   After:  {after_receipts} samples")
        print(f"   Added:  {after_receipts - before_receipts} samples")
        print(f"   Improvement: {((after_receipts - before_receipts) / max(before_receipts, 1)) * 100:.1f}% increase")
        
        print(f"\n📈 TOTAL DATASET:")
        print(f"   Before: {len(before_df)} samples")
        print(f"   After:  {len(after_df)} samples")
        print(f"   Growth: {len(after_df) - len(before_df)} samples")

def main():
    """Main function to extract and integrate SROIE dataset from existing zip file"""
    print("🚀 SROIE KAGGLE DATASET INTEGRATION FOR RECEIPT DETECTION\n")
    
    # Step 1: Extract SROIE dataset from zip file
    extract_dir = extract_sroie_dataset()
    if not extract_dir:
        print("❌ Failed to extract dataset. Exiting...")
        return
    
    # Step 2: Process text files
    receipt_samples = process_sroie_text_files(extract_dir, max_samples=1500)
    if not receipt_samples:
        print("❌ Failed to process receipt samples. Exiting...")
        return
    
    # Step 4: Show preview
    create_sroie_sample_preview(receipt_samples)
    
    # Step 5: Integrate with existing data
    success = integrate_with_existing_data(receipt_samples)
    if success:
        # Step 6: Show comparison
        show_class_distribution_comparison()
        
        print(f"\n🎉 SUCCESS! SROIE KAGGLE DATASET INTEGRATED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Retrain your model:")
        print(f"   python app.py train --enhanced")
        print(f"2. Test receipt detection:")
        print(f"   python app.py api")
        print(f"3. Expected improvement:")
        print(f"   Receipt accuracy: Current → 85%+ (estimated)")
        print(f"\n💡 TIP: The SROIE dataset contains high-quality receipt text")
        print(f"   which should significantly improve receipt classification!")
    else:
        print("❌ Failed to integrate dataset")

if __name__ == "__main__":
    main()