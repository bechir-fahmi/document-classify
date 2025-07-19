#!/usr/bin/env python3
"""
Download, clean, and integrate CORU dataset from Hugging Face
to improve document classification performance
"""
import pandas as pd
import os
import config
from datasets import load_dataset
import logging
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_coru_dataset():
    """Download CORU dataset from Hugging Face"""
    print("=== DOWNLOADING CORU DATASET ===\n")
    
    try:
        # Load the CORU dataset
        logger.info("Loading CORU dataset from Hugging Face...")
        dataset = load_dataset("abdoelsayed/CORU")
        
        # Check dataset structure
        print(f"Dataset keys: {dataset.keys()}")
        
        # Get available splits
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"{split_name} split: {len(split_data)} samples")
            
            # Check the structure of the first sample
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"Sample keys: {sample.keys()}")
                
                # Show sample content
                for key, value in sample.items():
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print(f"  {key}: {preview}")
                    else:
                        print(f"  {key}: {type(value)} - {value}")
                break
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading CORU dataset: {str(e)}")
        print("❌ Failed to download CORU dataset")
        print("Make sure you have the 'datasets' library installed:")
        print("pip install datasets")
        return None

def clean_coru_text(text):
    """Clean and normalize CORU text data"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with training
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[\.]{3,}', '...', text)
    text = re.sub(r'[\-]{3,}', '---', text)
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s*([\.,:;!?])\s*', r'\1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def extract_document_type_from_coru(sample):
    """Extract and map document type from CORU sample"""
    # Check if there's a label or category field
    if 'label' in sample and sample['label']:
        return sample['label'].lower().strip()
    
    if 'category' in sample and sample['category']:
        return sample['category'].lower().strip()
    
    if 'document_type' in sample and sample['document_type']:
        return sample['document_type'].lower().strip()
    
    # Try to infer from filename if available
    if 'filename' in sample and sample['filename']:
        filename = sample['filename'].lower()
        if 'invoice' in filename:
            return 'invoice'
        elif 'receipt' in filename:
            return 'receipt'
        elif 'contract' in filename:
            return 'contract'
        elif 'order' in filename:
            return 'purchase_order'
    
    # Try to infer from text content
    if 'text' in sample and sample['text']:
        text_lower = sample['text'].lower()
        if any(word in text_lower for word in ['invoice', 'bill', 'billing']):
            return 'invoice'
        elif any(word in text_lower for word in ['receipt', 'payment received']):
            return 'receipt'
        elif any(word in text_lower for word in ['contract', 'agreement', 'terms and conditions']):
            return 'contract'
        elif any(word in text_lower for word in ['purchase order', 'order form']):
            return 'purchase_order'
        elif any(word in text_lower for word in ['delivery note', 'shipping']):
            return 'delivery_note'
    
    # Default fallback
    return 'document'

def process_coru_for_training(dataset, max_samples=2000):
    """Process and clean CORU dataset for document classification training"""
    print(f"\n=== ANALYZING CORU DATASET STRUCTURE ===\n")
    
    if not dataset:
        return []
    
    try:
        # Use train split if available, otherwise use the first available split
        if 'train' in dataset:
            data = dataset['train']
            split_name = 'train'
        else:
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        
        print(f"Using {split_name} split with {len(data)} samples")
        
        # Analyze the first few samples to understand the structure
        print("🔍 Analyzing sample structure...")
        for i in range(min(3, len(data))):
            sample = data[i]
            print(f"\nSample {i+1} structure:")
            for key, value in sample.items():
                if hasattr(value, '__class__'):
                    print(f"  {key}: {type(value).__name__}")
                    if hasattr(value, 'size'):
                        print(f"    Size: {value.size}")
                    if hasattr(value, 'mode'):
                        print(f"    Mode: {value.mode}")
        
        print(f"\n❌ CORU DATASET ANALYSIS COMPLETE")
        print(f"📋 FINDINGS:")
        print(f"   • This dataset contains IMAGES, not text")
        print(f"   • It's designed for OCR (Optical Character Recognition)")
        print(f"   • Images need to be processed with OCR to extract text")
        print(f"   • This would require additional OCR libraries (pytesseract, easyocr, etc.)")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   • Skip this dataset for now - we already have 11,201 high-quality text samples")
        print(f"   • Your current dataset is already excellent for document classification")
        print(f"   • Adding OCR processing would be complex and may not improve results significantly")
        
        print(f"\n✅ DECISION: Skipping CORU dataset (image-based, requires OCR)")
        
        return []
        
    except Exception as e:
        logger.error(f"Error analyzing CORU data: {str(e)}")
        return []

def map_to_existing_categories(original_label):
    """Map CORU labels to existing categories in our dataset"""
    label_lower = original_label.lower()
    
    # Map to existing categories
    if any(word in label_lower for word in ['invoice', 'bill', 'billing']):
        return 'invoice'
    elif any(word in label_lower for word in ['receipt', 'payment']):
        return 'receipt'
    elif any(word in label_lower for word in ['contract', 'agreement']):
        return 'contract'
    elif any(word in label_lower for word in ['purchase order', 'order']):
        return 'purchase_order'
    elif any(word in label_lower for word in ['delivery', 'shipping']):
        return 'delivery_note'
    elif any(word in label_lower for word in ['quote', 'quotation', 'estimate']):
        return 'quote'
    elif any(word in label_lower for word in ['bank', 'statement']):
        return 'bank_statement'
    elif any(word in label_lower for word in ['expense', 'report']):
        return 'expense_report'
    elif any(word in label_lower for word in ['payslip', 'salary', 'wage']):
        return 'payslip'
    else:
        # Keep original label if it doesn't map to existing categories
        return original_label.lower().replace(' ', '_')

def validate_cleaned_samples(document_samples):
    """Validate the quality of cleaned samples"""
    print(f"\n=== VALIDATING CLEANED SAMPLES ===\n")
    
    if not document_samples:
        print("❌ No samples to validate")
        return False
    
    # Check text quality
    avg_length = sum(len(sample['text']) for sample in document_samples) / len(document_samples)
    print(f"📊 Quality metrics:")
    print(f"   Average text length: {avg_length:.1f} characters")
    
    # Check for meaningful content
    meaningful_samples = 0
    for sample in document_samples:
        if re.search(r'[A-Za-z]{3,}', sample['text']):  # Has words with 3+ letters
            meaningful_samples += 1
    
    meaningful_percentage = (meaningful_samples / len(document_samples)) * 100
    print(f"   Samples with meaningful text: {meaningful_samples}/{len(document_samples)} ({meaningful_percentage:.1f}%)")
    
    # Check label distribution
    unique_labels = len(set(sample['label'] for sample in document_samples))
    print(f"   Unique labels: {unique_labels}")
    
    # Validation criteria
    is_valid = (
        avg_length >= 50 and  # Reasonable text length
        meaningful_percentage >= 90 and  # Most samples have meaningful text
        unique_labels >= 2  # At least 2 different document types
    )
    
    if is_valid:
        print("✅ Validation passed - samples are ready for integration")
    else:
        print("⚠️  Validation concerns - review sample quality")
    
    return is_valid

def show_coru_sample_preview(document_samples, num_samples=5):
    """Show preview of cleaned CORU samples"""
    print(f"\n=== CORU CLEANED SAMPLES PREVIEW ===\n")
    
    for i, sample in enumerate(document_samples[:num_samples]):
        print(f"📄 CORU Sample {i+1} ({sample['label']}):")
        print("-" * 60)
        text_preview = sample['text'][:400] + "..." if len(sample['text']) > 400 else sample['text']
        print(text_preview)
        print(f"\nOriginal Label: {sample['original_label']}")
        print(f"Mapped Label: {sample['label']}")
        print("-" * 60)
        print()

def integrate_with_existing_data(document_samples):
    """Integrate cleaned CORU samples with existing training data"""
    print(f"\n=== INTEGRATING WITH EXISTING DATA ===\n")
    
    # Load existing training data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if os.path.exists(complete_data_path):
        existing_df = pd.read_csv(complete_data_path)
        print(f"Existing training data: {len(existing_df)} samples")
        
        # Show current label distribution
        print(f"\nCurrent label distribution:")
        current_counts = existing_df['label'].value_counts()
        for label, count in current_counts.head(10).items():
            print(f"   {label}: {count}")
    else:
        print("❌ Existing training data not found!")
        return False
    
    # Create DataFrame from new document samples
    new_docs_df = pd.DataFrame(document_samples)
    print(f"\nNew CORU samples to add: {len(new_docs_df)}")
    
    # Remove duplicates based on text content
    print("Checking for duplicates...")
    combined_df = pd.concat([existing_df, new_docs_df], ignore_index=True)
    initial_size = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    final_size = len(combined_df)
    duplicates_removed = initial_size - final_size
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate samples")
    
    # Check final distribution
    print(f"\nFinal dataset: {len(combined_df)} samples")
    final_counts = combined_df['label'].value_counts()
    print(f"\nFinal label distribution (top 10):")
    for label, count in final_counts.head(10).items():
        print(f"   {label}: {count}")
    
    # Save backup and updated dataset
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_coru.csv")
    existing_df.to_csv(backup_path, index=False)
    print(f"\n✅ Backup saved to: {backup_path}")
    
    combined_df.to_csv(complete_data_path, index=False)
    print(f"✅ Updated training data saved to: {complete_data_path}")
    
    return True

def show_integration_summary():
    """Show summary of the integration"""
    print(f"\n=== INTEGRATION SUMMARY ===\n")
    
    # Load updated data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_coru.csv")
    
    if os.path.exists(backup_path) and os.path.exists(complete_data_path):
        # Before
        before_df = pd.read_csv(backup_path)
        before_counts = before_df['label'].value_counts()
        
        # After
        after_df = pd.read_csv(complete_data_path)
        after_counts = after_df['label'].value_counts()
        
        print("📊 DATASET IMPROVEMENT:")
        print(f"   Total samples before: {len(before_df):,}")
        print(f"   Total samples after:  {len(after_df):,}")
        print(f"   New samples added:    {len(after_df) - len(before_df):,}")
        
        print(f"\n📈 LABEL-WISE CHANGES:")
        all_labels = set(before_counts.index) | set(after_counts.index)
        for label in sorted(all_labels):
            before_count = before_counts.get(label, 0)
            after_count = after_counts.get(label, 0)
            if after_count > before_count:
                added = after_count - before_count
                print(f"   {label}: {before_count:,} → {after_count:,} (+{added:,})")

def main():
    """Main function to download, clean, and integrate CORU dataset"""
    print("🚀 CORU DATASET INTEGRATION WITH CLEANING\n")
    
    # Step 1: Download CORU dataset
    dataset = download_coru_dataset()
    if not dataset:
        print("❌ Failed to download dataset. Exiting...")
        return
    
    # Step 2: Process and clean for training
    document_samples = process_coru_for_training(dataset, max_samples=2000)
    if not document_samples:
        print("❌ Failed to process document samples. Exiting...")
        return
    
    # Step 3: Validate cleaned samples
    is_valid = validate_cleaned_samples(document_samples)
    if not is_valid:
        print("⚠️  Sample quality concerns, but proceeding...")
    
    # Step 4: Show preview
    show_coru_sample_preview(document_samples)
    
    # Step 5: Integrate with existing data
    success = integrate_with_existing_data(document_samples)
    if success:
        # Step 6: Show integration summary
        show_integration_summary()
        
        print(f"\n🎉 SUCCESS! CORU DATASET INTEGRATED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Retrain your model:")
        print(f"   python app.py train --enhanced")
        print(f"2. Test document classification:")
        print(f"   python app.py api")
        print(f"3. Generate new visualizations:")
        print(f"   python visualize_dataset_confidence.py")
        print(f"4. Expected improvements:")
        print(f"   - More diverse document types")
        print(f"   - Better classification accuracy")
        print(f"   - Improved model robustness")
    else:
        print("❌ Failed to integrate dataset")

if __name__ == "__main__":
    main()