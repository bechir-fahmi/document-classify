#!/usr/bin/env python3
"""
Download and integrate CompanyDocuments dataset from Hugging Face
to improve document classification performance
"""
import pandas as pd
import os
import config
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_company_documents_dataset():
    """Download CompanyDocuments dataset from Hugging Face"""
    print("=== DOWNLOADING COMPANY DOCUMENTS DATASET ===\n")
    
    try:
        # Load the CompanyDocuments dataset
        logger.info("Loading CompanyDocuments dataset from Hugging Face...")
        dataset = load_dataset("AyoubChLin/CompanyDocuments")
        
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
                break
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading CompanyDocuments dataset: {str(e)}")
        print("❌ Failed to download CompanyDocuments dataset")
        print("Make sure you have the 'datasets' library installed:")
        print("pip install datasets")
        return None

def process_company_documents_for_training(dataset, max_samples=2000):
    """Process CompanyDocuments dataset for document classification training"""
    print(f"\n=== PROCESSING COMPANY DOCUMENTS DATA FOR TRAINING ===\n")
    
    if not dataset:
        return []
    
    try:
        # Use train split if available, otherwise use the first available split
        if 'train' in dataset:
            data = dataset['train']
        else:
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"Using {split_name} split")
        
        # First, let's examine the actual structure
        print("Examining dataset structure...")
        sample = data[0]
        print(f"Available keys: {list(sample.keys())}")
        for key in sample.keys():
            value = sample[key]
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        # Extract document samples
        document_samples = []
        logger.info(f"Processing up to {max_samples} document samples...")
        
        for i, sample in enumerate(data):
            if i >= max_samples:
                break
            
            # Get the document text and label based on actual structure
            text = sample.get('file_content', '') or sample.get('extracted_data', '')
            label = sample.get('document_type', '') or sample.get('file_name', '').split('.')[-2] if '.' in sample.get('file_name', '') else ''
            
            # If file_content is empty, try extracted_data
            if not text and 'extracted_data' in sample:
                extracted = sample['extracted_data']
                if isinstance(extracted, dict):
                    # Try to get text from extracted data
                    text = extracted.get('text', '') or extracted.get('content', '') or str(extracted)
                elif isinstance(extracted, str):
                    text = extracted
            
            # If still no text, try chat_format
            if not text and 'chat_format' in sample:
                chat_data = sample['chat_format']
                if isinstance(chat_data, list):
                    # Extract text from chat format
                    for msg in chat_data:
                        if isinstance(msg, dict) and 'content' in msg:
                            text += msg['content'] + " "
                elif isinstance(chat_data, str):
                    text = chat_data
            
            if text and len(text) > 20:  # Only use substantial texts
                # If no explicit label, try to infer from filename
                if not label and 'file_name' in sample:
                    filename = sample['file_name'].lower()
                    if 'invoice' in filename:
                        label = 'invoice'
                    elif 'contract' in filename:
                        label = 'contract'
                    elif 'report' in filename:
                        label = 'report'
                    elif 'letter' in filename:
                        label = 'letter'
                    else:
                        label = 'document'
                
                if label:
                    # Map labels to our classification system
                    mapped_label = map_document_label(label)
                    if mapped_label:
                        document_samples.append({
                            'text': text.strip(),
                            'label': mapped_label,
                            'original_label': label,
                            'filename': sample.get('file_name', '')
                        })
                        
                        if (i + 1) % 100 == 0:
                            print(f"Processed {i + 1} samples...")
        
        print(f"✅ Successfully processed {len(document_samples)} document samples")
        
        # Show label distribution
        label_counts = {}
        for sample in document_samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"\n📊 Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"   {label}: {count} samples")
        
        return document_samples
        
    except Exception as e:
        logger.error(f"Error processing CompanyDocuments data: {str(e)}")
        print(f"Error details: {str(e)}")
        return []

def map_document_label(original_label):
    """Map original labels to our classification system"""
    label_lower = original_label.lower()
    
    # Map to our existing categories
    if any(word in label_lower for word in ['invoice', 'bill', 'receipt']):
        return 'receipt'
    elif any(word in label_lower for word in ['contract', 'agreement', 'legal']):
        return 'contract'
    elif any(word in label_lower for word in ['report', 'financial', 'statement']):
        return 'report'
    elif any(word in label_lower for word in ['letter', 'correspondence', 'email']):
        return 'letter'
    elif any(word in label_lower for word in ['form', 'application', 'document']):
        return 'form'
    else:
        # Keep original label if it doesn't map to existing categories
        return original_label.lower().replace(' ', '_')

def integrate_with_existing_data(document_samples):
    """Integrate CompanyDocuments samples with existing training data"""
    print(f"\n=== INTEGRATING WITH EXISTING DATA ===\n")
    
    # Load existing training data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if os.path.exists(complete_data_path):
        existing_df = pd.read_csv(complete_data_path)
        print(f"Existing training data: {len(existing_df)} samples")
        
        # Show current label distribution
        print(f"\nCurrent label distribution:")
        current_counts = existing_df['label'].value_counts()
        for label, count in current_counts.items():
            print(f"   {label}: {count}")
    else:
        print("❌ Existing training data not found!")
        return False
    
    # Create DataFrame from new document samples
    new_docs_df = pd.DataFrame(document_samples)
    print(f"\nNew document samples to add: {len(new_docs_df)}")
    
    # Combine datasets
    combined_df = pd.concat([existing_df, new_docs_df], ignore_index=True)
    
    # Check final distribution
    print(f"\nFinal dataset: {len(combined_df)} samples")
    final_counts = combined_df['label'].value_counts()
    print(f"\nFinal label distribution:")
    for label, count in final_counts.items():
        print(f"   {label}: {count}")
    
    # Save backup and updated dataset
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_company_docs.csv")
    existing_df.to_csv(backup_path, index=False)
    print(f"\n✅ Backup saved to: {backup_path}")
    
    combined_df.to_csv(complete_data_path, index=False)
    print(f"✅ Updated training data saved to: {complete_data_path}")
    
    return True

def create_company_docs_sample_preview(document_samples, num_samples=5):
    """Create a preview of CompanyDocuments samples"""
    print(f"\n=== COMPANY DOCUMENTS SAMPLES PREVIEW ===\n")
    
    for i, sample in enumerate(document_samples[:num_samples]):
        print(f"📄 Document Sample {i+1} ({sample['label']}):")
        print("-" * 60)
        text_preview = sample['text'][:400] + "..." if len(sample['text']) > 400 else sample['text']
        print(text_preview)
        print(f"\nOriginal Label: {sample['original_label']}")
        print(f"Mapped Label: {sample['label']}")
        print("-" * 60)
        print()

def show_class_distribution_comparison():
    """Show before/after class distribution"""
    print(f"\n=== CLASS DISTRIBUTION COMPARISON ===\n")
    
    # Load updated data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_company_docs.csv")
    
    if os.path.exists(backup_path) and os.path.exists(complete_data_path):
        # Before
        before_df = pd.read_csv(backup_path)
        before_counts = before_df['label'].value_counts()
        
        # After
        after_df = pd.read_csv(complete_data_path)
        after_counts = after_df['label'].value_counts()
        
        print("📊 DOCUMENT CLASSIFICATION IMPROVEMENT:")
        print(f"   Total samples before: {len(before_df)}")
        print(f"   Total samples after:  {len(after_df)}")
        print(f"   New samples added:    {len(after_df) - len(before_df)}")
        
        print(f"\n📈 LABEL-WISE COMPARISON:")
        all_labels = set(before_counts.index) | set(after_counts.index)
        for label in sorted(all_labels):
            before_count = before_counts.get(label, 0)
            after_count = after_counts.get(label, 0)
            added = after_count - before_count
            if added > 0:
                print(f"   {label}: {before_count} → {after_count} (+{added})")
            else:
                print(f"   {label}: {before_count}")

def main():
    """Main function to download and integrate CompanyDocuments dataset"""
    print("🚀 COMPANY DOCUMENTS DATASET INTEGRATION FOR DOCUMENT CLASSIFICATION\n")
    
    # Step 1: Download CompanyDocuments dataset
    dataset = download_company_documents_dataset()
    if not dataset:
        print("❌ Failed to download dataset. Exiting...")
        return
    
    # Step 2: Process for training
    document_samples = process_company_documents_for_training(dataset, max_samples=2000)
    if not document_samples:
        print("❌ Failed to process document samples. Exiting...")
        return
    
    # Step 3: Show preview
    create_company_docs_sample_preview(document_samples)
    
    # Step 4: Integrate with existing data
    success = integrate_with_existing_data(document_samples)
    if success:
        # Step 5: Show comparison
        show_class_distribution_comparison()
        
        print(f"\n🎉 SUCCESS! COMPANY DOCUMENTS DATASET INTEGRATED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Retrain your model:")
        print(f"   python app.py train --enhanced")
        print(f"2. Test document classification:")
        print(f"   python app.py api")
        print(f"3. Expected improvements:")
        print(f"   - Better document type recognition")
        print(f"   - More diverse training examples")
        print(f"   - Improved classification accuracy")
    else:
        print("❌ Failed to integrate dataset")

if __name__ == "__main__":
    main()