#!/usr/bin/env python3
"""
Extract, clean, and integrate archive2.zip company document data
"""
import pandas as pd
import os
import config
import zipfile
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_company_document_csv():
    """Extract the company-document-text.csv from archive2.zip"""
    print("📦 EXTRACTING COMPANY DOCUMENT DATA FROM ARCHIVE2.ZIP\n")
    
    zip_path = os.path.join(config.DATA_DIR, "archive2.zip")
    extract_dir = os.path.join(config.DATA_DIR, "temp", "archive2")
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract the CSV file
            csv_file = "company-document-text.csv"
            zip_ref.extract(csv_file, extract_dir)
            
            csv_path = os.path.join(extract_dir, csv_file)
            print(f"✅ Extracted: {csv_path}")
            
            return csv_path
            
    except Exception as e:
        logger.error(f"Error extracting CSV: {str(e)}")
        return None

def analyze_company_document_data(csv_path):
    """Analyze the structure and quality of the company document data"""
    print("🔍 ANALYZING COMPANY DOCUMENT DATA\n")
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"📊 Dataset Overview:")
        print(f"   Total samples: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Data types: {df.dtypes.to_dict()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"\n❓ Missing values:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"   {col}: {missing} ({missing/len(df)*100:.1f}%)")
            else:
                print(f"   {col}: 0")
        
        # Analyze labels
        print(f"\n🏷️  Label distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"   {label}: {count:,} samples")
        
        # Analyze text quality
        print(f"\n📝 Text quality analysis:")
        df['text_length'] = df['text'].str.len()
        print(f"   Average text length: {df['text_length'].mean():.1f} characters")
        print(f"   Min text length: {df['text_length'].min()}")
        print(f"   Max text length: {df['text_length'].max()}")
        
        # Check for very short texts
        short_texts = len(df[df['text_length'] < 50])
        print(f"   Very short texts (<50 chars): {short_texts} ({short_texts/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return None

def clean_company_document_text(text):
    """Clean and normalize company document text"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Clean up common OCR artifacts and formatting issues
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([\.,:;!?])\s*', r'\1 ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[\.]{3,}', '...', text)
    text = re.sub(r'[\-]{3,}', '---', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def map_archive2_labels(original_label):
    """Map archive2 labels to our existing categories"""
    label_lower = original_label.lower().strip()
    
    # Direct mappings
    label_mappings = {
        'shippingorder': 'shipping_orders',
        'invoice': 'invoice',
        'purchaseorder': 'purchase_order',
        'deliverynote': 'delivery_note',
        'quote': 'quote',
        'receipt': 'receipt',
        'contract': 'contract',
        'report': 'report',
        'statement': 'bank_statement'
    }
    
    # Check direct mappings first
    if label_lower in label_mappings:
        return label_mappings[label_lower]
    
    # Pattern-based mappings
    if any(word in label_lower for word in ['shipping', 'ship']):
        return 'shipping_orders'
    elif any(word in label_lower for word in ['purchase', 'order']):
        return 'purchase_order'
    elif any(word in label_lower for word in ['invoice', 'bill']):
        return 'invoice'
    elif any(word in label_lower for word in ['delivery', 'deliver']):
        return 'delivery_note'
    elif any(word in label_lower for word in ['quote', 'quotation']):
        return 'quote'
    elif any(word in label_lower for word in ['receipt', 'payment']):
        return 'receipt'
    elif any(word in label_lower for word in ['contract', 'agreement']):
        return 'contract'
    elif any(word in label_lower for word in ['report', 'summary']):
        return 'report'
    elif any(word in label_lower for word in ['statement', 'bank']):
        return 'bank_statement'
    else:
        # Keep original but clean it
        return label_lower.replace(' ', '_')

def process_and_clean_archive2_data(df):
    """Process and clean the archive2 data for integration"""
    print("🧹 PROCESSING AND CLEANING ARCHIVE2 DATA\n")
    
    processed_samples = []
    skipped_samples = 0
    
    for idx, row in df.iterrows():
        # Clean the text
        cleaned_text = clean_company_document_text(row['text'])
        
        # Skip if text is too short after cleaning
        if not cleaned_text or len(cleaned_text) < 30:
            skipped_samples += 1
            continue
        
        # Map the label
        mapped_label = map_archive2_labels(row['label'])
        
        processed_samples.append({
            'text': cleaned_text,
            'label': mapped_label,
            'original_label': row['label'],
            'source': 'Archive2',
            'word_count': row.get('word_count', len(cleaned_text.split()))
        })
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1:,} samples...")
    
    print(f"✅ Successfully processed {len(processed_samples):,} samples")
    print(f"⚠️  Skipped {skipped_samples} samples (too short after cleaning)")
    
    # Show label distribution
    label_counts = {}
    for sample in processed_samples:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n📊 Processed label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count:,} samples")
    
    return processed_samples

def show_sample_preview(processed_samples, num_samples=3):
    """Show preview of processed samples"""
    print(f"\n📄 SAMPLE PREVIEW (First {num_samples} samples):\n")
    
    for i, sample in enumerate(processed_samples[:num_samples]):
        print(f"🔸 Sample {i+1} ({sample['label']}):")
        print("-" * 60)
        text_preview = sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text']
        print(text_preview)
        print(f"\nOriginal Label: {sample['original_label']}")
        print(f"Mapped Label: {sample['label']}")
        print(f"Word Count: {sample['word_count']}")
        print("-" * 60)
        print()

def integrate_with_existing_data(processed_samples):
    """Integrate archive2 samples with existing training data"""
    print(f"\n🔄 INTEGRATING WITH EXISTING TRAINING DATA\n")
    
    # Load existing training data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if os.path.exists(complete_data_path):
        existing_df = pd.read_csv(complete_data_path)
        print(f"Existing training data: {len(existing_df):,} samples")
        
        # Show current top categories
        print(f"\nCurrent top categories:")
        current_counts = existing_df['label'].value_counts().head(8)
        for label, count in current_counts.items():
            print(f"   {label}: {count:,}")
    else:
        print("❌ Existing training data not found!")
        return False
    
    # Create DataFrame from new samples
    new_df = pd.DataFrame(processed_samples)
    print(f"\nNew archive2 samples to add: {len(new_df):,}")
    
    # Remove duplicates based on text content
    print("Checking for duplicates...")
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    initial_size = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    final_size = len(combined_df)
    duplicates_removed = initial_size - final_size
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed:,} duplicate samples")
    
    # Check final distribution
    print(f"\nFinal dataset: {len(combined_df):,} samples")
    final_counts = combined_df['label'].value_counts()
    print(f"\nFinal top categories:")
    for label, count in final_counts.head(10).items():
        print(f"   {label}: {count:,}")
    
    # Save backup and updated dataset
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_archive2.csv")
    existing_df.to_csv(backup_path, index=False)
    print(f"\n✅ Backup saved to: {backup_path}")
    
    combined_df.to_csv(complete_data_path, index=False)
    print(f"✅ Updated training data saved to: {complete_data_path}")
    
    return True

def show_integration_summary():
    """Show summary of the integration"""
    print(f"\n📊 INTEGRATION SUMMARY\n")
    
    # Load updated data
    complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_archive2.csv")
    
    if os.path.exists(backup_path) and os.path.exists(complete_data_path):
        # Before
        before_df = pd.read_csv(backup_path)
        before_counts = before_df['label'].value_counts()
        
        # After
        after_df = pd.read_csv(complete_data_path)
        after_counts = after_df['label'].value_counts()
        
        print("🎯 DATASET IMPROVEMENT:")
        print(f"   Total samples before: {len(before_df):,}")
        print(f"   Total samples after:  {len(after_df):,}")
        print(f"   New samples added:    {len(after_df) - len(before_df):,}")
        print(f"   Growth percentage:    {((len(after_df) - len(before_df)) / len(before_df)) * 100:.1f}%")
        
        print(f"\n📈 SIGNIFICANT CHANGES:")
        all_labels = set(before_counts.index) | set(after_counts.index)
        for label in sorted(all_labels):
            before_count = before_counts.get(label, 0)
            after_count = after_counts.get(label, 0)
            if after_count > before_count:
                added = after_count - before_count
                if added >= 10:  # Only show significant additions
                    print(f"   {label}: {before_count:,} → {after_count:,} (+{added:,})")

def main():
    """Main function to process archive2.zip data"""
    print("🚀 ARCHIVE2.ZIP COMPANY DOCUMENTS INTEGRATION\n")
    
    # Step 1: Extract CSV file
    csv_path = extract_company_document_csv()
    if not csv_path:
        print("❌ Failed to extract CSV file. Exiting...")
        return
    
    # Step 2: Analyze the data
    df = analyze_company_document_data(csv_path)
    if df is None:
        print("❌ Failed to analyze data. Exiting...")
        return
    
    # Step 3: Process and clean the data
    processed_samples = process_and_clean_archive2_data(df)
    if not processed_samples:
        print("❌ Failed to process samples. Exiting...")
        return
    
    # Step 4: Show sample preview
    show_sample_preview(processed_samples)
    
    # Step 5: Integrate with existing data
    success = integrate_with_existing_data(processed_samples)
    if success:
        # Step 6: Show integration summary
        show_integration_summary()
        
        print(f"\n🎉 SUCCESS! ARCHIVE2 DATA INTEGRATED!")
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Update visualizations:")
        print(f"   python visualize_dataset_confidence.py")
        print(f"2. Retrain your model:")
        print(f"   python app.py train --enhanced")
        print(f"3. Test improved classification:")
        print(f"   python app.py api")
        print(f"4. Expected improvements:")
        print(f"   - More diverse document samples")
        print(f"   - Better classification accuracy")
        print(f"   - Enhanced model robustness")
    else:
        print("❌ Failed to integrate dataset")

if __name__ == "__main__":
    main()