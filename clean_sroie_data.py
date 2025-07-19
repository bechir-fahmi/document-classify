#!/usr/bin/env python3
"""
Clean and improve SROIE data quality by converting JSON format to readable text
"""
import pandas as pd
import os
import config
import json
import re

def clean_json_receipt_samples(df):
    """Convert JSON format receipt samples to readable text"""
    print("🧹 CLEANING JSON RECEIPT SAMPLES\n")
    
    # Find JSON format samples
    json_mask = df['text'].str.contains(r'^\s*{', na=False)
    json_samples = df[json_mask].copy()
    
    print(f"Found {len(json_samples)} JSON format samples to clean")
    
    cleaned_samples = []
    failed_samples = []
    
    for idx, row in json_samples.iterrows():
        try:
            # Parse JSON
            json_text = row['text'].strip()
            data = json.loads(json_text)
            
            # Extract information and create readable text
            readable_text = create_readable_receipt_text(data)
            
            if readable_text and len(readable_text) > 20:
                cleaned_samples.append({
                    'index': idx,
                    'original': json_text,
                    'cleaned': readable_text,
                    'label': row['label']
                })
            else:
                failed_samples.append(idx)
                
        except (json.JSONDecodeError, Exception) as e:
            failed_samples.append(idx)
            continue
    
    print(f"✅ Successfully cleaned: {len(cleaned_samples)} samples")
    print(f"❌ Failed to clean: {len(failed_samples)} samples")
    
    return cleaned_samples, failed_samples

def create_readable_receipt_text(data):
    """Convert JSON receipt data to readable text format"""
    text_parts = []
    
    # Company name
    if 'company' in data and data['company']:
        text_parts.append(f"Company: {data['company']}")
    
    # Date
    if 'date' in data and data['date']:
        text_parts.append(f"Date: {data['date']}")
    
    # Address
    if 'address' in data and data['address']:
        text_parts.append(f"Address: {data['address']}")
    
    # Total amount
    if 'total' in data and data['total']:
        total = data['total']
        if isinstance(total, str) and not total.startswith('RM'):
            total = f"RM{total}"
        text_parts.append(f"Total: {total}")
    
    # Items if available
    if 'items' in data and isinstance(data['items'], list):
        text_parts.append("Items:")
        for item in data['items'][:10]:  # Limit to first 10 items
            if isinstance(item, dict):
                item_text = []
                if 'name' in item:
                    item_text.append(item['name'])
                if 'quantity' in item:
                    item_text.append(f"Qty: {item['quantity']}")
                if 'price' in item:
                    item_text.append(f"Price: {item['price']}")
                if item_text:
                    text_parts.append(f"- {' | '.join(item_text)}")
    
    # Additional fields
    for key, value in data.items():
        if key not in ['company', 'date', 'address', 'total', 'items'] and value:
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key.title()}: {value}")
    
    return '\n'.join(text_parts)

def apply_cleaning_to_dataframe(df, cleaned_samples):
    """Apply the cleaning changes to the dataframe"""
    print(f"\n🔄 APPLYING CLEANING CHANGES\n")
    
    df_cleaned = df.copy()
    
    # Update the cleaned samples
    for sample in cleaned_samples:
        idx = sample['index']
        df_cleaned.at[idx, 'text'] = sample['cleaned']
    
    print(f"✅ Updated {len(cleaned_samples)} samples with cleaned text")
    
    return df_cleaned

def show_cleaning_examples(cleaned_samples, num_examples=3):
    """Show examples of before/after cleaning"""
    print(f"\n📄 CLEANING EXAMPLES (Before → After):\n")
    
    for i, sample in enumerate(cleaned_samples[:num_examples]):
        print(f"🔸 Example {i+1}:")
        print("BEFORE (JSON):")
        print(sample['original'][:200] + "..." if len(sample['original']) > 200 else sample['original'])
        print("\nAFTER (Readable):")
        print(sample['cleaned'][:300] + "..." if len(sample['cleaned']) > 300 else sample['cleaned'])
        print("-" * 60)

def validate_cleaning_results(df_original, df_cleaned):
    """Validate that cleaning didn't break anything"""
    print(f"\n✅ VALIDATION RESULTS:\n")
    
    # Check sample counts
    print(f"Original samples: {len(df_original)}")
    print(f"Cleaned samples: {len(df_cleaned)}")
    print(f"Sample count preserved: {'✅' if len(df_original) == len(df_cleaned) else '❌'}")
    
    # Check label distribution
    original_labels = df_original['label'].value_counts()
    cleaned_labels = df_cleaned['label'].value_counts()
    
    print(f"\nLabel distribution preserved:")
    for label in original_labels.index:
        orig_count = original_labels[label]
        clean_count = cleaned_labels.get(label, 0)
        status = "✅" if orig_count == clean_count else "❌"
        print(f"   {label}: {orig_count} → {clean_count} {status}")
    
    # Check receipt samples specifically
    orig_receipts = len(df_original[df_original['label'] == 'receipt'])
    clean_receipts = len(df_cleaned[df_cleaned['label'] == 'receipt'])
    print(f"\nReceipt samples: {orig_receipts} → {clean_receipts}")
    
    # Check for JSON samples remaining
    remaining_json = len(df_cleaned[df_cleaned['text'].str.contains(r'^\s*{', na=False)])
    print(f"Remaining JSON samples: {remaining_json}")

def main():
    """Main function to clean SROIE data"""
    print("🧹 SROIE DATA CLEANING FOR IMPROVED TRAINING QUALITY\n")
    
    # Load the training data
    data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if not os.path.exists(data_path):
        print("❌ Training data file not found!")
        return
    
    print(f"📂 Loading data from: {data_path}")
    df_original = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df_original)} samples")
    
    # Clean JSON receipt samples
    cleaned_samples, failed_samples = clean_json_receipt_samples(df_original)
    
    if not cleaned_samples:
        print("ℹ️  No samples to clean. Data is already in good format!")
        return
    
    # Show cleaning examples
    show_cleaning_examples(cleaned_samples)
    
    # Apply cleaning to dataframe
    df_cleaned = apply_cleaning_to_dataframe(df_original, cleaned_samples)
    
    # Validate results
    validate_cleaning_results(df_original, df_cleaned)
    
    # Save cleaned data
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_cleaning.csv")
    df_original.to_csv(backup_path, index=False)
    print(f"\n💾 Backup saved to: {backup_path}")
    
    df_cleaned.to_csv(data_path, index=False)
    print(f"💾 Cleaned data saved to: {data_path}")
    
    print(f"\n🎉 DATA CLEANING COMPLETED!")
    print(f"\n📋 SUMMARY:")
    print(f"   • Cleaned {len(cleaned_samples)} JSON samples")
    print(f"   • Converted to readable receipt format")
    print(f"   • Preserved all {len(df_cleaned)} samples")
    print(f"   • Ready for model training!")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Retrain your model: python app.py train --enhanced")
    print(f"   2. Test improved receipt detection")
    print(f"   3. Expected better accuracy with cleaner data")

if __name__ == "__main__":
    main()