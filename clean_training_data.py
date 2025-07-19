#!/usr/bin/env python3
"""
Clean and optimize training data based on quality check results
"""
import pandas as pd
import os
import config
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data():
    """Load the training dataset"""
    print("📊 LOADING TRAINING DATASET FOR CLEANING\n")
    
    data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if not os.path.exists(data_path):
        print("❌ Training data file not found!")
        return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} samples")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def clean_text_content(df):
    """Clean text content issues"""
    print("🧹 CLEANING TEXT CONTENT\n")
    
    original_count = len(df)
    
    # Clean non-printable characters
    print("🔤 Removing non-printable characters...")
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\x20-\x7E\n\r\t]', ' ', str(x)))
    
    # Clean OCR artifacts
    print("🔧 Cleaning OCR artifacts...")
    
    # Remove excessive uppercase sequences (likely OCR errors)
    df['text'] = df['text'].apply(lambda x: re.sub(r'\b[A-Z]{10,}\b', ' ', x))
    
    # Clean excessive whitespace
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))
    
    # Remove coordinate patterns (OCR box coordinates)
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d+,\d+,\d+,\d+', ' ', x))
    
    # Clean up common OCR artifacts
    ocr_patterns = [
        r'\[PAD\]|\[unused\d+\]|\[CLS\]|\[SEP\]',  # Model tokens
        r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=]',  # Special chars
        r'\.{3,}',  # Multiple dots
        r'\-{3,}',  # Multiple dashes
    ]
    
    for pattern in ocr_patterns:
        df['text'] = df['text'].apply(lambda x: re.sub(pattern, ' ', x))
    
    # Normalize whitespace
    df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))
    
    # Remove very short texts after cleaning
    min_length = 30
    before_filter = len(df)
    df = df[df['text'].str.len() >= min_length]
    removed_short = before_filter - len(df)
    
    if removed_short > 0:
        print(f"   Removed {removed_short:,} samples that became too short after cleaning")
    
    print(f"✅ Text cleaning complete. Samples: {original_count:,} → {len(df):,}")
    
    return df

def fix_label_issues(df):
    """Fix label-related issues"""
    print("🏷️  FIXING LABEL ISSUES\n")
    
    # Merge similar labels
    print("🔄 Merging similar labels...")
    
    # Map purchase_orders to purchase_order (they're the same thing)
    df['label'] = df['label'].replace('purchase_orders', 'purchase_order')
    
    # Remove classes with very few samples (less than 5)
    print("🗑️  Removing classes with very few samples...")
    
    label_counts = df['label'].value_counts()
    small_classes = label_counts[label_counts < 5].index.tolist()
    
    if small_classes:
        print(f"   Removing classes: {small_classes}")
        df = df[~df['label'].isin(small_classes)]
        removed_samples = sum(label_counts[label] for label in small_classes)
        print(f"   Removed {removed_samples:,} samples from {len(small_classes)} small classes")
    
    # Show final label distribution
    final_counts = df['label'].value_counts()
    print(f"\n📊 Final label distribution:")
    for label, count in final_counts.head(10).items():
        print(f"   {label}: {count:,} samples")
    
    return df

def clean_metadata_columns(df):
    """Clean up metadata columns"""
    print("📋 CLEANING METADATA COLUMNS\n")
    
    # Keep only essential columns for training
    essential_columns = ['text', 'label']
    
    # Keep useful metadata if it exists and has reasonable coverage
    optional_columns = ['source', 'word_count', 'original_label']
    
    columns_to_keep = essential_columns.copy()
    
    for col in optional_columns:
        if col in df.columns:
            missing_percentage = df[col].isnull().sum() / len(df) * 100
            if missing_percentage < 50:  # Keep if less than 50% missing
                columns_to_keep.append(col)
                print(f"   Keeping {col} ({missing_percentage:.1f}% missing)")
            else:
                print(f"   Dropping {col} ({missing_percentage:.1f}% missing)")
    
    # Drop unnecessary columns
    df_cleaned = df[columns_to_keep].copy()
    
    print(f"✅ Kept {len(columns_to_keep)} essential columns: {columns_to_keep}")
    
    return df_cleaned

def balance_dataset(df, max_samples_per_class=2000):
    """Balance the dataset to prevent severe imbalance"""
    print("⚖️  BALANCING DATASET\n")
    
    label_counts = df['label'].value_counts()
    print("Current distribution:")
    for label, count in label_counts.head(10).items():
        print(f"   {label}: {count:,} samples")
    
    # Cap very large classes
    balanced_dfs = []
    
    for label in label_counts.index:
        label_data = df[df['label'] == label]
        
        if len(label_data) > max_samples_per_class:
            # Sample randomly to reduce size
            label_data = label_data.sample(n=max_samples_per_class, random_state=42)
            print(f"   Capped {label}: {label_counts[label]:,} → {max_samples_per_class:,}")
        
        balanced_dfs.append(label_data)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Balanced dataset: {len(df):,} → {len(balanced_df):,} samples")
    
    # Show final distribution
    final_counts = balanced_df['label'].value_counts()
    print(f"\nFinal balanced distribution:")
    for label, count in final_counts.items():
        print(f"   {label}: {count:,} samples")
    
    return balanced_df

def validate_cleaned_data(df):
    """Validate the cleaned dataset"""
    print("✅ VALIDATING CLEANED DATASET\n")
    
    issues = []
    
    # Check basic integrity
    if df['text'].isnull().sum() > 0:
        issues.append("Null texts found")
    
    if df['label'].isnull().sum() > 0:
        issues.append("Null labels found")
    
    # Check text quality
    avg_length = df['text'].str.len().mean()
    min_length = df['text'].str.len().min()
    
    print(f"📊 Validation Results:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Unique labels: {df['label'].nunique()}")
    print(f"   Average text length: {avg_length:.1f} characters")
    print(f"   Minimum text length: {min_length}")
    
    # Check for remaining issues
    if min_length < 20:
        issues.append("Very short texts still present")
    
    if avg_length < 100:
        issues.append("Average text length seems low")
    
    # Check label balance
    label_counts = df['label'].value_counts()
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"   Label balance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 50:
        issues.append("Severe imbalance still present")
    
    if issues:
        print(f"\n⚠️  Remaining issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ Dataset validation passed - ready for training!")
    
    return len(issues) == 0

def save_cleaned_dataset(df):
    """Save the cleaned dataset"""
    print("💾 SAVING CLEANED DATASET\n")
    
    # Create backup of original
    original_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    backup_path = os.path.join(config.SAMPLE_DIR, "complete_training_data_backup_before_cleaning_final.csv")
    
    if os.path.exists(original_path):
        original_df = pd.read_csv(original_path)
        original_df.to_csv(backup_path, index=False)
        print(f"✅ Backup saved: {backup_path}")
    
    # Save cleaned dataset
    df.to_csv(original_path, index=False)
    print(f"✅ Cleaned dataset saved: {original_path}")
    
    # Save a summary
    summary_path = os.path.join(config.DATA_DIR, "cleaning_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("DATASET CLEANING SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Final dataset size: {len(df):,} samples\n")
        f.write(f"Number of labels: {df['label'].nunique()}\n")
        f.write(f"Average text length: {df['text'].str.len().mean():.1f} characters\n\n")
        f.write("Label distribution:\n")
        for label, count in df['label'].value_counts().items():
            f.write(f"  {label}: {count:,} samples\n")
    
    print(f"✅ Cleaning summary saved: {summary_path}")

def main():
    """Main function to clean the training dataset"""
    print("🧹 COMPREHENSIVE TRAINING DATA CLEANING\n")
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    print(f"Starting with {len(df):,} samples\n")
    
    # Step 1: Clean text content
    df = clean_text_content(df)
    
    # Step 2: Fix label issues
    df = fix_label_issues(df)
    
    # Step 3: Clean metadata columns
    df = clean_metadata_columns(df)
    
    # Step 4: Balance dataset
    df = balance_dataset(df)
    
    # Step 5: Validate cleaned data
    is_valid = validate_cleaned_data(df)
    
    # Step 6: Save cleaned dataset
    save_cleaned_dataset(df)
    
    print(f"\n🎉 CLEANING COMPLETE!")
    print(f"   Final dataset: {len(df):,} samples")
    print(f"   Labels: {df['label'].nunique()}")
    print(f"   Status: {'✅ Ready for training!' if is_valid else '⚠️  Some issues remain'}")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Run quality check again:")
    print(f"   python check_data_quality.py")
    print(f"2. Update visualizations:")
    print(f"   python visualize_dataset_confidence.py")
    print(f"3. Retrain your model:")
    print(f"   python app.py train --enhanced")

if __name__ == "__main__":
    main()