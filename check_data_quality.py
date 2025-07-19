#!/usr/bin/env python3
"""
Comprehensive data quality check for training dataset
"""
import pandas as pd
import os
import config
import re
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def load_training_data():
    """Load the complete training dataset"""
    print("📊 LOADING TRAINING DATASET FOR QUALITY CHECK\n")
    
    data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if not os.path.exists(data_path):
        print("❌ Training data file not found!")
        return None
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} samples from training dataset")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def check_basic_data_integrity(df):
    """Check basic data integrity issues"""
    print("🔍 CHECKING BASIC DATA INTEGRITY\n")
    
    issues = []
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("❓ Missing Values Check:")
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   ❌ {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")
            issues.append(f"Missing values in {col}")
        else:
            print(f"   ✅ {col}: No missing values")
    
    # Check for empty strings
    print(f"\n📝 Empty Text Check:")
    empty_texts = len(df[df['text'].str.strip() == ''])
    if empty_texts > 0:
        print(f"   ❌ Empty text fields: {empty_texts:,}")
        issues.append("Empty text fields found")
    else:
        print(f"   ✅ No empty text fields")
    
    # Check for very short texts
    short_texts = len(df[df['text'].str.len() < 20])
    print(f"   📏 Very short texts (<20 chars): {short_texts:,} ({short_texts/len(df)*100:.1f}%)")
    if short_texts > len(df) * 0.05:  # More than 5%
        issues.append("Too many very short texts")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['text']).sum()
    print(f"\n🔄 Duplicate Check:")
    if duplicates > 0:
        print(f"   ❌ Duplicate texts: {duplicates:,}")
        issues.append("Duplicate texts found")
    else:
        print(f"   ✅ No duplicate texts")
    
    return issues

def analyze_text_quality(df):
    """Analyze the quality of text content"""
    print("\n📝 ANALYZING TEXT QUALITY\n")
    
    issues = []
    
    # Text length analysis
    df['text_length'] = df['text'].str.len()
    print("📏 Text Length Analysis:")
    print(f"   Average length: {df['text_length'].mean():.1f} characters")
    print(f"   Median length: {df['text_length'].median():.1f} characters")
    print(f"   Min length: {df['text_length'].min()}")
    print(f"   Max length: {df['text_length'].max()}")
    
    # Check for extremely long texts (potential data issues)
    very_long = len(df[df['text_length'] > 5000])
    if very_long > 0:
        print(f"   ⚠️  Very long texts (>5000 chars): {very_long}")
        if very_long > len(df) * 0.01:  # More than 1%
            issues.append("Too many extremely long texts")
    
    # Check for non-printable characters
    print(f"\n🔤 Character Quality Check:")
    non_printable_count = 0
    for idx, text in df['text'].head(1000).items():  # Check first 1000 samples
        if re.search(r'[^\x20-\x7E\n\r\t]', str(text)):
            non_printable_count += 1
    
    if non_printable_count > 0:
        print(f"   ⚠️  Samples with non-printable chars: {non_printable_count}/1000 checked")
        if non_printable_count > 50:
            issues.append("Many samples contain non-printable characters")
    else:
        print(f"   ✅ No non-printable characters in checked samples")
    
    # Check for meaningful content
    print(f"\n💭 Content Meaningfulness Check:")
    meaningful_samples = 0
    for text in df['text'].head(1000):
        # Check if text has at least 3 words with 3+ characters
        words = re.findall(r'\b[A-Za-z]{3,}\b', str(text))
        if len(words) >= 3:
            meaningful_samples += 1
    
    meaningful_percentage = (meaningful_samples / 1000) * 100
    print(f"   Meaningful samples: {meaningful_samples}/1000 ({meaningful_percentage:.1f}%)")
    
    if meaningful_percentage < 90:
        issues.append("Low percentage of meaningful text content")
    
    return issues

def analyze_label_quality(df):
    """Analyze the quality and distribution of labels"""
    print("\n🏷️  ANALYZING LABEL QUALITY\n")
    
    issues = []
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print("📊 Label Distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count:,} samples ({percentage:.1f}%)")
    
    # Check for label imbalance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n⚖️  Label Balance Analysis:")
    print(f"   Most common label: {label_counts.index[0]} ({max_count:,} samples)")
    print(f"   Least common label: {label_counts.index[-1]} ({min_count:,} samples)")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 100:
        issues.append("Severe label imbalance detected")
    elif imbalance_ratio > 20:
        issues.append("Moderate label imbalance detected")
    
    # Check for very small classes
    small_classes = len(label_counts[label_counts < 10])
    if small_classes > 0:
        print(f"   ⚠️  Classes with <10 samples: {small_classes}")
        issues.append("Classes with very few samples")
    
    # Check for inconsistent label formatting
    print(f"\n🔤 Label Format Check:")
    label_issues = []
    for label in label_counts.index:
        if ' ' in label and '_' in label:
            label_issues.append(f"Mixed separators: '{label}'")
        elif label != label.lower():
            label_issues.append(f"Not lowercase: '{label}'")
        elif re.search(r'[^a-z0-9_\s]', label):
            label_issues.append(f"Special characters: '{label}'")
    
    if label_issues:
        print(f"   ⚠️  Label formatting issues:")
        for issue in label_issues[:5]:  # Show first 5
            print(f"      • {issue}")
        if len(label_issues) > 5:
            print(f"      ... and {len(label_issues) - 5} more")
        issues.append("Label formatting inconsistencies")
    else:
        print(f"   ✅ Label formatting looks consistent")
    
    return issues

def check_data_sources_quality(df):
    """Check quality by data source if source information is available"""
    print("\n📦 CHECKING DATA SOURCES QUALITY\n")
    
    issues = []
    
    # Check if we have source information
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print("📊 Data Sources:")
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {source}: {count:,} samples ({percentage:.1f}%)")
        
        # Analyze quality by source
        print(f"\n🔍 Quality by Source:")
        for source in source_counts.index:
            source_data = df[df['source'] == source]
            avg_length = source_data['text'].str.len().mean()
            print(f"   {source}: Avg length {avg_length:.1f} chars")
    else:
        print("ℹ️  No source information available in dataset")
    
    return issues

def detect_potential_data_contamination(df):
    """Detect potential data contamination issues"""
    print("\n🧪 DETECTING POTENTIAL DATA CONTAMINATION\n")
    
    issues = []
    
    # Check for repeated patterns that might indicate data issues
    print("🔍 Pattern Analysis:")
    
    # Check for texts that are too similar (potential near-duplicates)
    sample_size = min(1000, len(df))
    sample_texts = df['text'].head(sample_size).tolist()
    
    similar_pairs = 0
    for i in range(min(100, len(sample_texts))):
        for j in range(i+1, min(i+10, len(sample_texts))):
            # Simple similarity check based on common words
            words1 = set(sample_texts[i].lower().split())
            words2 = set(sample_texts[j].lower().split())
            if len(words1) > 0 and len(words2) > 0:
                similarity = len(words1 & words2) / len(words1 | words2)
                if similarity > 0.8:  # 80% similarity
                    similar_pairs += 1
    
    if similar_pairs > 5:
        print(f"   ⚠️  Found {similar_pairs} highly similar text pairs")
        issues.append("Potential near-duplicate texts")
    else:
        print(f"   ✅ No significant text similarity issues detected")
    
    # Check for common OCR artifacts
    ocr_artifacts = [
        r'\d+,\d+,\d+,\d+',  # Coordinate patterns
        r'\[PAD\]|\[unused\d+\]',  # Model tokens
        r'[A-Z]{10,}',  # Long uppercase sequences
        r'\s{5,}',  # Excessive whitespace
    ]
    
    print(f"\n🔧 OCR Artifact Check:")
    artifact_counts = {}
    for pattern in ocr_artifacts:
        count = df['text'].str.contains(pattern, na=False).sum()
        if count > 0:
            artifact_counts[pattern] = count
    
    if artifact_counts:
        print(f"   ⚠️  OCR artifacts found:")
        for pattern, count in artifact_counts.items():
            print(f"      • Pattern '{pattern}': {count} samples")
        issues.append("OCR artifacts detected")
    else:
        print(f"   ✅ No OCR artifacts detected")
    
    return issues

def generate_data_quality_report(df, all_issues):
    """Generate a comprehensive data quality report"""
    print("\n📋 GENERATING DATA QUALITY REPORT\n")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TRAINING DATA QUALITY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("📊 DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total samples: {len(df):,}")
    report_lines.append(f"Unique labels: {df['label'].nunique()}")
    report_lines.append(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    report_lines.append(f"Dataset size on disk: ~{len(df) * df['text'].str.len().mean() / 1024 / 1024:.1f} MB")
    report_lines.append("")
    
    # Quality assessment
    report_lines.append("🎯 QUALITY ASSESSMENT")
    report_lines.append("-" * 40)
    
    if not all_issues:
        report_lines.append("✅ EXCELLENT: No significant data quality issues detected!")
        report_lines.append("   Your dataset is clean and ready for training.")
        quality_score = "A+ (Excellent)"
    elif len(all_issues) <= 2:
        report_lines.append("✅ GOOD: Minor issues detected, but dataset is training-ready.")
        quality_score = "A (Good)"
    elif len(all_issues) <= 5:
        report_lines.append("⚠️  FAIR: Some issues detected, consider cleaning before training.")
        quality_score = "B (Fair)"
    else:
        report_lines.append("❌ POOR: Multiple issues detected, cleaning recommended.")
        quality_score = "C (Needs Improvement)"
    
    report_lines.append(f"Overall Quality Score: {quality_score}")
    report_lines.append("")
    
    # Issues found
    if all_issues:
        report_lines.append("⚠️  ISSUES DETECTED")
        report_lines.append("-" * 40)
        for i, issue in enumerate(all_issues, 1):
            report_lines.append(f"{i}. {issue}")
        report_lines.append("")
    
    # Label distribution
    report_lines.append("📊 LABEL DISTRIBUTION")
    report_lines.append("-" * 40)
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        report_lines.append(f"{label:25} {count:6,} samples ({percentage:5.1f}%)")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("💡 RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if not all_issues:
        report_lines.append("• Your dataset is excellent and ready for training!")
        report_lines.append("• Proceed with model training using this high-quality data.")
    else:
        if "Duplicate texts found" in all_issues:
            report_lines.append("• Remove duplicate texts to prevent overfitting")
        if "Empty text fields found" in all_issues:
            report_lines.append("• Remove or fix empty text entries")
        if "OCR artifacts detected" in all_issues:
            report_lines.append("• Clean OCR artifacts for better text quality")
        if "Label formatting inconsistencies" in all_issues:
            report_lines.append("• Standardize label formatting (lowercase, underscores)")
        if any("imbalance" in issue for issue in all_issues):
            report_lines.append("• Consider data augmentation for underrepresented classes")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(config.DATA_DIR, "data_quality_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Quality report saved to: {report_path}")
    
    # Print summary
    print("\n" + "\n".join(report_lines))
    
    return quality_score, all_issues

def main():
    """Main function to check data quality"""
    print("🔍 COMPREHENSIVE TRAINING DATA QUALITY CHECK\n")
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    all_issues = []
    
    # Run all quality checks
    print("=" * 60)
    issues1 = check_basic_data_integrity(df)
    all_issues.extend(issues1)
    
    print("=" * 60)
    issues2 = analyze_text_quality(df)
    all_issues.extend(issues2)
    
    print("=" * 60)
    issues3 = analyze_label_quality(df)
    all_issues.extend(issues3)
    
    print("=" * 60)
    issues4 = check_data_sources_quality(df)
    all_issues.extend(issues4)
    
    print("=" * 60)
    issues5 = detect_potential_data_contamination(df)
    all_issues.extend(issues5)
    
    print("=" * 60)
    
    # Generate final report
    quality_score, final_issues = generate_data_quality_report(df, all_issues)
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"   Quality Score: {quality_score}")
    print(f"   Issues Found: {len(final_issues)}")
    
    if len(final_issues) == 0:
        print(f"   Status: ✅ READY FOR TRAINING!")
    elif len(final_issues) <= 2:
        print(f"   Status: ✅ GOOD TO GO (minor issues)")
    else:
        print(f"   Status: ⚠️  CONSIDER CLEANING FIRST")

if __name__ == "__main__":
    main()