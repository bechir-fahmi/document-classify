#!/usr/bin/env python3
"""
Generate visualizations for dataset distribution and confidence scores by document type
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import config
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and analyze the training dataset"""
    print("📊 LOADING AND ANALYZING DATASET\n")
    
    data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if not os.path.exists(data_path):
        print("❌ Training data file not found!")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Basic statistics
    print(f"\n📈 Dataset Overview:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Unique labels: {df['label'].nunique()}")
    print(f"   Average text length: {df['text'].str.len().mean():.1f} characters")
    
    return df

def create_dataset_distribution_charts(df):
    """Create separate dataset distribution visualizations"""
    print("📊 Creating dataset distribution charts...")
    
    label_counts = df['label'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(label_counts)))
    
    # 1. Document Type Distribution Bar Chart
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    bars = ax1.bar(range(len(label_counts)), label_counts.values, color=colors)
    ax1.set_title('Document Type Distribution', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Document Types', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xticks(range(len(label_counts)))
    ax1.set_xticklabels(label_counts.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, label_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path1 = os.path.join(config.DATA_DIR, "1_document_type_distribution.png")
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✅ Document type distribution saved to: {output_path1}")
    plt.close()
    
    # 2. Pie Chart for Document Type Proportions
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    major_labels = label_counts.head(8)
    other_count = label_counts.tail(-8).sum()
    if other_count > 0:
        pie_data = major_labels.copy()
        pie_data['Others'] = other_count
    else:
        pie_data = major_labels
    
    wedges, texts, autotexts = ax2.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
                                       startangle=90, colors=colors[:len(pie_data)])
    ax2.set_title('Document Type Proportions', fontsize=16, fontweight='bold', pad=20)
    
    # Make percentage text bold and readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    output_path2 = os.path.join(config.DATA_DIR, "2_document_type_proportions.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Document type proportions saved to: {output_path2}")
    plt.close()
    
    # 3. Text Length Distribution by Document Type
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    top_labels = label_counts.head(6).index
    text_lengths = []
    labels_for_box = []
    
    for label in top_labels:
        lengths = df[df['label'] == label]['text'].str.len()
        text_lengths.append(lengths)
        labels_for_box.append(f"{label}\n({len(lengths)} samples)")
    
    box_plot = ax3.boxplot(text_lengths, labels=labels_for_box, patch_artist=True)
    ax3.set_title('Text Length Distribution by Document Type', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Text Length (characters)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors[:len(text_lengths)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    output_path3 = os.path.join(config.DATA_DIR, "3_text_length_distribution.png")
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✅ Text length distribution saved to: {output_path3}")
    plt.close()
    
    # 4. Dataset Growth Over Time
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    data_sources = {
        'Original Data': 7272,
        '+ Company Docs': 9272,
        '+ SROIE Dataset': 11201
    }
    
    source_names = list(data_sources.keys())
    source_counts = list(data_sources.values())
    
    bars4 = ax4.bar(source_names, source_counts, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax4.set_title('Dataset Growth Over Time', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('Total Samples', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars4, source_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    output_path4 = os.path.join(config.DATA_DIR, "4_dataset_growth.png")
    plt.savefig(output_path4, dpi=300, bbox_inches='tight')
    print(f"✅ Dataset growth chart saved to: {output_path4}")
    plt.close()
    
    return [output_path1, output_path2, output_path3, output_path4]

def train_quick_model_for_confidence(df):
    """Train a quick model to get confidence scores for each document type"""
    print("\n🤖 Training model for confidence analysis...")
    
    # Prepare data
    X = df['text']
    y = df['label']
    
    # Check class distribution and filter out classes with too few samples
    class_counts = y.value_counts()
    print(f"   Classes with sample counts: {len(class_counts)}")
    
    # Keep only classes with at least 2 samples for stratified split
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"   Using {len(valid_classes)} classes with sufficient samples")
    print(f"   Filtered dataset: {len(X_filtered)} samples")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    
    # Vectorize text (limit features for speed)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)
    
    # Calculate confidence scores per class
    confidence_scores = {}
    classes = model.classes_
    
    for i, class_name in enumerate(classes):
        # Get samples that were predicted as this class
        class_mask = y_pred == class_name
        if class_mask.sum() > 0:
            # Average confidence for this class
            class_confidences = y_proba[class_mask, i]
            confidence_scores[class_name] = {
                'mean_confidence': class_confidences.mean(),
                'std_confidence': class_confidences.std(),
                'min_confidence': class_confidences.min(),
                'max_confidence': class_confidences.max(),
                'sample_count': class_mask.sum()
            }
    
    print(f"✅ Model trained with {len(X_train)} training samples")
    
    return model, confidence_scores, y_test, y_pred, classes

def create_confidence_visualizations(confidence_scores, df):
    """Create separate confidence score visualizations"""
    print("📊 Creating confidence score visualizations...")
    
    # Prepare data for plotting
    labels = []
    mean_conf = []
    std_conf = []
    sample_counts = []
    
    for label, scores in confidence_scores.items():
        labels.append(label)
        mean_conf.append(scores['mean_confidence'])
        std_conf.append(scores['std_confidence'])
        sample_counts.append(scores['sample_count'])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    
    # 1. Mean confidence scores with error bars
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    bars1 = ax1.bar(range(len(labels)), mean_conf, yerr=std_conf, capsize=5, color=colors, alpha=0.8)
    ax1.set_title('Average Confidence Score by Document Type', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Document Types', fontsize=12)
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, conf in zip(bars1, mean_conf):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path1 = os.path.join(config.DATA_DIR, "5_confidence_scores.png")
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence scores saved to: {output_path1}")
    plt.close()
    
    # 2. Confidence vs Sample Count scatter plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    scatter = ax2.scatter(sample_counts, mean_conf, c=colors, s=150, alpha=0.7, edgecolors='black')
    ax2.set_title('Confidence vs Sample Count Correlation', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Number of Test Samples', fontsize=12)
    ax2.set_ylabel('Mean Confidence Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add labels to points
    for i, label in enumerate(labels):
        ax2.annotate(label, (sample_counts[i], mean_conf[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path2 = os.path.join(config.DATA_DIR, "6_confidence_vs_samples.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence vs samples correlation saved to: {output_path2}")
    plt.close()
    
    # 3. Confidence score distribution
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    conf_ranges = ['Very High\n(>0.9)', 'High\n(0.8-0.9)', 'Medium\n(0.6-0.8)', 'Low\n(<0.6)']
    range_counts = [0, 0, 0, 0]
    
    for conf in mean_conf:
        if conf > 0.9:
            range_counts[0] += 1
        elif conf > 0.8:
            range_counts[1] += 1
        elif conf > 0.6:
            range_counts[2] += 1
        else:
            range_counts[3] += 1
    
    colors_range = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    bars3 = ax3.bar(conf_ranges, range_counts, color=colors_range, alpha=0.8)
    ax3.set_title('Distribution of Confidence Levels', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Number of Document Types', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars3, range_counts):
        if count > 0:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    output_path3 = os.path.join(config.DATA_DIR, "7_confidence_distribution.png")
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"✅ Confidence distribution saved to: {output_path3}")
    plt.close()
    
    # 4. Top and bottom performers
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    sorted_conf = sorted(zip(labels, mean_conf), key=lambda x: x[1], reverse=True)
    
    # Take all performers, not just top/bottom 5
    performance_labels = [item[0] for item in sorted_conf]
    performance_scores = [item[1] for item in sorted_conf]
    
    # Color based on performance level
    performance_colors = []
    for score in performance_scores:
        if score > 0.9:
            performance_colors.append('#2ecc71')  # Green for very high
        elif score > 0.8:
            performance_colors.append('#3498db')  # Blue for high
        elif score > 0.6:
            performance_colors.append('#f39c12')  # Orange for medium
        else:
            performance_colors.append('#e74c3c')  # Red for low
    
    bars4 = ax4.barh(range(len(performance_labels)), performance_scores, color=performance_colors, alpha=0.8)
    ax4.set_title('Document Type Performance Ranking', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Confidence Score', fontsize=12)
    ax4.set_yticks(range(len(performance_labels)))
    ax4.set_yticklabels(performance_labels)
    ax4.set_xlim(0, 1)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars4, performance_scores):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path4 = os.path.join(config.DATA_DIR, "8_performance_ranking.png")
    plt.savefig(output_path4, dpi=300, bbox_inches='tight')
    print(f"✅ Performance ranking saved to: {output_path4}")
    plt.close()
    
    return [output_path1, output_path2, output_path3, output_path4]

def create_detailed_report(df, confidence_scores):
    """Create a detailed text report of the analysis"""
    print("\n📝 Creating detailed analysis report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATASET AND CONFIDENCE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("📊 DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total samples: {len(df):,}")
    report_lines.append(f"Unique document types: {df['label'].nunique()}")
    report_lines.append(f"Average text length: {df['text'].str.len().mean():.1f} characters")
    report_lines.append("")
    
    # Label distribution
    report_lines.append("📈 DOCUMENT TYPE DISTRIBUTION")
    report_lines.append("-" * 40)
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        report_lines.append(f"{label:25} {count:6,} samples ({percentage:5.1f}%)")
    report_lines.append("")
    
    # Confidence analysis
    report_lines.append("🎯 CONFIDENCE SCORE ANALYSIS")
    report_lines.append("-" * 40)
    
    # Sort by confidence
    sorted_confidence = sorted(confidence_scores.items(), key=lambda x: x[1]['mean_confidence'], reverse=True)
    
    report_lines.append(f"{'Document Type':25} {'Confidence':>10} {'Std Dev':>8} {'Samples':>8}")
    report_lines.append("-" * 60)
    
    for label, scores in sorted_confidence:
        report_lines.append(f"{label:25} {scores['mean_confidence']:>9.3f} {scores['std_confidence']:>7.3f} {scores['sample_count']:>7}")
    
    report_lines.append("")
    
    # Performance categories
    high_conf = [label for label, scores in confidence_scores.items() if scores['mean_confidence'] > 0.8]
    medium_conf = [label for label, scores in confidence_scores.items() if 0.6 <= scores['mean_confidence'] <= 0.8]
    low_conf = [label for label, scores in confidence_scores.items() if scores['mean_confidence'] < 0.6]
    
    report_lines.append("🏆 PERFORMANCE CATEGORIES")
    report_lines.append("-" * 40)
    report_lines.append(f"High Confidence (>0.8): {len(high_conf)} types")
    for label in high_conf:
        report_lines.append(f"  • {label}")
    report_lines.append("")
    
    report_lines.append(f"Medium Confidence (0.6-0.8): {len(medium_conf)} types")
    for label in medium_conf:
        report_lines.append(f"  • {label}")
    report_lines.append("")
    
    report_lines.append(f"Low Confidence (<0.6): {len(low_conf)} types")
    for label in low_conf:
        report_lines.append(f"  • {label}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("💡 RECOMMENDATIONS")
    report_lines.append("-" * 40)
    
    if low_conf:
        report_lines.append("For low confidence document types:")
        report_lines.append("  • Collect more training samples")
        report_lines.append("  • Review data quality and consistency")
        report_lines.append("  • Consider feature engineering")
    
    if len(high_conf) > len(low_conf):
        report_lines.append("Overall model performance looks good!")
        report_lines.append("  • Most document types have high confidence")
        report_lines.append("  • Ready for production use")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(config.DATA_DIR, "dataset_confidence_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Detailed report saved to: {report_path}")
    
    return report_lines

def main():
    """Main function to generate all visualizations and analysis"""
    print("🎨 DATASET VISUALIZATION AND CONFIDENCE ANALYSIS\n")
    
    # Load data
    df = load_and_analyze_data()
    if df is None:
        return
    
    # Create dataset distribution charts (separate files)
    dist_files = create_dataset_distribution_charts(df)
    
    # Train model and get confidence scores
    model, confidence_scores, y_test, y_pred, classes = train_quick_model_for_confidence(df)
    
    # Create confidence visualizations (separate files)
    conf_files = create_confidence_visualizations(confidence_scores, df)
    
    # Create detailed report
    report = create_detailed_report(df, confidence_scores)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"\n📁 Generated files:")
    print(f"\n📊 Dataset Distribution Charts:")
    for i, file_path in enumerate(dist_files, 1):
        filename = os.path.basename(file_path)
        print(f"   {i}. {filename}")
    
    print(f"\n🎯 Confidence Analysis Charts:")
    for i, file_path in enumerate(conf_files, 5):
        filename = os.path.basename(file_path)
        print(f"   {i}. {filename}")
    
    print(f"\n📝 Report:")
    print(f"   9. dataset_confidence_report.txt - Detailed analysis report")
    
    print(f"\n📊 Quick Summary:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Document types: {df['label'].nunique()}")
    print(f"   • Average confidence: {np.mean([scores['mean_confidence'] for scores in confidence_scores.values()]):.3f}")
    
    high_conf_count = len([s for s in confidence_scores.values() if s['mean_confidence'] > 0.8])
    print(f"   • High confidence types: {high_conf_count}/{len(confidence_scores)}")
    
    print(f"\n🎨 All visualizations saved as separate PNG files in the data directory!")

if __name__ == "__main__":
    main()