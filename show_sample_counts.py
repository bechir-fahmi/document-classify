#!/usr/bin/env python3
"""
Show detailed sample counts by document type
"""
import pandas as pd
import os
import config

def show_detailed_sample_counts():
    """Show detailed breakdown of samples by document type"""
    print("📊 DETAILED SAMPLE COUNT BY DOCUMENT TYPE")
    print("=" * 60)
    
    # Load the training data
    data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
    
    if not os.path.exists(data_path):
        print("❌ Training data file not found!")
        return
    
    try:
        df = pd.read_csv(data_path)
        
        # Get label counts
        label_counts = df['label'].value_counts()
        total_samples = len(df)
        
        print(f"🎯 TOTAL SAMPLES: {total_samples:,}")
        print(f"🏷️  UNIQUE DOCUMENT TYPES: {len(label_counts)}")
        print("=" * 60)
        
        # Show detailed breakdown
        print("RANK | DOCUMENT TYPE        | SAMPLES  | PERCENTAGE")
        print("-" * 60)
        
        for i, (label, count) in enumerate(label_counts.items(), 1):
            percentage = (count / total_samples) * 100
            print(f"{i:4d} | {label:<20} | {count:>7,} | {percentage:>8.1f}%")
        
        print("-" * 60)
        print(f"     | {'TOTAL':<20} | {total_samples:>7,} | {100.0:>8.1f}%")
        print("=" * 60)
        
        # Show categories by size
        print("\n📈 CATEGORIES BY SIZE:")
        print("-" * 40)
        
        large_categories = label_counts[label_counts >= 1000]
        medium_categories = label_counts[(label_counts >= 100) & (label_counts < 1000)]
        small_categories = label_counts[label_counts < 100]
        
        if len(large_categories) > 0:
            print(f"🔵 LARGE (≥1,000 samples): {len(large_categories)} types")
            for label, count in large_categories.items():
                print(f"   • {label}: {count:,}")
        
        if len(medium_categories) > 0:
            print(f"\n🟡 MEDIUM (100-999 samples): {len(medium_categories)} types")
            for label, count in medium_categories.items():
                print(f"   • {label}: {count:,}")
        
        if len(small_categories) > 0:
            print(f"\n🔴 SMALL (<100 samples): {len(small_categories)} types")
            for label, count in small_categories.items():
                print(f"   • {label}: {count:,}")
        
        # Show balance analysis
        print(f"\n⚖️  BALANCE ANALYSIS:")
        print("-" * 40)
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"Most samples: {label_counts.index[0]} ({max_count:,})")
        print(f"Least samples: {label_counts.index[-1]} ({min_count:,})")
        print(f"Balance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio <= 10:
            balance_status = "✅ EXCELLENT"
        elif imbalance_ratio <= 50:
            balance_status = "✅ GOOD"
        elif imbalance_ratio <= 100:
            balance_status = "⚠️  FAIR"
        else:
            balance_status = "❌ POOR"
        
        print(f"Balance status: {balance_status}")
        
        # Show training readiness
        print(f"\n🎯 TRAINING READINESS:")
        print("-" * 40)
        
        ready_categories = len(label_counts[label_counts >= 50])
        total_categories = len(label_counts)
        
        print(f"Categories ready for training (≥50 samples): {ready_categories}/{total_categories}")
        print(f"Total training samples: {total_samples:,}")
        print(f"Average samples per category: {total_samples/total_categories:.0f}")
        
        if ready_categories == total_categories and total_samples >= 5000:
            readiness = "✅ EXCELLENT - Ready for production training!"
        elif ready_categories >= total_categories * 0.8 and total_samples >= 3000:
            readiness = "✅ GOOD - Ready for training!"
        elif ready_categories >= total_categories * 0.6 and total_samples >= 1000:
            readiness = "⚠️  FAIR - Can train but consider more data"
        else:
            readiness = "❌ POOR - Need more data"
        
        print(f"Overall readiness: {readiness}")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")

if __name__ == "__main__":
    show_detailed_sample_counts()