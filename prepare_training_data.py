import os
import subprocess
import sys

def install_requirements():
    """Install required packages for data generation and processing"""
    print("Installing required packages...")
    
    requirements = [
        "pandas",
        "requests",
        "tqdm",
        "scikit-learn",
        "nltk"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")
            return False
    
    return True

def run_data_generation():
    """Run the data generation scripts"""
    scripts = [
        "download_datasets.py",
        "generate_purchase_orders.py"
    ]
    
    for script in scripts:
        print(f"\nRunning {script}...")
        try:
            subprocess.check_call([sys.executable, script])
            print(f"Successfully ran {script}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {str(e)}")
            return False
    
    return True

def create_merged_dataset():
    """Merge all generated datasets into a single training file"""
    print("\nMerging all datasets...")
    try:
        import pandas as pd
        from pathlib import Path
        
        dfs = []
        
        # Load existing CSV files
        for csv_file in Path('data/samples').glob('**/*.csv'):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                if 'text' in df.columns and 'label' in df.columns:
                    print(f"Adding {len(df)} records from {csv_file}")
                    dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {str(e)}")
        
        if dfs:
            # Merge all dataframes
            merged_df = pd.concat(dfs, ignore_index=True)
            
            # Save merged dataset
            output_file = 'data/samples/complete_training_data.csv'
            merged_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nCreated merged dataset with {len(merged_df)} records at {output_file}")
            
            # Show class distribution
            class_counts = merged_df['label'].value_counts()
            print("\nClass distribution:")
            for label, count in class_counts.items():
                print(f"  {label}: {count} samples")
                
            return True
        else:
            print("No data to merge")
            return False
            
    except Exception as e:
        print(f"Error creating merged dataset: {str(e)}")
        return False

def main():
    """Main function to prepare training data"""
    print("=== Preparing Training Data ===\n")
    
    # Create necessary directories
    os.makedirs('data/samples/enhanced', exist_ok=True)
    os.makedirs('data/samples/downloaded', exist_ok=True)
    os.makedirs('data/samples/synthetic', exist_ok=True)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install required packages. Exiting.")
        return
    
    # Run data generation scripts
    if not run_data_generation():
        print("Failed to generate all data. Continuing with what we have.")
    
    # Create merged dataset
    create_merged_dataset()
    
    print("\n=== Training Data Preparation Complete ===")

if __name__ == "__main__":
    main() 