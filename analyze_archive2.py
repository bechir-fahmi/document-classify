#!/usr/bin/env python3
"""
Analyze and process archive2.zip dataset
"""
import zipfile
import os
import pandas as pd
import json
import config

def analyze_archive2_contents():
    """Analyze what's inside archive2.zip"""
    print("📦 ANALYZING ARCHIVE2.ZIP CONTENTS\n")
    
    zip_path = os.path.join(config.DATA_DIR, "archive2.zip")
    
    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            
            print(f"📊 Archive Analysis:")
            print(f"   Total files: {len(files)}")
            
            # Categorize files by extension
            file_types = {}
            for file in files:
                if '.' in file:
                    ext = file.split('.')[-1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
                else:
                    file_types['no_extension'] = file_types.get('no_extension', 0) + 1
            
            print(f"\n📁 File types:")
            for ext, count in sorted(file_types.items()):
                print(f"   .{ext}: {count} files")
            
            print(f"\n📄 First 20 files:")
            for i, file in enumerate(files[:20]):
                print(f"   {i+1:2d}. {file}")
            
            if len(files) > 20:
                print(f"   ... and {len(files) - 20} more files")
            
            # Look for specific patterns
            csv_files = [f for f in files if f.endswith('.csv')]
            json_files = [f for f in files if f.endswith('.json')]
            txt_files = [f for f in files if f.endswith('.txt')]
            
            print(f"\n🎯 Data files found:")
            print(f"   CSV files: {len(csv_files)}")
            print(f"   JSON files: {len(json_files)}")
            print(f"   TXT files: {len(txt_files)}")
            
            if csv_files:
                print(f"\n📊 CSV files:")
                for csv_file in csv_files[:10]:
                    print(f"   • {csv_file}")
            
            if json_files:
                print(f"\n📋 JSON files:")
                for json_file in json_files[:10]:
                    print(f"   • {json_file}")
            
            return files, file_types
            
    except Exception as e:
        print(f"❌ Error analyzing archive: {str(e)}")
        return None, None

def extract_and_preview_sample_files(files):
    """Extract and preview some sample files to understand the data structure"""
    print(f"\n🔍 EXTRACTING SAMPLE FILES FOR ANALYSIS\n")
    
    zip_path = os.path.join(config.DATA_DIR, "archive2.zip")
    extract_dir = os.path.join(config.DATA_DIR, "temp", "archive2_preview")
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract a few sample files for analysis
            sample_files = []
            
            # Get some CSV files
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                sample_files.extend(csv_files[:3])
            
            # Get some JSON files
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                sample_files.extend(json_files[:3])
            
            # Get some TXT files
            txt_files = [f for f in files if f.endswith('.txt')]
            if txt_files:
                sample_files.extend(txt_files[:3])
            
            print(f"Extracting {len(sample_files)} sample files for analysis...")
            
            for file in sample_files:
                try:
                    zip_ref.extract(file, extract_dir)
                    print(f"✅ Extracted: {file}")
                except Exception as e:
                    print(f"❌ Failed to extract {file}: {str(e)}")
            
            # Analyze extracted files
            analyze_extracted_files(extract_dir, sample_files)
            
    except Exception as e:
        print(f"❌ Error extracting files: {str(e)}")

def analyze_extracted_files(extract_dir, sample_files):
    """Analyze the structure of extracted sample files"""
    print(f"\n📋 ANALYZING EXTRACTED FILES\n")
    
    for file in sample_files:
        file_path = os.path.join(extract_dir, file)
        
        if not os.path.exists(file_path):
            continue
            
        print(f"🔸 Analyzing: {file}")
        print("-" * 50)
        
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"   CSV Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Sample data:")
                print(df.head(2).to_string())
                
            elif file.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   JSON Type: {type(data)}")
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())}")
                    for key, value in list(data.items())[:3]:
                        print(f"   {key}: {type(value)} - {str(value)[:100]}...")
                elif isinstance(data, list):
                    print(f"   List Length: {len(data)}")
                    if data:
                        print(f"   First item: {type(data[0])} - {str(data[0])[:100]}...")
                        
            elif file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                print(f"   Text Length: {len(content)} characters")
                print(f"   Preview: {content[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Error analyzing file: {str(e)}")
        
        print("-" * 50)
        print()

def main():
    """Main function to analyze archive2.zip"""
    print("🔍 ARCHIVE2.ZIP ANALYSIS TOOL\n")
    
    # Step 1: Analyze archive contents
    files, file_types = analyze_archive2_contents()
    
    if not files:
        print("❌ Could not analyze archive contents")
        return
    
    # Step 2: Extract and preview sample files
    extract_and_preview_sample_files(files)
    
    print(f"\n📋 ANALYSIS COMPLETE!")
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Review the file structure above")
    print(f"   2. Identify which files contain document text data")
    print(f"   3. Create integration script based on data format")
    print(f"   4. Clean and add to training dataset")

if __name__ == "__main__":
    main()