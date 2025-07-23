#!/usr/bin/env python3
"""
Automatic Dataset Processor
Just provide a URL or zip file path, and this script will:
1. Download/extract the data
2. Analyze the structure
3. Clean and process the data
4. Integrate with existing training data
5. Validate data quality
6. Generate reports and visualizations
"""
import pandas as pd
import os
import config
import zipfile
import requests
import json
import re
import logging
from urllib.parse import urlparse
from pathlib import Path
import tempfile
import shutil
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoDatasetProcessor:
    def __init__(self):
        self.temp_dir = None
        self.processed_samples = []
        self.source_name = ""
        
    def process_dataset(self, source_path):
        """Main function to process any dataset source"""
        print("🚀 AUTOMATIC DATASET PROCESSOR")
        print("=" * 50)
        
        try:
            # Step 1: Identify and load data source
            data_path = self._identify_and_load_source(source_path)
            if not data_path:
                return False
            
            # Step 2: Analyze data structure
            data_info = self._analyze_data_structure(data_path)
            if not data_info:
                return False
            
            # Step 3: Extract and clean data
            self.processed_samples = self._extract_and_clean_data(data_path, data_info)
            if not self.processed_samples:
                return False
            
            # Step 4: Validate processed data
            if not self._validate_processed_data():
                return False
            
            # Step 5: Integrate with existing data
            if not self._integrate_with_training_data():
                return False
            
            # Step 6: Final quality check and report
            self._generate_final_report()
            
            print("\n🎉 SUCCESS! Dataset processed and integrated!")
            return True
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            print(f"❌ Failed to process dataset: {str(e)}")
            return False
        finally:
            self._cleanup()
    
    def _identify_and_load_source(self, source_path):
        """Identify the source type and load the data"""
        print(f"\n📥 LOADING DATA SOURCE")
        print("-" * 30)
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Check if it's a URL
        if source_path.startswith(('http://', 'https://')):
            return self._download_from_url(source_path)
        
        # Check if it's a Hugging Face dataset
        elif '/' in source_path and not os.path.exists(source_path):
            return self._load_huggingface_dataset(source_path)
        
        # Check if it's a local zip file
        elif source_path.endswith('.zip') and os.path.exists(source_path):
            return self._extract_zip_file(source_path)
        
        # Check if it's a local directory
        elif os.path.isdir(source_path):
            return source_path
        
        # Check if it's a local file
        elif os.path.isfile(source_path):
            return source_path
        
        else:
            print(f"❌ Unknown source type: {source_path}")
            return None
    
    def _download_from_url(self, url):
        """Download data from URL"""
        print(f"🌐 Downloading from URL: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Determine file type from URL or headers
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "downloaded_data"
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'zip' in content_type or filename.endswith('.zip'):
                filename = filename if filename.endswith('.zip') else filename + '.zip'
            
            file_path = os.path.join(self.temp_dir, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {file_path}")
            
            # If it's a zip file, extract it
            if filename.endswith('.zip'):
                return self._extract_zip_file(file_path)
            else:
                return file_path
                
        except Exception as e:
            print(f"❌ Failed to download: {str(e)}")
            return None
    
    def _load_huggingface_dataset(self, dataset_name):
        """Load dataset from Hugging Face"""
        print(f"🤗 Loading Hugging Face dataset: {dataset_name}")
        
        try:
            dataset = load_dataset(dataset_name)
            
            # Save dataset info for processing
            dataset_info = {
                'type': 'huggingface',
                'dataset': dataset,
                'name': dataset_name
            }
            
            info_path = os.path.join(self.temp_dir, 'dataset_info.json')
            with open(info_path, 'w') as f:
                json.dump({'type': 'huggingface', 'name': dataset_name}, f)
            
            print(f"✅ Loaded Hugging Face dataset")
            return self.temp_dir
            
        except Exception as e:
            print(f"❌ Failed to load Hugging Face dataset: {str(e)}")
            return None
    
    def _extract_zip_file(self, zip_path):
        """Extract zip file"""
        print(f"📦 Extracting zip file: {zip_path}")
        
        try:
            extract_dir = os.path.join(self.temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"✅ Extracted to: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            print(f"❌ Failed to extract zip: {str(e)}")
            return None
    
    def _analyze_data_structure(self, data_path):
        """Analyze the structure of the loaded data"""
        print(f"\n🔍 ANALYZING DATA STRUCTURE")
        print("-" * 30)
        
        data_info = {
            'type': 'unknown',
            'files': [],
            'structure': {}
        }
        
        try:
            # Check for Hugging Face dataset info
            hf_info_path = os.path.join(data_path, 'dataset_info.json')
            if os.path.exists(hf_info_path):
                return self._analyze_huggingface_dataset(data_path)
            
            # Analyze directory structure
            if os.path.isdir(data_path):
                return self._analyze_directory_structure(data_path)
            
            # Analyze single file
            elif os.path.isfile(data_path):
                return self._analyze_single_file(data_path)
            
        except Exception as e:
            print(f"❌ Error analyzing structure: {str(e)}")
            return None
    
    def _analyze_huggingface_dataset(self, data_path):
        """Analyze Hugging Face dataset"""
        print("🤗 Analyzing Hugging Face dataset...")
        
        try:
            with open(os.path.join(data_path, 'dataset_info.json'), 'r') as f:
                info = json.load(f)
            
            dataset_name = info['name']
            dataset = load_dataset(dataset_name)
            
            print(f"Dataset: {dataset_name}")
            print(f"Splits: {list(dataset.keys())}")
            
            # Analyze first split
            first_split = list(dataset.keys())[0]
            sample = dataset[first_split][0]
            
            print(f"Sample keys: {list(sample.keys())}")
            
            return {
                'type': 'huggingface',
                'dataset': dataset,
                'name': dataset_name,
                'splits': list(dataset.keys()),
                'sample_keys': list(sample.keys())
            }
            
        except Exception as e:
            print(f"❌ Error analyzing HF dataset: {str(e)}")
            return None
    
    def _analyze_directory_structure(self, data_path):
        """Analyze directory structure"""
        print(f"📁 Analyzing directory: {data_path}")
        
        files = []
        for root, dirs, filenames in os.walk(data_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                files.append(file_path)
        
        # Categorize files
        csv_files = [f for f in files if f.endswith('.csv')]
        json_files = [f for f in files if f.endswith('.json')]
        txt_files = [f for f in files if f.endswith('.txt')]
        
        print(f"Found {len(files)} total files:")
        print(f"  CSV files: {len(csv_files)}")
        print(f"  JSON files: {len(json_files)}")
        print(f"  TXT files: {len(txt_files)}")
        
        # Determine primary data source
        if csv_files:
            primary_file = csv_files[0]
            file_type = 'csv'
        elif json_files:
            primary_file = json_files[0]
            file_type = 'json'
        elif txt_files:
            primary_file = txt_files[0]
            file_type = 'txt'
        else:
            print("❌ No supported data files found")
            return None
        
        return {
            'type': 'directory',
            'primary_file': primary_file,
            'file_type': file_type,
            'all_files': files,
            'csv_files': csv_files,
            'json_files': json_files,
            'txt_files': txt_files
        }
    
    def _analyze_single_file(self, file_path):
        """Analyze single file"""
        print(f"📄 Analyzing file: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            try:
                df = pd.read_csv(file_path)
                print(f"CSV shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                
                return {
                    'type': 'csv',
                    'file_path': file_path,
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
            except Exception as e:
                print(f"❌ Error reading CSV: {str(e)}")
                return None
        
        elif file_ext == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"JSON type: {type(data)}")
                if isinstance(data, list) and data:
                    print(f"List length: {len(data)}")
                    print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                
                return {
                    'type': 'json',
                    'file_path': file_path,
                    'data_type': type(data).__name__,
                    'sample': data[0] if isinstance(data, list) and data else data
                }
            except Exception as e:
                print(f"❌ Error reading JSON: {str(e)}")
                return None
        
        else:
            print(f"❌ Unsupported file type: {file_ext}")
            return None
    
    def _extract_and_clean_data(self, data_path, data_info):
        """Extract and clean data based on the analyzed structure"""
        print(f"\n🧹 EXTRACTING AND CLEANING DATA")
        print("-" * 30)
        
        if data_info['type'] == 'huggingface':
            return self._process_huggingface_data(data_info)
        elif data_info['type'] == 'csv':
            return self._process_csv_data(data_info)
        elif data_info['type'] == 'json':
            return self._process_json_data(data_info)
        elif data_info['type'] == 'directory':
            return self._process_directory_data(data_info)
        else:
            print(f"❌ Unknown data type: {data_info['type']}")
            return []
    
    def _process_huggingface_data(self, data_info):
        """Process Hugging Face dataset"""
        print("🤗 Processing Hugging Face dataset...")
        
        try:
            dataset = data_info['dataset']
            
            # Use train split if available, otherwise first split
            split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
            data = dataset[split_name]
            
            samples = []
            max_samples = 2000
            
            for i, sample in enumerate(data):
                if i >= max_samples:
                    break
                
                # Extract text and label
                text = self._extract_text_from_sample(sample)
                label = self._extract_label_from_sample(sample)
                
                if text and label and len(text) > 20:
                    cleaned_text = self._clean_text(text)
                    mapped_label = self._map_label(label)
                    
                    samples.append({
                        'text': cleaned_text,
                        'label': mapped_label,
                        'source': data_info['name']
                    })
                
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1} samples...")
            
            print(f"✅ Processed {len(samples)} samples from Hugging Face")
            return samples
            
        except Exception as e:
            print(f"❌ Error processing HF data: {str(e)}")
            return []
    
    def _process_csv_data(self, data_info):
        """Process CSV data"""
        print("📊 Processing CSV data...")
        
        try:
            df = pd.read_csv(data_info['file_path'])
            
            # Auto-detect text and label columns
            text_col = self._detect_text_column(df)
            label_col = self._detect_label_column(df)
            
            if not text_col or not label_col:
                print("❌ Could not detect text and label columns")
                return []
            
            print(f"Using text column: {text_col}")
            print(f"Using label column: {label_col}")
            
            samples = []
            for _, row in df.iterrows():
                text = str(row[text_col])
                label = str(row[label_col])
                
                if len(text) > 20:
                    cleaned_text = self._clean_text(text)
                    mapped_label = self._map_label(label)
                    
                    samples.append({
                        'text': cleaned_text,
                        'label': mapped_label,
                        'source': 'CSV'
                    })
            
            print(f"✅ Processed {len(samples)} samples from CSV")
            return samples
            
        except Exception as e:
            print(f"❌ Error processing CSV: {str(e)}")
            return []
    
    def _process_json_data(self, data_info):
        """Process JSON data"""
        print("📋 Processing JSON data...")
        
        try:
            with open(data_info['file_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            samples = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = self._extract_text_from_sample(item)
                        label = self._extract_label_from_sample(item)
                        
                        if text and label and len(text) > 20:
                            cleaned_text = self._clean_text(text)
                            mapped_label = self._map_label(label)
                            
                            samples.append({
                                'text': cleaned_text,
                                'label': mapped_label,
                                'source': 'JSON'
                            })
            
            print(f"✅ Processed {len(samples)} samples from JSON")
            return samples
            
        except Exception as e:
            print(f"❌ Error processing JSON: {str(e)}")
            return []
    
    def _process_directory_data(self, data_info):
        """Process directory with multiple files"""
        print("📁 Processing directory data...")
        
        if data_info['file_type'] == 'csv':
            return self._process_csv_data({'file_path': data_info['primary_file']})
        elif data_info['file_type'] == 'json':
            return self._process_json_data({'file_path': data_info['primary_file']})
        else:
            print(f"❌ Unsupported primary file type: {data_info['file_type']}")
            return []
    
    def _extract_text_from_sample(self, sample):
        """Extract text from a sample"""
        text_fields = ['text', 'content', 'document', 'body', 'description', 'message']
        
        for field in text_fields:
            if field in sample and sample[field]:
                return str(sample[field])
        
        # If no direct text field, try to combine multiple fields
        text_parts = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 10:
                text_parts.append(value)
        
        return ' '.join(text_parts) if text_parts else ""
    
    def _extract_label_from_sample(self, sample):
        """Extract label from a sample"""
        label_fields = ['label', 'category', 'type', 'class', 'document_type']
        
        for field in label_fields:
            if field in sample and sample[field]:
                return str(sample[field])
        
        return "document"  # Default label
    
    def _detect_text_column(self, df):
        """Auto-detect text column in DataFrame"""
        text_candidates = ['text', 'content', 'document', 'body', 'description']
        
        for col in text_candidates:
            if col in df.columns:
                return col
        
        # Find column with longest average text
        text_col = None
        max_avg_length = 0
        
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > max_avg_length:
                    max_avg_length = avg_length
                    text_col = col
        
        return text_col
    
    def _detect_label_column(self, df):
        """Auto-detect label column in DataFrame"""
        label_candidates = ['label', 'category', 'type', 'class', 'document_type']
        
        for col in label_candidates:
            if col in df.columns:
                return col
        
        # Find column with reasonable number of unique values
        for col in df.columns:
            if df[col].dtype == 'object' and col != self._detect_text_column(df):
                unique_ratio = df[col].nunique() / len(df)
                if 0.01 <= unique_ratio <= 0.5:  # Between 1% and 50% unique
                    return col
        
        return None
    
    def _clean_text(self, text):
        """Clean text content"""
        if not text:
            return ""
        
        text = str(text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
        
        # Clean OCR artifacts
        text = re.sub(r'\b[A-Z]{10,}\b', ' ', text)
        text = re.sub(r'\d+,\d+,\d+,\d+', ' ', text)
        text = re.sub(r'\[PAD\]|\[unused\d+\]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _map_label(self, label):
        """Map label to standard categories"""
        if not label:
            return "document"
        
        label_lower = str(label).lower().strip()
        
        # Standard mappings
        mappings = {
            'invoice': 'invoice',
            'receipt': 'receipt',
            'purchase_order': 'purchase_order',
            'delivery_note': 'delivery_note',
            'contract': 'contract',
            'quote': 'quote',
            'report': 'report',
            'statement': 'bank_statement'
        }
        
        # Direct mapping
        if label_lower in mappings:
            return mappings[label_lower]
        
        # Pattern matching
        for pattern, mapped_label in mappings.items():
            if pattern in label_lower:
                return mapped_label
        
        # Clean and return original
        return re.sub(r'[^a-z0-9_]', '_', label_lower)
    
    def _validate_processed_data(self):
        """Validate the processed data"""
        print(f"\n✅ VALIDATING PROCESSED DATA")
        print("-" * 30)
        
        if not self.processed_samples:
            print("❌ No samples processed")
            return False
        
        print(f"Total samples: {len(self.processed_samples)}")
        
        # Check text quality
        avg_length = sum(len(s['text']) for s in self.processed_samples) / len(self.processed_samples)
        print(f"Average text length: {avg_length:.1f} characters")
        
        # Check label distribution
        label_counts = {}
        for sample in self.processed_samples:
            label = sample['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        # Validation checks
        if avg_length < 50:
            print("⚠️  Warning: Average text length is quite short")
        
        if len(label_counts) < 2:
            print("⚠️  Warning: Only one label type found")
        
        print("✅ Data validation complete")
        return True
    
    def _integrate_with_training_data(self):
        """Integrate with existing training data"""
        print(f"\n🔄 INTEGRATING WITH TRAINING DATA")
        print("-" * 30)
        
        try:
            # Load existing data
            data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
            
            if os.path.exists(data_path):
                existing_df = pd.read_csv(data_path)
                print(f"Existing data: {len(existing_df)} samples")
            else:
                existing_df = pd.DataFrame(columns=['text', 'label'])
                print("No existing data found, creating new dataset")
            
            # Create new data DataFrame
            new_df = pd.DataFrame(self.processed_samples)
            print(f"New data: {len(new_df)} samples")
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            initial_size = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
            final_size = len(combined_df)
            
            duplicates_removed = initial_size - final_size
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicates")
            
            # Create backup
            if os.path.exists(data_path):
                backup_path = data_path.replace('.csv', '_backup_auto.csv')
                existing_df.to_csv(backup_path, index=False)
                print(f"Backup saved: {backup_path}")
            
            # Save updated data
            combined_df.to_csv(data_path, index=False)
            print(f"✅ Updated training data saved: {len(combined_df)} total samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Error integrating data: {str(e)}")
            return False
    
    def _generate_final_report(self):
        """Generate final processing report"""
        print(f"\n📋 GENERATING FINAL REPORT")
        print("-" * 30)
        
        try:
            # Load final dataset
            data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
            df = pd.read_csv(data_path)
            
            # Generate report
            report_lines = []
            report_lines.append("AUTOMATIC DATASET PROCESSING REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Processing Date: {pd.Timestamp.now()}")
            report_lines.append(f"Source: {self.source_name}")
            report_lines.append("")
            report_lines.append("RESULTS:")
            report_lines.append(f"  New samples added: {len(self.processed_samples)}")
            report_lines.append(f"  Total dataset size: {len(df)}")
            report_lines.append(f"  Unique labels: {df['label'].nunique()}")
            report_lines.append("")
            report_lines.append("LABEL DISTRIBUTION:")
            for label, count in df['label'].value_counts().items():
                report_lines.append(f"  {label}: {count}")
            
            # Save report
            report_path = os.path.join(config.DATA_DIR, "auto_processing_report.txt")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            print(f"✅ Report saved: {report_path}")
            
            # Print summary
            print("\n📊 PROCESSING SUMMARY:")
            print(f"  Added: {len(self.processed_samples)} new samples")
            print(f"  Total: {len(df)} samples")
            print(f"  Labels: {df['label'].nunique()}")
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

def main():
    """Main function"""
    print("🚀 AUTOMATIC DATASET PROCESSOR")
    print("=" * 50)
    print("Supported sources:")
    print("  • URLs (http/https)")
    print("  • Hugging Face datasets (username/dataset-name)")
    print("  • Local zip files")
    print("  • Local directories")
    print("  • Local CSV/JSON files")
    print("=" * 50)
    
    # Get source from user
    source = input("\n📥 Enter dataset source (URL, HF dataset, or file path): ").strip()
    
    if not source:
        print("❌ No source provided")
        return
    
    # Process the dataset
    processor = AutoDatasetProcessor()
    processor.source_name = source
    
    success = processor.process_dataset(source)
    
    if success:
        print("\n🎉 PROCESSING COMPLETE!")
        print("\n📋 NEXT STEPS:")
        print("1. Check the processing report")
        print("2. Update visualizations: python visualize_dataset_confidence.py")
        print("3. Retrain model: python app.py train --enhanced")
    else:
        print("\n❌ PROCESSING FAILED!")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()