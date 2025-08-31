"""
Document Classification Service - Clean Architecture
Following SOLID Principles
Author: Bachir Fahmi
Email: bachir.fahmi@example.com
Description: Clean, maintainable document classification CLI and API
"""

import os
import sys
import argparse
import logging
import uvicorn
import config
from api.app import app
from api.factories.service_factory import service_factory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentClassificationCLI:
    """
    CLI interface for document classification
    Following Single Responsibility Principle
    """
    
    def __init__(self):
        self._classification_service = service_factory.get_classification_service()
        self._financial_service = service_factory.get_financial_service()
    
    def train_model(self, model_type=None, optimize=False, enhanced=False):
        """
        Train a model
        
        Args:
            model_type: Type of model to train
            optimize: Whether to perform hyperparameter optimization
            enhanced: Whether to use the enhanced training process
            
        Returns:
            Training metrics
        """
        logger.info(f"Training model: {model_type if model_type else config.DEFAULT_MODEL}")
        
        if enhanced:
            logger.info("Using enhanced training process with expanded dataset")
            return self._train_enhanced_model(model_type, optimize)
        
        # Regular training process
        from utils import load_sample_data, augment_with_additional_data
        from models import get_model
        
        # Load sample data
        X, y = load_sample_data()
        
        # Augment with additional data
        additional_paths = [
            os.path.join(config.SAMPLE_DIR, "additional_invoices.csv"),
            os.path.join(config.SAMPLE_DIR, "commercial_documents.csv"),
            os.path.join(config.SAMPLE_DIR, "multilingual_invoice_examples.csv")
        ]
        
        for path in additional_paths:
            if os.path.exists(path):
                X, y = augment_with_additional_data(X, y, path)
                logger.info(f"Added data from {path}")
                # Add multilingual examples twice for importance
                if "multilingual" in path:
                    X, y = augment_with_additional_data(X, y, path)
        
        # Get and train model
        model = get_model(model_type)
        metrics = model.train(X, y, optimize=optimize)
        
        logger.info(f"Model training complete: {metrics}")
        return metrics

    def _train_enhanced_model(self, model_type=None, optimize=False):
        """
        Train a model using the enhanced dataset and process
        
        Args:
            model_type: Type of model to train
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Training metrics
        """
        import pandas as pd
        import pickle
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
        
        logger.info("Starting enhanced model training process")
        
        # Check if the complete training data file exists
        complete_data_path = os.path.join(config.SAMPLE_DIR, "complete_training_data.csv")
        if not os.path.exists(complete_data_path):
            logger.warning(f"Enhanced training data not found at {complete_data_path}")
            logger.info("Generating enhanced training data...")
            
            # Generate the enhanced dataset
            try:
                import subprocess
                result = subprocess.run([sys.executable, "prepare_training_data.py"], 
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Error generating training data: {result.stderr}")
                    logger.info("Falling back to standard training process")
                    return self.train_model(model_type, optimize)
            except Exception as e:
                logger.error(f"Error running prepare_training_data.py: {str(e)}")
                logger.info("Falling back to standard training process")
                return self.train_model(model_type, optimize)
        
        # Load the enhanced training data
        try:
            logger.info(f"Loading enhanced training data from {complete_data_path}")
            df = pd.read_csv(complete_data_path)
            
            logger.info(f"Dataset loaded with {len(df)} samples")
            class_counts = df['label'].value_counts()
            logger.info("Class distribution:")
            for label, count in class_counts.items():
                logger.info(f"  {label}: {count} samples")
            
            # Preprocess text
            df['processed_text'] = df['text'].apply(lambda text: text.lower() if isinstance(text, str) else "")
            
            # Split into training and testing sets
            # Check if stratification is possible (all classes need at least 2 samples)
            min_class_count = df['label'].value_counts().min()
            
            if min_class_count >= 2:
                logger.info("Using stratified split (all classes have >= 2 samples)")
                X_train, X_test, y_train, y_test = train_test_split(
                    df['processed_text'], 
                    df['label'], 
                    test_size=0.2, 
                    random_state=42,
                    stratify=df['label']
                )
            else:
                logger.info(f"Using random split (some classes have only {min_class_count} sample)")
                X_train, X_test, y_train, y_test = train_test_split(
                    df['processed_text'], 
                    df['label'], 
                    test_size=0.2, 
                    random_state=42
                )
            
            logger.info(f"Training set size: {len(X_train)}")
            logger.info(f"Testing set size: {len(X_test)}")
            
            # Use the SklearnClassifier for consistency
            logger.info("Training enhanced model using SklearnClassifier...")
            from models import get_model
            model = get_model(model_type)
            
            # Train the model with the complete dataset
            X_data = df['text'].values
            y_data = df['label'].values
            
            success = model.train(X_data, y_data, test_size=0.2, optimize=optimize)
            
            if success:
                logger.info("Enhanced model training completed successfully!")
                
                # Save the model
                model.save_model()
                logger.info("Enhanced model saved successfully!")
                
                return {"status": "success", "message": "Enhanced training completed"}
            else:
                logger.error("Enhanced model training failed!")
                return {"status": "error", "message": "Enhanced training failed"}
        
        except Exception as e:
            logger.error(f"Error in enhanced training process: {str(e)}")
            logger.info("Falling back to standard training process")
            return self.train_model(model_type, optimize)

    def classify_file(self, file_path, model_type=None):
        """
        Classify a document file
        
        Args:
            file_path: Path to the document file
            model_type: Type of model to use
            
        Returns:
            Classification result
        """
        logger.info(f"Classifying file: {file_path}")
        
        try:
            result = self._classification_service.classify_document(file_path, upload_to_cloud=True)
            return result
        except Exception as e:
            logger.error(f"Error classifying file: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "classification": None
            }

    def start_api(self):
        """Start the API server"""
        logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
        uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

def main():
    """Main CLI entry point"""
    cli = DocumentClassificationCLI()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Document Classification Service - Clean Architecture")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", choices=["sklearn_tfidf_svm", "bert", "layoutlm"], help="Model type")
    train_parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    train_parser.add_argument("--enhanced", action="store_true", help="Use enhanced training process")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a document")
    classify_parser.add_argument("file", help="Path to the document file")
    classify_parser.add_argument("--model", choices=["sklearn_tfidf_svm", "bert", "layoutlm"], help="Model type")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "train":
        cli.train_model(args.model, args.optimize, args.enhanced)
    elif args.command == "classify":
        result = cli.classify_file(args.file, args.model)
        if result.get("success", True):
            print(f"Document classified as: {result['final_prediction']} (confidence: {result['model_confidence']:.2f})")
        else:
            print(f"Classification failed: {result.get('error')}")
    elif args.command == "api":
        cli.start_api()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()