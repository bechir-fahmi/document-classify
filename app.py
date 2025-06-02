import os
import sys
import argparse
import logging
import uvicorn
import config
from api import app
from models import get_model
from utils import load_sample_data, augment_with_additional_data
from utils.cloudinary_utils import upload_document
from utils.text_extraction import extract_text_from_file
from preprocessor.text_processor import preprocess_for_model
from utils.document_analyzer import analyze_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model_type=None, optimize=False, enhanced=False):
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
        return train_enhanced_model(model_type, optimize)
    
    # Regular training process
    # Load sample data
    X, y = load_sample_data()
    
    # Augment with additional invoice examples
    additional_data_path = os.path.join(config.SAMPLE_DIR, "additional_invoices.csv")
    X, y = augment_with_additional_data(X, y, additional_data_path)
    
    # Augment with additional commercial document examples
    commercial_data_path = os.path.join(config.SAMPLE_DIR, "commercial_documents.csv")
    X, y = augment_with_additional_data(X, y, commercial_data_path)
    
    # Augment with multilingual invoice examples (particularly focusing on mixed language documents)
    multilingual_path = os.path.join(config.SAMPLE_DIR, "multilingual_invoice_examples.csv")
    if os.path.exists(multilingual_path):
        logger.info("Adding multilingual invoice examples to training data")
        X, y = augment_with_additional_data(X, y, multilingual_path)
        # Add these examples twice to increase their importance in training
        X, y = augment_with_additional_data(X, y, multilingual_path)
    else:
        logger.warning(f"Multilingual invoice examples file not found: {multilingual_path}")
    
    # Get model
    model = get_model(model_type)
    
    # Train model
    metrics = model.train(X, y, optimize=optimize)
    
    logger.info(f"Model training complete: {metrics}")
    return metrics

def train_enhanced_model(model_type=None, optimize=False):
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
                return train_model(model_type, optimize)
        except Exception as e:
            logger.error(f"Error running prepare_training_data.py: {str(e)}")
            logger.info("Falling back to standard training process")
            return train_model(model_type, optimize)
    
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
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Testing set size: {len(X_test)}")
        
        # Create the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=2
            )),
            ('classifier', OneVsRestClassifier(LinearSVC(random_state=42)))
        ])
        
        # Train the model
        logger.info("Training enhanced SVM model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        from sklearn.metrics import classification_report, accuracy_score
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Test accuracy: {accuracy:.4f}")
        
        # Generate the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save the enhanced model
        enhanced_model_path = os.path.join(config.MODEL_DIR, "commercial_doc_classifier_enhanced.pkl")
        
        logger.info(f"Saving enhanced model to {enhanced_model_path}")
        with open(enhanced_model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Update the default SklearnClassifier model as well for compatibility
        standard_model_path = os.path.join(config.MODEL_DIR, "sklearn_tfidf_svm.pkl")
        logger.info(f"Updating standard model file at {standard_model_path}")
        with open(standard_model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Return the metrics
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    except Exception as e:
        logger.error(f"Error in enhanced training process: {str(e)}")
        logger.info("Falling back to standard training process")
        return train_model(model_type, optimize)

def classify_file(file_path, model_type=None):
    """
    Classify a document file and upload it to Cloudinary
    
    Args:
        file_path: Path to the document file
        model_type: Type of model to use
        
    Returns:
        Dictionary containing classification result and Cloudinary URL
    """
    from preprocessor import extract_text_from_file, preprocess_for_model
    from utils.document_analyzer import analyze_document
    import re
    
    SUPPORTED_TYPES = [
        "invoice", "quote", "purchase_order", "delivery_note", "receipt", "bank_statement", "expense_report", "payslip"
    ]
    
    logger.info(f"Classifying file: {file_path}")
    
    # Upload file to Cloudinary
    upload_result = upload_document(file_path)
    if not upload_result["success"]:
        logger.error(f"Failed to upload file to Cloudinary: {upload_result.get('error')}")
        return {
            "success": False,
            "error": "Failed to upload document to cloud storage",
            "classification": None
        }
    
    # First, try rule-based classification which is more reliable for specific document types
    try:
        # Extract text from the file
        text = extract_text_from_file(file_path)
        
        # Preprocess the text
        processed_text = preprocess_for_model(text)
        
        # Get the model
        model = get_model(model_type)
        
        # Classify the document
        classification = model.classify(processed_text)
        
        # Analyze the document for additional metadata
        metadata = analyze_document(text)
        
        return {
            "success": True,
            "classification": classification,
            "metadata": metadata,
            "cloudinary_url": upload_result["url"],
            "public_id": upload_result["public_id"]
        }
        
    except Exception as e:
        logger.error(f"Error classifying file: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "classification": None,
            "cloudinary_url": upload_result["url"],
            "public_id": upload_result["public_id"]
        }

def start_api():
    """
    Start the API server
    """
    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Document Classification Service")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", choices=["sklearn_tfidf_svm", "bert", "layoutlm"], help="Model type")
    train_parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    train_parser.add_argument("--enhanced", action="store_true", help="Use enhanced training process with expanded dataset")
    
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
        train_model(args.model, args.optimize, args.enhanced)
    elif args.command == "classify":
        result = classify_file(args.file, args.model)
        print(f"Document classified as: {result['prediction']} (confidence: {result['confidence']:.2f})")
    elif args.command == "api":
        start_api()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()