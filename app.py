import os
import sys
import argparse
import logging
import uvicorn
import config
from api import app
from models import get_model
from utils import load_sample_data, augment_with_additional_data

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
    Classify a document file
    
    Args:
        file_path: Path to the document file
        model_type: Type of model to use
        
    Returns:
        Classification result
    """
    from preprocessor import extract_text_from_file, preprocess_for_model
    from utils.document_analyzer import analyze_document
    import re
    
    SUPPORTED_TYPES = [
        "invoice", "quote", "purchase_order", "delivery_note", "receipt", "bank_statement", "expense_report", "payslip"
    ]
    
    logger.info(f"Classifying file: {file_path}")
    
    # First, try rule-based classification which is more reliable for specific document types
    rule_based_type = analyze_document(file_path)
    logger.info(f"Rule-based classification result: {rule_based_type}")
    
    # Extract text from file
    text = extract_text_from_file(file_path=file_path)
    
    # Check for invoice structural patterns before classification
    # These patterns focus on the structure typical of invoices in any language
    text_lower = text.lower()
    
    # Check for invoice structural components
    invoice_indicators = {
        'invoice_keyword': any(keyword in text_lower for keyword in ['facture', 'invoice', 'فاتورة']),
        'invoice_number': re.search(r'(?:facture|invoice|فاتورة).*?(?:n[°o]|number|#|رقم|\d{4,})', text_lower) is not None,
        'amount_pattern': len(re.findall(r'\d+[.,]\d{2,3}', text_lower)) >= 2,  # At least two decimal amounts
        'tax_indicator': any(tax in text_lower for tax in ['tva', 'tax', 'vat', 'ضريبة', 'أداء']),
        'total_amount': re.search(r'(?:total|montant|amount|إجمالي|المبلغ).*?\d+[.,]\d+', text_lower) is not None,
        'date_pattern': re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text_lower) is not None,
        'tunisian_invoice': any(pattern in text_lower for pattern in ['facture du mois (dt)', 'فاتورة الشهر', 'montant total hors'])
    }
    
    # Count how many structural components are present
    invoice_structure_score = sum(1 for indicator in invoice_indicators.values() if indicator)
    
    # If document has strong invoice structural characteristics, override with invoice classification
    # Lower threshold for multilingual invoices, especially Tunisian ones with mixed French/Arabic
    if (invoice_structure_score >= 4 and invoice_indicators['invoice_keyword'] and invoice_indicators['amount_pattern']) or \
       (invoice_structure_score >= 3 and invoice_indicators['invoice_keyword'] and invoice_indicators['tax_indicator']) or \
       (invoice_indicators['tunisian_invoice'] and invoice_indicators['amount_pattern']):
        logger.info(f"Detected strong invoice structure ({invoice_structure_score}/7 indicators) - overriding classification")
        logger.info(f"Invoice indicators: {invoice_indicators}")
        
        # Create confidence scores with invoice as highest, only for supported types
        confidence_scores = {t: (0.75 if t == "invoice" else 0.05) for t in SUPPORTED_TYPES}
        
        return {
            'prediction': "invoice",
            'confidence': 0.75,
            'confidence_scores': confidence_scores,
            'override': True,
            'invoice_structure_score': invoice_structure_score
        }
    
    # Preprocess text
    processed_text = preprocess_for_model(text, model_type=model_type)
    
    # Get model
    model = get_model(model_type)
    
    # Get prediction
    result = model.predict(processed_text)
    
    # If rule-based classification identified a document type with high confidence, use that instead
    if rule_based_type != "❓ Unknown Document Type":
        logger.info(f"Using rule-based classification: {rule_based_type}")
        
        # Create confidence scores with the rule-based type as highest, only for supported types
        confidence_scores = {t: 0.05 for t in SUPPORTED_TYPES}
        rule_type_key = rule_based_type.lower()
        if rule_type_key in confidence_scores:
            confidence_scores[rule_type_key] = 0.75
        
        # Normalize scores
        total = sum(confidence_scores.values())
        for key in confidence_scores:
            confidence_scores[key] = confidence_scores[key] / total
        
        return {
            'prediction': rule_type_key,
            'confidence': confidence_scores[rule_type_key],
            'confidence_scores': confidence_scores,
            'override': True,
            'rule_based_classification': True
        }
    
    # Override borderline cases with proper structural analysis
    if (invoice_structure_score >= 3 and result['confidence'] < 0.3) or \
       (invoice_indicators['tunisian_invoice'] and result['confidence'] < 0.4) or \
       (invoice_indicators['invoice_keyword'] and invoice_indicators['tax_indicator'] and result['confidence'] < 0.35):
        logger.info(f"Borderline case with invoice structural indicators ({invoice_structure_score}/7) - adjusting scores")
        
        # Boost invoice confidence for borderline cases with good structural indicators
        if 'invoice' in result['confidence_scores']:
            # Higher boost for Tunisian invoices with mixed French/Arabic
            if invoice_indicators['tunisian_invoice']:
                result['confidence_scores']['invoice'] = max(result['confidence_scores']['invoice'], 0.65)
            else:
                result['confidence_scores']['invoice'] = max(result['confidence_scores']['invoice'], 0.55)
            
            # Normalize scores
            total = sum(result['confidence_scores'].values())
            for key in result['confidence_scores']:
                result['confidence_scores'][key] = result['confidence_scores'][key] / total
            
            # Update prediction and confidence
            result['prediction'] = 'invoice'
            result['confidence'] = result['confidence_scores']['invoice']
            result['invoice_structure_analysis'] = True
            logger.info(f"Adjusted invoice confidence to {result['confidence']}")
            logger.info(f"Tunisian invoice indicators detected: {invoice_indicators['tunisian_invoice']}")
    
    # Filter model confidence scores to only supported types
    filtered_scores = {k: v for k, v in result['confidence_scores'].items() if k in SUPPORTED_TYPES}
    total = sum(filtered_scores.values())
    if total > 0:
        for k in filtered_scores:
            filtered_scores[k] = filtered_scores[k] / total
    else:
        filtered_scores = {k: 1.0 / len(SUPPORTED_TYPES) for k in SUPPORTED_TYPES}
    result['confidence_scores'] = filtered_scores
    if result['prediction'] not in SUPPORTED_TYPES:
        result['prediction'] = max(filtered_scores, key=filtered_scores.get)
        result['confidence'] = filtered_scores[result['prediction']]
    
    logger.info(f"Classification result: {result}")
    return result

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