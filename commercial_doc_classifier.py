#!/usr/bin/env python3
"""
Commercial Document Classifier - CLI Tool

This tool provides a user-friendly command-line interface for classifying commercial documents.
It can handle various document types (invoices, purchase orders, receipts, etc.) in multiple
languages (English, French, Arabic) and supports different file formats (PDF, images, text).

Usage:
    python commercial_doc_classifier.py classify <file>
    python commercial_doc_classifier.py train [--enhanced] [--optimize]
    python commercial_doc_classifier.py info
"""

import os
import sys
import argparse
import logging
from app import classify_file, train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_examples():
    """Show examples of document classification commands"""
    print("Example commands:")
    print("  python commercial_doc_classifier.py classify path/to/invoice.pdf")
    print("  python commercial_doc_classifier.py train --enhanced")
    print("  python commercial_doc_classifier.py info")

def show_info():
    """Display detailed information about the commercial document classification system"""
    print("\n=== Commercial Document Classification System ===\n")
    print("Supported document types:")
    print("  • Invoices (Factures / فاتورة)")
    print("  • Quotes/Estimates (Devis)")
    print("  • Purchase Orders (Bons de commande)")
    print("  • Delivery Notes (Bons de livraison)")
    print("  • Receipts (Reçus / Justificatifs de paiement)")
    print("  • Bank Statements (Relevés bancaires)")
    print("  • Expense Reports (Notes de frais / Remboursements)")
    print("  • Payslips (Bulletins de paie)")
    
    print("\nSupported languages:")
    print("  • English")
    print("  • French")
    print("  • Arabic")
    print("  • Mixed-language documents")
    
    print("\nSupported file formats:")
    print("  • PDF documents")
    print("  • Images (JPG, PNG, TIFF)")
    print("  • Text files (TXT, MD, CSV)")
    
    print("\nModel information:")
    model_file = os.path.join("data", "models", "commercial_doc_classifier_enhanced.pkl")
    standard_model = os.path.join("data", "models", "sklearn_tfidf_svm.pkl")
    
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        modified_time = os.path.getmtime(model_file)
        import datetime
        modified_date = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  • Enhanced model: {model_file} ({size_mb:.2f} MB)")
        print(f"  • Last updated: {modified_date}")
    elif os.path.exists(standard_model):
        size_mb = os.path.getsize(standard_model) / (1024 * 1024)
        modified_time = os.path.getmtime(standard_model)
        import datetime
        modified_date = datetime.datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  • Standard model: {standard_model} ({size_mb:.2f} MB)")
        print(f"  • Last updated: {modified_date}")
    else:
        print("  • Model not found. Run 'python commercial_doc_classifier.py train --enhanced' to create it.")
    
    # Show recent improvements summary
    print("\nRecent Improvements:")
    print("  • Enhanced multilingual invoice detection")
    print("  • Improved Arabic text processing")
    print("  • Better mixed-language document handling")
    print("  • Stronger pattern recognition for invoices")
    print("  • Special handling for definitive invoice markers")
    print("  • Tunisian invoice compatibility")

def classify_document(file_path):
    """
    Classify a document file
    
    Args:
        file_path (str): Path to the document file to classify
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Classifying document: {file_path}")
    
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    
    # Check if the file type is supported
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.txt', '.md', '.csv']
    if file_extension.lower() not in supported_extensions:
        print(f"Warning: File extension {file_extension} may not be supported. Supported formats: {', '.join(supported_extensions)}")
    
    # Classify the document
    try:
        result = classify_file(file_path)
        
        # Print the results
        print("\n=== Classification Results ===")
        print(f"Document type: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Print top 3 confidence scores
        print("\nTop 3 confidence scores:")
        sorted_scores = sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True)
        for i, (label, confidence) in enumerate(sorted_scores[:3]):
            print(f"  {i+1}. {label}: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error classifying document: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Commercial Document Classifier")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify a document")
    classify_parser.add_argument("file", help="Path to the document file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--enhanced", action="store_true", help="Use enhanced training process with expanded dataset")
    train_parser.add_argument("--optimize", action="store_true", help="Perform hyperparameter optimization")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about the system")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "classify":
        classify_document(args.file)
    elif args.command == "train":
        print("Training model with enhanced multilingual support...")
        train_model(enhanced=args.enhanced, optimize=args.optimize)
        print("\nTraining complete! You can now classify documents using:")
        print("  python commercial_doc_classifier.py classify <file>")
    elif args.command == "info":
        show_info()
    else:
        show_examples()

if __name__ == "__main__":
    main() 