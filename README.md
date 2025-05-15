# Commercial Document Classification System

A robust multilingual system for classifying commercial documents like invoices, purchase orders, receipts, and more.

## Overview

This system can identify various types of commercial documents in multiple languages (English, French, and Arabic) with high accuracy. It uses machine learning techniques to classify documents based on their textual content, with specialized handling for different document types.

## Key Features

- **Multilingual Support**: Handles documents in English, French, and Arabic
- **Multiple Document Types**: Classifies 8 different document types:
  - Invoices (Factures)
  - Quotes/Estimates (Devis)
  - Purchase Orders (Bons de commande)
  - Delivery Notes (Bons de livraison)
  - Receipts (Reçus / Justificatifs de paiement)
  - Bank Statements (Relevés bancaires)
  - Expense Reports (Notes de frais / Remboursements)
  - Payslips (Bulletins de paie)
- **Flexible Input Formats**: Processes PDF files, images (JPG, PNG, TIFF), and text files
- **High-Performance OCR**: Extracts text using multiple OCR engines with fallback mechanisms
- **Robust Classification**: Uses TF-IDF vectorization with SVM classification enhanced by pattern matching

## Project Structure

```
.
├── api/                     # API server code
├── data/                    # Data storage
│   ├── models/              # Trained models
│   ├── sample/              # Sample documents
│   └── temp/                # Temporary files
├── models/                  # Model definitions
│   ├── sklearn_model.py     # TF-IDF + SVM model
│   ├── bert_model.py        # BERT-based model
│   ├── layoutlm_model.py    # LayoutLM model
│   └── model_factory.py     # Factory pattern for models
├── preprocessor/            # Text preprocessing
│   ├── text_extractor.py    # Extract text from documents
│   └── text_processor.py    # Clean and process text
├── tests/                   # Test suite
│   ├── test_purchase_orders.py  # Purchase order tests
│   ├── test_enhanced_model.py   # Enhanced model tests
│   └── run_all_tests.py     # Run all tests
├── utils/                   # Utility functions
├── app.py                   # Main application entry point
├── commercial_doc_classifier.py  # CLI interface
└── config.py                # Configuration settings
```

## Recent Improvements

The system has undergone significant improvements to fix issues with misclassified documents:

1. **Enhanced Training Data**:
   - Expanded dataset from 124 to 1,748 examples
   - Increased purchase order examples from 2 to 504 samples
   - Added synthetic data in multiple languages

2. **Improved Classification Logic**:
   - Implemented specialized pattern matching for document types
   - Added confidence boosting for specific document types
   - Created cross-category confidence adjustment system
   - Added special handling for multilingual documents

3. **Text Processing Enhancements**:
   - Added direct text file handling for more efficient processing
   - Enhanced OCR fallback mechanisms for better text extraction
   - Improved language detection and multilingual support
   - Added specialized text handling for different document formats

4. **System Integration**:
   - Enhanced text extraction from multiple file types
   - Added a comprehensive test suite
   - Created a user-friendly CLI interface

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Tesseract OCR (for text extraction from images and PDFs)
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd commercial-doc-classifier
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - For Windows: Use the installer in the repository or follow instructions in `TESSERACT_INSTALL_GUIDE.md`
   - For Linux: `sudo apt-get install tesseract-ocr`
   - For macOS: `brew install tesseract`

### Usage

#### Command Line Interface

Classify a document:
```
python commercial_doc_classifier.py classify path/to/document.pdf
```

Train the enhanced model:
```
python commercial_doc_classifier.py train --enhanced
```

Get information about the system:
```
python commercial_doc_classifier.py info
```

#### API Server

Start the API server:
```
python app.py api
```

### Running Tests

Run all tests:
```
python -m tests.run_all_tests
```

Run specific tests:
```
python -m tests.test_purchase_orders
python -m tests.test_enhanced_model
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Test Suite

The project includes a comprehensive test suite to verify the classification capabilities:

- Purchase Order tests - Verifies the fixes to purchase order classification
- Enhanced Model tests - Tests general document type classification

Running all tests:
```
python -m tests.run_all_tests
```

Test results after recent improvements:
- Purchase order tests: 3/3 passed
- Invoice classification tests: 2/2 passed
- Overall test success: 5/5 passed

## Acknowledgements

- Tesseract OCR
- scikit-learn
- FastAPI
- PyMuPDF (for PDF handling)
- EasyOCR (for multilingual OCR) 