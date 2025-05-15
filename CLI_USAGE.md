# Commercial Document Classifier

A multilingual document classification system capable of identifying various commercial document types including invoices, purchase orders, quotes, and more in English, French, and Arabic.

## Key Features

- Classifies 8+ different types of commercial documents
- Supports multiple languages (English, French, Arabic)
- Works with various file formats (PDF, images, text files)
- Handles mixed-language documents
- Special support for multilingual invoices

## Usage

### Classifying Documents

```bash
python commercial_doc_classifier.py classify path/to/document.pdf
```

### Training the Model

To train with enhanced multilingual support:

```bash
python commercial_doc_classifier.py train --enhanced
```

For hyperparameter optimization (longer training time):

```bash
python commercial_doc_classifier.py train --enhanced --optimize
```

### Getting Information

Display system information and capabilities:

```bash
python commercial_doc_classifier.py info
```

## Recent Improvements

- Enhanced multilingual invoice detection
- Improved Arabic text processing
- Better mixed-language document handling
- Stronger pattern recognition for multilingual invoices
- Special handling for definitive invoice markers
- Tunisian invoice compatibility

## Supported Document Types

- Invoices (Factures / فاتورة)
- Quotes/Estimates (Devis)
- Purchase Orders (Bons de commande)
- Delivery Notes (Bons de livraison)
- Receipts (Reçus)
- Bank Statements (Relevés bancaires)
- Expense Reports (Notes de frais)
- Payslips (Bulletins de paie) 