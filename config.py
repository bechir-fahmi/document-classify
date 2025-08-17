"""
Configuration settings for Document Classification API
Author: Bachir Fahmi
Email: bachir.fahmi@example.com
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SAMPLE_DIR = os.path.join(DATA_DIR, "samples")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, SAMPLE_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# OCR Settings
# Use tesseract from PATH instead of a specific installation location
TESSERACT_PATH = "tesseract"  # Default to system PATH
# Fallback for specific OS installations if needed
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows default

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Model Settings
DEFAULT_MODEL = "sklearn_tfidf_svm"  # Options: "sklearn_tfidf_svm", "enhanced_sklearn", "bert", "layoutlm"
CONFIDENCE_THRESHOLD = 0.10  # Lowered to better handle multilingual documents with low confidence

# Document Classes
DOCUMENT_CLASSES = [
    "invoice",         # Facture
    "quote",           # Devis
    "purchase_order",  # Bon de commande
    "delivery_note",   # Bon de livraison
    "receipt",         # Reçu / Justificatif de paiement
    "bank_statement",  # Relevé bancaire
    "expense_report",  # Note de frais / Remboursement
    "payslip",         # Bulletin de paie
    "credit_note",     # Note de crédit
    "debit_note",      # Note de débit
    "tax_declaration", # Déclaration fiscale
    "fixed_asset_document", # Document d'immobilisation
    "inventory_document",   # Document d'inventaire
    "journal_entry",        # Écriture comptable
    "unknown"          # For any other document type
]

# Preprocessing
MIN_TEXT_LENGTH = 50  # Minimum text length to consider a valid document
MAX_TEXT_LENGTH = 100000  # Maximum text length to process 