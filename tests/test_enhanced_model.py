#!/usr/bin/env python3
"""
Enhanced Model Classification Tests

This module tests the enhanced document classification model against
a variety of commercial document types to ensure proper classification.
"""

import os
import sys
import logging
import unittest
from app import classify_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelTests(unittest.TestCase):
    """Tests for the enhanced document classification model"""
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create the temp directory if it doesn't exist
        self.temp_dir = os.path.join("data", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Files to clean up after tests
        self.test_files = []
    
    def tearDown(self):
        """Clean up after each test"""
        # Remove any test files created
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def test_invoice_classification(self):
        """Test classification of an invoice"""
        # Invoice text
        invoice_text = """FACTURE

Numéro: FAC-2023-001
Date: 15/05/2023

ÉMETTEUR:
Tech Solutions SARL
123 Avenue Victor Hugo
75016 Paris, France
SIRET: 123 456 789 00012

CLIENT:
Entreprise Client SA
45 Rue du Commerce
69002 Lyon, France
SIRET: 987 654 321 00025

PRESTATIONS:
Service de maintenance informatique - Mai 2023: 1500.00€
Remplacement matériel réseau: 750.00€
Formation équipe IT (3 jours): 2250.00€

MONTANT:
Total HT: 4500.00€
TVA (20%): 900.00€
Total TTC: 5400.00€

Mode de paiement: Virement bancaire
Échéance: 30 jours

FACTURE ACQUITTÉE
Merci pour votre confiance!"""

        # Create test file
        test_file_path = os.path.join(self.temp_dir, "test_invoice.txt")
        self.test_files.append(test_file_path)
        
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(invoice_text)
        
        # Classify the document
        result = classify_file(test_file_path)
        
        # Verify classification
        self.assertEqual(result['prediction'], "invoice", 
                         f"Invoice misclassified as {result['prediction']}")
        self.assertGreater(result['confidence'], 0.25, 
                          f"Confidence too low: {result['confidence']}")
    
    def test_invoice_confidence_scores(self):
        """Test that confidence scores for invoices are properly distributed"""
        # Get the result from the invoice test
        invoice_text = """FACTURE

Numéro: FAC-2023-001
Date: 15/05/2023

ÉMETTEUR:
Tech Solutions SARL
123 Avenue Victor Hugo
75016 Paris, France
SIRET: 123 456 789 00012

CLIENT:
Entreprise Client SA
45 Rue du Commerce
69002 Lyon, France
SIRET: 987 654 321 00025

PRESTATIONS:
Service de maintenance informatique - Mai 2023: 1500.00€
Remplacement matériel réseau: 750.00€
Formation équipe IT (3 jours): 2250.00€

MONTANT:
Total HT: 4500.00€
TVA (20%): 900.00€
Total TTC: 5400.00€

Mode de paiement: Virement bancaire
Échéance: 30 jours

FACTURE ACQUITTÉE
Merci pour votre confiance!"""

        # Create test file
        test_file_path = os.path.join(self.temp_dir, "test_invoice2.txt")
        self.test_files.append(test_file_path)
        
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(invoice_text)
        
        # Classify the document
        result = classify_file(test_file_path)
        
        # Verify confidence score is higher for invoice than purchase order
        invoice_confidence = result['confidence_scores']['invoice']
        purchase_order_confidence = result['confidence_scores']['purchase_order']
        receipt_confidence = result['confidence_scores'].get('receipt', 0)
        
        self.assertGreater(invoice_confidence, purchase_order_confidence, 
                           "Invoice confidence not higher than purchase order confidence")
        self.assertGreater(invoice_confidence, receipt_confidence, 
                           "Invoice confidence not higher than receipt confidence")

def run_tests():
    """Run the enhanced model tests"""
    print("=== Running Enhanced Model Classification Tests ===\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    run_tests() 