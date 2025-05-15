#!/usr/bin/env python3
"""
Purchase Order Classification Tests

This module contains tests for the purchase order classification functionality,
including tests for both standard and edge case purchase orders in multiple languages.
"""

import os
import sys
import logging
import unittest
from app import classify_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PurchaseOrderClassificationTests(unittest.TestCase):
    """Tests for purchase order classification"""
    
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
    
    def test_french_purchase_order(self):
        """Test classification of a French purchase order"""
        # French purchase order text
        purchase_order_text = """Bon de commande ©

Mon Entreprise

22, Avenue Voltaire

13000 Marsaile, France
Téléphone : +33 4 9299 99 99

Date
Bon de commande N°
Numéro de client:
Modalité de paiement
Mode de paiement
Emis par

Contact lient
'Téléphone du client

Informations additionnelles.

482020
123

45

30 jours

CB/ Chèque
Pierre Fournisseur
Michael Acheteur
04 82 95 35 56

Merci d'avoir choisi Mon Entreprise pour nos services.

'Service après-vente - Garantie : 1 an

Destinataire
Acheteur SA

Michel Acheteur

31, rue..."""

        # Create test file
        test_file_path = os.path.join(self.temp_dir, "french_purchase_order.txt")
        self.test_files.append(test_file_path)
        
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(purchase_order_text)
        
        # Classify the document
        result = classify_file(test_file_path)
        
        # Verify classification
        self.assertEqual(result['prediction'], "purchase_order", 
                         f"French purchase order misclassified as {result['prediction']}")
        self.assertGreater(result['confidence'], 0.3, 
                          f"Confidence too low: {result['confidence']}")
        
        # Verify invoice confidence is significantly lower
        self.assertLess(result['confidence_scores'].get('invoice', 0), 0.2,
                       f"Invoice confidence too high: {result['confidence_scores'].get('invoice', 0)}")
    
    def test_standard_purchase_order(self):
        """Test classification of a standard purchase order"""
        # Check if the standard example file exists
        standard_po_path = os.path.join("data", "temp", "purchase_order_example.txt")
        
        if not os.path.exists(standard_po_path):
            self.skipTest(f"Standard purchase order example not found at {standard_po_path}")
        
        # Classify the document
        result = classify_file(standard_po_path)
        
        # Verify classification
        self.assertEqual(result['prediction'], "purchase_order",
                        f"Standard purchase order misclassified as {result['prediction']}")
        self.assertGreater(result['confidence'], 0.3,
                          f"Confidence too low: {result['confidence']}")
    
    def test_english_purchase_order(self):
        """Test classification of an English purchase order"""
        # English purchase order text
        purchase_order_text = """PURCHASE ORDER

Order Number: PO-12345
Date: 2023-05-15

SUPPLIER:
Tech Solutions Inc.
123 Innovation Drive
San Francisco, CA 94102, USA
Tax ID: 123-45-6789

BILL TO:
Acme Corporation
456 Main Street
New York, NY 10001, USA

ITEMS:
2 x Dell XPS 15 Laptop - Unit Price: $1,899.99 - Total: $3,799.98
5 x Dell 27" Monitor - Unit Price: $349.99 - Total: $1,749.95
10 x Logitech Wireless Mouse - Unit Price: $29.99 - Total: $299.90

AMOUNTS:
Subtotal: $5,849.83
Tax (8.875%): $519.17
Total: $6,369.00

TERMS:
Delivery: 15 days
Payment: Net 30
Warranty: 2 years parts and labor

SIGNATURES:

_________________                      _________________
For the Supplier                       For the Customer

Date: ____________                     Date: ____________"""

        # Create test file
        test_file_path = os.path.join(self.temp_dir, "english_purchase_order.txt")
        self.test_files.append(test_file_path)
        
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(purchase_order_text)
        
        # Classify the document
        result = classify_file(test_file_path)
        
        # Verify classification
        self.assertEqual(result['prediction'], "purchase_order", 
                         f"English purchase order misclassified as {result['prediction']}")
        self.assertGreater(result['confidence'], 0.25, 
                          f"Confidence too low: {result['confidence']}")

def run_tests():
    """Run the purchase order classification tests"""
    print("=== Running Purchase Order Classification Tests ===\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    run_tests() 