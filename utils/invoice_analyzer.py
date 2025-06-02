import os
import logging
import re
from .text_extraction import extract_text_from_file
from preprocessor.text_processor import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_invoice(file_path):
    """
    Analyze an invoice image/PDF and extract key information
    
    Args:
        file_path: Path to the invoice file
        
    Returns:
        Dictionary with extracted invoice data
    """
    # Extract text from file
    text = extract_text_from_file(file_path=file_path)
    
    # Clean text for analysis
    cleaned_text = clean_text(text)
    
    # Extract invoice number
    invoice_number = None
    invoice_patterns = [
        r'facture\s*[#:n°]*\s*([a-z0-9\-]+)',
        r'facture\s+à\s+envoyer\s+à\s+facture.*\s+n[o°][^a-z0-9]*([a-z0-9\-]+)',
        r'invoice\s*[#:n°]*\s*([a-z0-9\-]+)',
        r'[#:]?\s*([a-z0-9\-]{6,})',
        r'n[o°]\s*(?:facture)?[^a-z0-9]*([a-z0-9\-]+)',
        r'fac\.?\s*n[o°][^a-z0-9]*([a-z0-9\-]+)',
        r'reference\s*:?\s*([a-z0-9\-]+)',
        r'référence\s*:?\s*([a-z0-9\-]+)',
        r'numéro\s*:?\s*([a-z0-9\-]+)',
        r'n°\s*([a-z0-9\-]+)'
    ]
    
    for pattern in invoice_patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            invoice_number = match.group(1)
            break
    
    # If we still don't have an invoice number, try a more aggressive approach
    if not invoice_number and 'facture' in cleaned_text.lower():
        # Look for any alphanumeric string near "facture"
        facture_context = re.search(r'facture\s*[^\w]*(\w+)', cleaned_text, re.IGNORECASE)
        if facture_context:
            invoice_number = facture_context.group(1)
    
    # Extract amount
    amount = None
    amount_patterns = [
        r'total\s*:?\s*(\d+[.,]\d+)',
        r'total\s+ttc\s*:?\s*(\d+[.,]\d+)',
        r'montant\s*:?\s*(\d+[.,]\d+)',
        r'montant\s+ttc\s*:?\s*(\d+[.,]\d+)',
        r'ttc\s*:?\s*(\d+[.,]\d+)',
        r'à\s+payer\s*:?\s*(\d+[.,]\d+)',
        r'amount\s*:?\s*(\d+[.,]\d+)',
        r'price\s*:?\s*(\d+[.,]\d+)',
        r'prix\s*:?\s*(\d+[.,]\d+)'
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            amount = match.group(1)
            break
    
    # Check if it is an invoice by counting key terms
    invoice_terms = [
        'facture', 'invoice', 'total', 'montant', 'date', 'client', 
        'number', 'numéro', 'payment', 'paiement', 'amount',
        'echéance', 'tva', 'tax', 'vat', 'ttc', 'ht', 'prix', 'price'
    ]
    
    invoice_term_count = sum(1 for term in invoice_terms if term in cleaned_text.lower())
    
    # Check for definitive invoice markers
    definitive_markers = ['facture', 'invoice', 'فاتورة']
    is_definitely_invoice = any(marker in cleaned_text.lower() for marker in definitive_markers)
    
    # If we have a definitive marker or multiple invoice terms, it's an invoice
    is_invoice = is_definitely_invoice or invoice_term_count >= 2
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'invoice_number': invoice_number,
        'amount': amount,
        'is_invoice': is_invoice,
        'is_definitely_invoice': is_definitely_invoice,
        'invoice_term_count': invoice_term_count
    } 