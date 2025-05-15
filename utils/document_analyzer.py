import os
import re
import logging
from preprocessor import extract_text_from_file, clean_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keywords and patterns for different document types
DOCUMENT_PATTERNS = {
    "invoice": {
        "keywords": [
            # Enhanced list with more multilingual invoice keywords
            "facture", "invoice", "فاتورة", "montant", "tva", "ttc", "tax", "vat", 
            "total", "amount", "payment", "paiement", "due", "customer", "client",
            "échéance", "مبلغ", "ضريبة", "تاريخ", "إجمالي", "دفع", "مستحق", "عميل", 
            "رقم الفاتورة", "فاتورة رقم", "montant total", "total amount", "المبلغ الإجمالي",
            "facture n°", "invoice #", "invoice number", "numéro de facture", "facture du mois",
            "فاتورة الشهر", "facture proforma", "proforma invoice", "facture fournisseur", "facture client"
        ],
        "strong_patterns": [
            # Enhanced patterns with more variations and multilingual support
            r'facture\s*[#:n°]',
            r'invoice\s*[#:n°]',
            r'facture\s+.*\s+n[o°]',
            r'n[o°]\s*facture',
            r'fac\.?\s*n[o°]',
            r'facture\s*du\s*mois',
            r'facture\s*n.*\s*\d+',
            r'invoice\s*n.*\s*\d+',
            r'montant\s*(ht|ttc)',
            r'total\s*(amount|ttc|ht)',
            r'amount\s*due',
            r'فاتورة\s*رقم',
            r'رقم\s*الفاتورة',
            r'فاتورة\s*الشهر',
            r'مبلغ\s*إجمالي',
            r'المبلغ\s*الإجمالي',
            r'ضريبة\s*القيمة\s*المضافة',
            r'montant\s*total\s*hors',
            r'total\s*tva',
            r'facture.*\d{4,}'
        ]
    },
    "quote": {
        "keywords": [
            "devis", "quotation", "quote", "estimate", "proposal", "validité", 
            "valid until", "offer", "offre", "offre de prix", "prix estimatif", 
            "conditions de vente", "référence devis", "valable jusqu'au"
        ],
        "strong_patterns": [
            r'devis\s*[#:n°]',
            r'quotation\s*[#:n°]',
            r'quote\s*[#:n°]',
            r'estimate\s*[#:n°]',
            r'valid\s*until',
            r'validité',
            r'valable\s*jusqu',
            r'offre\s*de\s*prix',
            r'prix\s*estimatif',
            r'conditions\s*de\s*vente',
            r'référence\s*devis'
        ]
    },
    "purchase_order": {
        "keywords": [
            "bon de commande", "purchase order", "order", "commande", "po", "bc",
            "fournisseur", "vendor", "supplier", "delivery date", "date de livraison",
            "commande n°", "articles commandés", "quantité", "prix unitaire"
        ],
        "strong_patterns": [
            r'bon\s*de\s*commande',
            r'purchase\s*order',
            r'po\s*[#:n°]',
            r'bc\s*[#:n°]',
            r'order\s*[#:n°]',
            r'commande\s*[#:n°]',
            r'commande\s*n',
            r'articles\s*commandés',
            r'prix\s*unitaire'
        ]
    },
    "delivery_note": {
        "keywords": [
            "bon de livraison", "delivery note", "livraison", "delivered", "delivery",
            "carrier", "transporteur", "shipped", "shipping", "bl", "dn", "réception",
            "quantité livrée", "date de livraison", "conformité", "référence commande"
        ],
        "strong_patterns": [
            r'bon\s*de\s*livraison',
            r'delivery\s*note',
            r'bl\s*[#:n°]',
            r'dn\s*[#:n°]',
            r'delivery\s*[#:n°]',
            r'livraison\s*[#:n°]',
            r'quantité\s*livrée',
            r'date\s*de\s*livraison',
            r'conformité',
            r'référence\s*commande'
        ]
    },
    "receipt": {
        "keywords": [
            "reçu", "receipt", "payment received", "paiement reçu", "thank you for your payment",
            "merci pour votre paiement", "paid", "payé", "justificatif de paiement",
            "montant payé", "mode de paiement", "transaction", "virement", "cb"
        ],
        "strong_patterns": [
            r'reçu\s*[#:n°]',
            r'receipt\s*[#:n°]',
            r'payment\s*receipt',
            r'reçu\s*de\s*paiement',
            r'thank\s*you\s*for\s*your\s*payment',
            r'merci\s*pour\s*votre\s*paiement',
            r'justificatif\s*de\s*paiement',
            r'montant\s*payé',
            r'mode\s*de\s*paiement',
            r'transaction',
            r'virement',
            r'cb'
        ]
    },
    "bank_statement": {
        "keywords": [
            "relevé bancaire", "bank statement", "account", "compte", "balance", "solde",
            "transactions", "opérations", "credits", "debits", "period", "période",
            "relevé de compte", "crédit", "débit", "date"
        ],
        "strong_patterns": [
            r'relevé\s*bancaire',
            r'relevé\s*de\s*compte',
            r'bank\s*statement',
            r'account\s*[#:n°]',
            r'compte\s*[#:n°]',
            r'solde\s*initial',
            r'solde\s*final',
            r'opening\s*balance',
            r'closing\s*balance',
            r'solde\s*et\s*(crédit|débit)',
            r'crédit\s*et\s*débit'
        ]
    },
    "expense_report": {
        "keywords": [
            "note de frais", "expense report", "remboursement", "reimbursement", 
            "expenses", "dépenses", "justificatif", "receipts", "transport", "repas",
            "hébergement", "montant à rembourser", "ticket"
        ],
        "strong_patterns": [
            r'note\s*de\s*frais',
            r'expense\s*report',
            r'demande\s*de\s*remboursement',
            r'reimbursement\s*request',
            r'frais\s*professionnels',
            r'business\s*expenses',
            r'montant\s*à\s*rembourser',
            r'justificatif\s*de\s*frais',
            r'ticket\s*de\s*(transport|repas|hébergement)'
        ]
    },
    "payslip": {
        "keywords": [
            "bulletin de paie", "payslip", "salary", "salaire", "pay", "net à payer",
            "net pay", "gross", "brut", "deductions", "cotisations", "employer", "employeur",
            "fiche de paie", "congés", "heures"
        ],
        "strong_patterns": [
            r'bulletin\s*de\s*paie',
            r'fiche\s*de\s*paie',
            r'payslip',
            r'salaire\s*brut',
            r'gross\s*salary',
            r'net\s*à\s*payer',
            r'net\s*pay',
            r'cotisations\s*sociales',
            r'congés\s*payés',
            r'heures\s*travaillées'
        ]
    }
}

# Define the only supported document types
SUPPORTED_TYPES = [
    "invoice", "quote", "purchase_order", "delivery_note", "receipt", "bank_statement", "expense_report", "payslip"
]

def analyze_document(file_path):
    """
    Analyze a document and detect its type strictly by explicit keyword rules.
    Args:
        file_path: Path to the document file
    Returns:
        The document type as a string, or '❓ Unknown Document Type' if no rule matches.
    """
    text = extract_text_from_file(file_path=file_path)
    cleaned_text = clean_text(text)
    lower_text = cleaned_text.lower()
    
    # 1. Invoice (Facture / فاتورة) - Enhanced detection with more patterns
    # Check for strong invoice indicators first
    invoice_indicators = {
        # Basic keywords in multiple languages
        'invoice_keyword': any(keyword in lower_text for keyword in ['facture', 'invoice', 'فاتورة']),
        # Invoice number patterns
        'invoice_number': any(pattern in lower_text for pattern in ['facture n', 'facture du mois', 'فاتورة رقم', 'n° facture', 'facture #', 'facture n°']),
        # Tax-related terms
        'tax_indicator': any(tax in lower_text for tax in ['tva', 'tax', 'vat', 'ضريبة', 'أداء', 'أداء على القيمة المضافة']),
        # Amount-related terms
        'amount_indicator': any(amount in lower_text for amount in ['montant', 'total', 'amount', 'مبلغ', 'المبلغ', 'montant total', 'montant ht', 'montant ttc']),
        # Specific to Tunisian/Arabic invoices
        'arabic_amount': any(pattern in lower_text for pattern in ['المبلغ الإجمالي', 'مجموع', 'أداء على القيمة المضافة']),
        # Specific to Tunisian invoices with mixed French/Arabic
        'tunisian_invoice': any(pattern in lower_text for pattern in ['facture du mois (dt)', 'فاتورة الشهر', 'montant total hors'])
    }
    
    # Count how many invoice indicators are present
    invoice_score = sum(1 for indicator in invoice_indicators.values() if indicator)
    
    # Strong invoice detection logic (improved for English and newlines)
    if (
        # Classic invoice detection
        ("facture" in lower_text and ("tva" in lower_text or "ttc" in lower_text or "montant" in lower_text)) or
        # Arabic invoice detection
        ("فاتورة" in lower_text and ("ضريبة" in lower_text or "المبلغ" in lower_text or "أداء" in lower_text)) or
        # Tunisian invoice detection (mixed French/Arabic)
        ("facture" in lower_text and "فاتورة" in lower_text) or
        # Specific Tunisian invoice patterns
        ("facture du mois" in lower_text and ("montant total" in lower_text or "tva" in lower_text)) or
        # Strong structural indicators
        (invoice_indicators['invoice_keyword'] and invoice_score >= 3) or
        # Tunisian invoice with specific indicators
        (invoice_indicators['tunisian_invoice'] and invoice_indicators['tax_indicator']) or
        # High confidence invoice detection
        (invoice_score >= 4) or
        # --- NEW: English invoice fallback ---
        ("invoice" in lower_text and ("total" in lower_text or "amount" in lower_text)) or
        # --- NEW: Invoice with number, allowing for newlines or spaces ---
        (re.search(r'invoice\s*[#:n°\n\r\s]+\w+', lower_text))
    ):
        return "Invoice"

    # 2. Quote / Estimate (Devis / عرض أسعار)
    if (
        "devis" in lower_text and ("offre" in lower_text or "prix estimatif" in lower_text)
    ):
        return "Quote"

    # 3. Purchase Order (Bon de commande / أمر شراء)
    if (
        "bon de commande" in lower_text or "commande n" in lower_text
    ):
        return "Purchase Order"

    # 4. Delivery Note (Bon de livraison / مذكرة تسليم)
    if "bon de livraison" in lower_text:
        return "Delivery Note"

    # 5. Receipt / Payment Proof (Reçu / إيصال / سند دفع)
    if (
        "reçu" in lower_text or "justificatif de paiement" in lower_text
    ):
        return "Receipt"

    # 6. Bank Statement (Relevé bancaire / كشف حساب)
    if (
        "relevé de compte" in lower_text or
        ("solde" in lower_text and ("crédit" in lower_text or "débit" in lower_text))
    ):
        return "Bank Statement"

    # 7. Expense Report / Reimbursement (Note de frais / مذكرة مصاريف)
    if (
        "note de frais" in lower_text or "remboursement" in lower_text or "justificatif" in lower_text
    ):
        return "Expense Report"

    # 8. Payslip (Bulletin de paie / كشف راتب)
    if (
        "bulletin de paie" in lower_text or "salaire brut" in lower_text
    ):
        return "Payslip"

    # If no rule matches
    return "❓ Unknown Document Type"

def extract_document_info(text, doc_type):
    """
    Extract specific information based on document type
    
    Args:
        text: Cleaned text content
        doc_type: Detected document type
        
    Returns:
        Dictionary with extracted information
    """
    info = {}
    
    # Common patterns for all document types
    date_pattern = r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})'
    reference_pattern = r'[#:n°]?\s*([a-z0-9\-]{6,})'
    amount_pattern = r'(?:total|amount|montant|somme|sum)\s*:?\s*[$€£]?\s*(\d+[.,]\d+)'
    
    # Try to extract date
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    if date_match:
        info['date'] = date_match.group(1)
    
    # Extract document-specific information
    if doc_type == "invoice":
        # Invoice number
        for pattern in [r'facture\s*[#:n°]*\s*([a-z0-9\-]+)', r'invoice\s*[#:n°]*\s*([a-z0-9\-]+)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['invoice_number'] = match.group(1)
                break
                
        # Amount
        for pattern in [r'total\s*:?\s*(\d+[.,]\d+)', r'montant\s*:?\s*(\d+[.,]\d+)', r'ttc\s*:?\s*(\d+[.,]\d+)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['amount'] = match.group(1)
                break
    
    elif doc_type == "quote":
        # Quote/Estimate number
        for pattern in [r'devis\s*[#:n°]*\s*([a-z0-9\-]+)', r'quote\s*[#:n°]*\s*([a-z0-9\-]+)', r'quotation\s*[#:n°]*\s*([a-z0-9\-]+)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['quote_number'] = match.group(1)
                break
                
        # Validity
        valid_match = re.search(r'valid(?:ité)?\s*:?\s*(\d+\s*(?:jours|days|months|mois))', text, re.IGNORECASE)
        if valid_match:
            info['validity'] = valid_match.group(1)
    
    elif doc_type == "purchase_order":
        # PO number
        for pattern in [r'commande\s*[#:n°]*\s*([a-z0-9\-]+)', r'order\s*[#:n°]*\s*([a-z0-9\-]+)', r'po\s*[#:n°]*\s*([a-z0-9\-]+)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['po_number'] = match.group(1)
                break
    
    elif doc_type == "delivery_note":
        # Delivery note number
        for pattern in [r'livraison\s*[#:n°]*\s*([a-z0-9\-]+)', r'delivery\s*[#:n°]*\s*([a-z0-9\-]+)', r'bl\s*[#:n°]*\s*([a-z0-9\-]+)']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info['delivery_number'] = match.group(1)
                break
                
        # Related order
        order_match = re.search(r'(?:commande|order|po)\s*[#:n°]*\s*([a-z0-9\-]+)', text, re.IGNORECASE)
        if order_match:
            info['order_reference'] = match.group(1)
    
    elif doc_type == "receipt":
        # Payment method
        payment_match = re.search(r'(?:payment|paiement)\s*(?:method|mode|via|par)\s*:?\s*([a-zA-Z\s]+)', text, re.IGNORECASE)
        if payment_match:
            info['payment_method'] = payment_match.group(1).strip()
    
    elif doc_type == "bank_statement":
        # Account number (partially masked for privacy)
        account_match = re.search(r'(?:account|compte)\s*[#:n°]*\s*([a-zA-Z0-9\s\*]+)', text, re.IGNORECASE)
        if account_match:
            info['account_number'] = account_match.group(1).strip()
            
        # Period
        period_match = re.search(r'(?:period|période)\s*:?\s*([\w\s\-/]+)', text, re.IGNORECASE)
        if period_match:
            info['period'] = period_match.group(1).strip()
            
        # Balance
        balance_match = re.search(r'(?:closing|final)\s*balance\s*:?\s*[$€£]?\s*(\d+[.,]\d+)', text, re.IGNORECASE)
        if balance_match:
            info['balance'] = balance_match.group(1)
    
    elif doc_type == "expense_report":
        # Employee name
        employee_match = re.search(r'(?:employee|employé)\s*:?\s*([a-zA-Z\s]+)', text, re.IGNORECASE)
        if employee_match:
            info['employee'] = employee_match.group(1).strip()
            
        # Expense type
        expense_match = re.search(r'(?:expense|dépense)\s*(?:type|catégorie)\s*:?\s*([a-zA-Z\s]+)', text, re.IGNORECASE)
        if expense_match:
            info['expense_type'] = expense_match.group(1).strip()
    
    elif doc_type == "payslip":
        # Employee name
        employee_match = re.search(r'(?:employee|employé)\s*:?\s*([a-zA-Z\s]+)', text, re.IGNORECASE)
        if employee_match:
            info['employee'] = employee_match.group(1).strip()
            
        # Period/Month
        period_match = re.search(r'(?:period|période)\s*:?\s*([\w\s\-/]+)', text, re.IGNORECASE)
        if period_match:
            info['period'] = period_match.group(1).strip()
            
        # Net pay
        net_match = re.search(r'(?:net\s*(?:pay|à\s*payer))\s*:?\s*[$€£]?\s*(\d+[.,]\d+)', text, re.IGNORECASE)
        if net_match:
            info['net_pay'] = net_match.group(1)
    
    return info