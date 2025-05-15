import re
import string
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources on first import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define keywords that should be preserved during text preprocessing (including multilingual terms)
IMPORTANT_KEYWORDS = {
    # English invoice keywords
    'invoice', 'bill', 'payment', 'due', 'amount', 'total', 'tax', 'vat', 'receipt', 
    'order', 'customer', 'service', 'quantity', 'price', 'date', 'paid', 'subtotal',
    # French invoice keywords
    'facture', 'montant', 'paiement', 'client', 'tva', 'total', 'date', 'numéro',
    'échéance', 'prix', 'quantité', 'hors', 'taxes', 'ttc', 'ht', 'reçu',
    'envoyé', 'modalités', 'paiement', 'règlement', 'remise', 'réduction',
    'échéance', 'acquitté', 'versement', 'acompte', 'facture du mois', 'facture n°',
    # Arabic invoice keywords - expanded list with actual Arabic characters
    'فاتورة', 'مبلغ', 'ضريبة', 'تاريخ', 'إجمالي', 'دفع', 'مستحق', 'عميل', 
    'رقم الفاتورة', 'فاتورة رقم', 'المبلغ الإجمالي', 'فاتورة الشهر',
    'ضريبة القيمة المضافة', 'الحساب', 'سعر', 'كمية', 'المبلغ', 'دون', 'إجمالي',
    # Purchase order keywords - expanded list
    'bon', 'commande', 'purchase', 'order', 'po', 'bc', 'supplier', 'fournisseur', 
    'delivery', 'livraison', 'référence', 'reference', 'acheteur', 'buyer', 'articles',
    'modalité', 'emis', 'conditions', 'délai', 'vendeur', 'seller', 'commander',
    'bon de commande', 'purchase order', 'emetteur', 'issuer', 'destinataire',
    'numéro de commande', 'order number', 'order no', 'bc-', 'ref', 'conditions de paiement'
}

def clean_text(text):
    """
    Clean the extracted text by removing special characters, extra whitespace, etc.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase but preserve Arabic characters
    # Arabic characters are not affected by lowercase conversion
    text = text.lower()
    
    # Special handling for combining forms of arabic text
    text = text.replace('\u200f', ' ')  # Remove right-to-left mark
    text = text.replace('\u200e', ' ')  # Remove left-to-right mark
    
    # Normalize spaces around Arabic text
    # Add spaces around Arabic character sequences to ensure proper tokenization
    text = re.sub(r'([^\u0600-\u06FF\s])([^\u0600-\u06FF\s])', r'\1 \2', text)
    
    # Remove special characters but preserve spaces between words and Arabic characters
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Basic sanity check - if text is too short or too long, it might be invalid
    if len(text) < config.MIN_TEXT_LENGTH:
        logger.warning(f"Text is too short ({len(text)} chars), might be invalid")
    elif len(text) > config.MAX_TEXT_LENGTH:
        logger.warning(f"Text is very long ({len(text)} chars), truncating")
        text = text[:config.MAX_TEXT_LENGTH]
    
    return text

def tokenize_text(text):
    """
    Tokenize the text into words
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize, but preserve important keywords
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token in IMPORTANT_KEYWORDS or (token not in stop_words and token not in string.punctuation)]
    
    return tokens

def preprocess_for_model(text, model_type=None):
    """
    Preprocess the text for a specific model
    
    Args:
        text: Input text to preprocess
        model_type: Type of model to preprocess for
        
    Returns:
        Preprocessed text or tokens depending on the model
    """
    # If model_type is None, use the default model from config
    if model_type is None:
        model_type = config.DEFAULT_MODEL
        logger.info(f"No model type specified, using default: {model_type}")
    
    cleaned_text = clean_text(text)
    
    # Detect potentially important invoice-specific patterns
    invoice_patterns = [
        r'facture\s*[#:n°]', r'invoice\s*[#:n°]', r'فاتورة\s*رقم',
        r'facture\s*du\s*mois', r'فاتورة\s*الشهر', r'montant\s*ttc'
    ]
    
    # Check if any strong invoice pattern is present
    strong_invoice_match = False
    for pattern in invoice_patterns:
        if re.search(pattern, cleaned_text.lower()):
            strong_invoice_match = True
            break
    
    # Find and duplicate important keywords to give them more weight
    for keyword in IMPORTANT_KEYWORDS:
        if f" {keyword} " in cleaned_text or cleaned_text.startswith(f"{keyword} "):
            cleaned_text = cleaned_text + f" {keyword} {keyword}"

    # --- NEW: Purchase order boosting ---
    po_patterns = [
        r'bon\s*de\s*commande', r'purchase\s*order', r'po\s*[#:n°]', r'bc\s*[#:n°]',
        r'order\s*[#:n°]', r'commande\s*[#:n°]', r'order number', r'numéro de commande',
        r'vendeur', r'buyer', r'supplier', r'fournisseur', r'articles', r'délai de livraison'
    ]
    strong_po_match = False
    for pattern in po_patterns:
        if re.search(pattern, cleaned_text.lower()):
            strong_po_match = True
            break
    if strong_po_match:
        po_keywords = ['purchase order', 'bon de commande', 'po', 'bc', 'order', 'commande', 'supplier', 'fournisseur']
        for keyword in po_keywords:
            cleaned_text = cleaned_text + f" {keyword} {keyword} {keyword} {keyword} {keyword}"
        # If strong PO match, remove invoice boosting to avoid confusion
        cleaned_text = re.sub(r'(invoice ){3,}', 'invoice ', cleaned_text)
        cleaned_text = re.sub(r'(facture ){3,}', 'facture ', cleaned_text)

    # If both strong invoice and strong PO patterns, reduce invoice boosting
    if strong_invoice_match and strong_po_match:
        cleaned_text = cleaned_text.replace('invoice invoice invoice', 'invoice')
        cleaned_text = cleaned_text.replace('facture facture facture', 'facture')

    # If this is likely an invoice based on patterns, boost invoice keywords
    if strong_invoice_match and not strong_po_match:
        invoice_keywords = ['facture', 'invoice', 'فاتورة', 'montant', 'total']
        for keyword in invoice_keywords:
            # Add these keywords multiple times to strongly boost their weight
            cleaned_text = cleaned_text + f" {keyword} {keyword} {keyword} {keyword} {keyword}"

    if model_type == "sklearn_tfidf_svm":
        return cleaned_text
    elif model_type in ["bert", "layoutlm"]:
        tokens = tokenize_text(cleaned_text)
        return " ".join(tokens)
    else:
        logger.warning(f"Unsupported model type: {model_type}, falling back to default")
        return cleaned_text