import re
from typing import Optional

def generate_smart_excerpt(text: str, max_length: int = 300) -> str:
    """
    Generate a smart text excerpt that tries to capture meaningful content
    instead of just taking the first N characters.
    
    Args:
        text: The full text to excerpt from
        max_length: Maximum length of the excerpt
        
    Returns:
        A meaningful excerpt of the text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Clean the text first
    cleaned_text = clean_text_for_excerpt(text)
    
    if len(cleaned_text) <= max_length:
        return cleaned_text
    
    # Try to find meaningful sentences
    excerpt = find_meaningful_sentences(cleaned_text, max_length)
    
    if excerpt:
        return excerpt
    
    # Fallback to word boundary truncation
    return truncate_at_word_boundary(cleaned_text, max_length)

def clean_text_for_excerpt(text: str) -> str:
    """Clean text by removing excessive whitespace and formatting artifacts."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts and formatting
    text = re.sub(r'[^\w\s\.,;:!?\-€$£¥₹\(\)\/]', '', text)
    
    # Remove repeated characters (common in OCR)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    return text.strip()

def find_meaningful_sentences(text: str, max_length: int) -> Optional[str]:
    """
    Try to extract complete sentences that fit within the max_length.
    Prioritizes sentences that might contain important information.
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Score sentences based on potential importance
    scored_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
            
        score = calculate_sentence_importance(sentence)
        scored_sentences.append((sentence, score))
    
    # Sort by importance score
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Try to build excerpt with most important sentences
    excerpt_parts = []
    current_length = 0
    
    for sentence, score in scored_sentences:
        sentence_with_period = sentence + '.'
        if current_length + len(sentence_with_period) + 1 <= max_length:
            excerpt_parts.append(sentence_with_period)
            current_length += len(sentence_with_period) + 1
        elif not excerpt_parts:  # If no sentences fit, take the first important one truncated
            return truncate_at_word_boundary(sentence, max_length - 3) + '...'
    
    if excerpt_parts:
        return ' '.join(excerpt_parts)
    
    return None

def calculate_sentence_importance(sentence: str) -> float:
    """
    Calculate importance score for a sentence based on various factors.
    Higher score means more important.
    """
    score = 0.0
    sentence_lower = sentence.lower()
    
    # Financial keywords (high importance for financial documents)
    financial_keywords = [
        'facture', 'invoice', 'montant', 'amount', 'total', 'prix', 'price',
        'tva', 'tax', 'ht', 'ttc', 'devis', 'quote', 'commande', 'order',
        '€', '$', '£', 'eur', 'usd', 'gbp'
    ]
    
    # Company/client keywords
    company_keywords = [
        'société', 'company', 'sarl', 'sas', 'ltd', 'inc', 'corp',
        'client', 'customer', 'fournisseur', 'supplier'
    ]
    
    # Date keywords
    date_keywords = [
        'date', 'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
        'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        '2024', '2025', '2023'
    ]
    
    # Count keyword matches
    for keyword in financial_keywords:
        if keyword in sentence_lower:
            score += 3.0
    
    for keyword in company_keywords:
        if keyword in sentence_lower:
            score += 2.0
    
    for keyword in date_keywords:
        if keyword in sentence_lower:
            score += 1.0
    
    # Prefer sentences with numbers (likely important data)
    if re.search(r'\d+', sentence):
        score += 1.5
    
    # Prefer sentences with proper capitalization (likely headers/titles)
    if re.search(r'^[A-Z][a-z]', sentence):
        score += 0.5
    
    # Penalize very short or very long sentences
    length = len(sentence)
    if length < 20:
        score -= 1.0
    elif length > 200:
        score -= 0.5
    else:
        score += 0.5  # Sweet spot for sentence length
    
    return score

def truncate_at_word_boundary(text: str, max_length: int) -> str:
    """Truncate text at word boundary to avoid cutting words in half."""
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # Only use word boundary if it's not too far back
        return text[:last_space] + '...'
    else:
        return text[:max_length - 3] + '...'

def extract_key_information(text: str) -> dict:
    """
    Extract key information that should be highlighted in excerpts.
    """
    info = {
        'amounts': [],
        'dates': [],
        'companies': [],
        'invoice_numbers': []
    }
    
    # Extract monetary amounts
    amount_patterns = [
        r'(\d+[,.]?\d*)\s*€',
        r'€\s*(\d+[,.]?\d*)',
        r'(\d+[,.]?\d*)\s*EUR',
        r'total[:\s]+(\d+[,.]?\d*)',
        r'montant[:\s]+(\d+[,.]?\d*)'
    ]
    
    for pattern in amount_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info['amounts'].extend(matches)
    
    # Extract dates
    date_patterns = [
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
        r'\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
        r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info['dates'].extend(matches)
    
    # Extract invoice numbers
    invoice_patterns = [
        r'(?:facture|invoice|n°|#)\s*:?\s*([A-Z0-9\-]+)',
        r'(?:devis|quote)\s*:?\s*([A-Z0-9\-]+)'
    ]
    
    for pattern in invoice_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        info['invoice_numbers'].extend(matches)
    
    return info