"""
Financial Analyzer Adapter - Adapter Pattern
Adapts existing financial analysis utilities to the interface
"""
import logging
from typing import Dict, Any

from api.interfaces.document_classifier import IFinancialAnalyzer

logger = logging.getLogger(__name__)


class FinancialAnalyzerAdapter(IFinancialAnalyzer):
    """Adapter for existing financial analysis utilities"""
    
    def analyze_financial_data(self, text: str, document_type: str, document_id: str) -> Dict[str, Any]:
        """Analyze financial information from document text"""
        try:
            from utils.financial_analyzer import analyze_document_for_bilan
            
            # Analyze using existing utility
            financial_transaction = analyze_document_for_bilan(text, document_type, document_id)
            
            # Convert to dictionary format
            return {
                "document_type": financial_transaction.document_type,
                "amount": financial_transaction.amount,
                "currency": financial_transaction.currency,
                "date": financial_transaction.date.isoformat() if financial_transaction.date else None,
                "description": financial_transaction.description,
                "category": financial_transaction.category,
                "subcategory": financial_transaction.subcategory,
                "document_id": financial_transaction.document_id,
                "confidence": financial_transaction.confidence
            }
        except Exception as e:
            logger.error(f"Error in financial analysis: {str(e)}")
            return {
                "document_type": document_type,
                "amount": 0.0,
                "currency": "EUR",
                "date": None,
                "description": "Error in analysis",
                "category": "unknown",
                "subcategory": "unknown",
                "document_id": document_id,
                "confidence": 0.0
            }


class GroqFinancialAnalyzerAdapter(IFinancialAnalyzer):
    """Adapter for Groq-powered financial analysis"""
    
    def analyze_financial_data(self, text: str, document_type: str, document_id: str) -> Dict[str, Any]:
        """Analyze financial information using Groq AI"""
        try:
            from utils.groq_financial_analyzer import analyze_document_with_groq
            
            # Analyze using Groq
            groq_result = analyze_document_with_groq(text, document_type, document_id)
            
            # Convert to dictionary format
            return {
                "document_type": groq_result.document_type,
                "amount": groq_result.amount,
                "currency": groq_result.currency,
                "date": groq_result.date.isoformat() if groq_result.date else None,
                "description": groq_result.description,
                "category": groq_result.category,
                "subcategory": groq_result.subcategory,
                "document_id": groq_result.document_id,
                "confidence": groq_result.confidence,
                "raw_groq_response": groq_result.raw_groq_response,
                "line_items": getattr(groq_result, 'line_items', None),
                "tax_amount": getattr(groq_result, 'tax_amount', None),
                "subtotal": getattr(groq_result, 'subtotal', None),
                "payment_terms": getattr(groq_result, 'payment_terms', None),
                "vendor_customer": getattr(groq_result, 'vendor_customer', None)
            }
        except Exception as e:
            logger.error(f"Error in Groq financial analysis: {str(e)}")
            return {
                "document_type": document_type,
                "amount": 0.0,
                "currency": "EUR",
                "date": None,
                "description": "Error in Groq analysis",
                "category": "unknown",
                "subcategory": "unknown",
                "document_id": document_id,
                "confidence": 0.0,
                "raw_groq_response": {"error": str(e)}
            }