"""
Document Analyzer Adapter - Adapter Pattern
Adapts existing document analysis utilities to the interface
"""
import logging
from typing import Dict, Any

from api.interfaces.document_classifier import IDocumentAnalyzer

logger = logging.getLogger(__name__)


class DocumentAnalyzerAdapter(IDocumentAnalyzer):
    """Adapter for existing document analysis utilities"""
    
    def analyze(self, file_path: str) -> str:
        """Analyze document and return document type"""
        try:
            from utils.document_analyzer import analyze_document
            return analyze_document(file_path)
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return "❓ Unknown Document Type"
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from document text"""
        try:
            from utils.document_analyzer import extract_document_info
            return extract_document_info(text)
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}