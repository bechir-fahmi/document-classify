"""
Text Extractor Adapter - Adapter Pattern
Adapts existing text extraction utilities to the interface
"""
import logging

from api.interfaces.document_classifier import ITextExtractor

logger = logging.getLogger(__name__)


class TextExtractorAdapter(ITextExtractor):
    """Adapter for existing text extraction utilities"""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from document file"""
        try:
            from utils.text_extraction import extract_text_from_file
            return extract_text_from_file(file_path)
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""