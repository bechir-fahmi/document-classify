import os
import logging
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import config
from easyocr import Reader
from typing import Dict, Any, Optional
from .groq_utils import extract_text_from_image as groq_extract_text, analyze_document_content

# Fix for PIL/Pillow compatibility issue with EasyOCR
try:
    Image.ANTIALIAS
except AttributeError:
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pytesseract path
if hasattr(config, 'TESSERACT_PATH'):
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

# Cache for EasyOCR readers
_easyocr_reader_latin = None
_easyocr_reader_arabic = None

def get_easyocr_reader(lang_set='latin'):
    global _easyocr_reader_latin, _easyocr_reader_arabic
    
    if lang_set == 'latin':
        if _easyocr_reader_latin is None:
            logger.info("Initializing EasyOCR for Latin languages: English, French")
            _easyocr_reader_latin = Reader(['en', 'fr'])
        return _easyocr_reader_latin
    elif lang_set == 'arabic':
        if _easyocr_reader_arabic is None:
            logger.info("Initializing EasyOCR for Arabic: Arabic, English")
            _easyocr_reader_arabic = Reader(['en', 'ar'])
        return _easyocr_reader_arabic
    else:
        raise ValueError(f"Unknown language set: {lang_set}")

def extract_text_from_file(file_path: str) -> str:
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            result = groq_extract_text(file_path)
            if result["success"]:
                return result["text"]
            else:
                logger.warning(f"Groq extraction failed, falling back to Tesseract: {result.get('error')}")
                return extract_text_with_tesseract(file_path)
                
        elif ext == '.pdf':
            return extract_text_from_pdf(file_path)
            
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
    except Exception as e:
        logger.error(f"Error extracting text from file: {str(e)}")
        raise

def extract_text_with_tesseract(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error in Tesseract OCR: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def analyze_document(file_path: str) -> Dict[str, Any]:
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf']:
            result = analyze_document_content(file_path)
            if result["success"]:
                return result["analysis"]
            else:
                logger.warning(f"Groq analysis failed: {result.get('error')}")
                return {}
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return {} 