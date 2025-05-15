import os
import logging
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import config
from easyocr import Reader

# Fix for PIL/Pillow compatibility issue with EasyOCR
# Newer versions of Pillow have replaced ANTIALIAS with Resampling.LANCZOS
try:
    # Check if ANTIALIAS is available
    Image.ANTIALIAS
except AttributeError:
    # If not, create a patch for it using the new Resampling.LANCZOS
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pytesseract path
if hasattr(config, 'TESSERACT_PATH'):
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

# Cache for EasyOCR readers to avoid reloading
_easyocr_reader_latin = None  # For English and French
_easyocr_reader_arabic = None  # For Arabic and English

def get_easyocr_reader(lang_set='latin'):
    """
    Get or create EasyOCR reader with specified languages
    
    Args:
        lang_set: 'latin' for English+French, 'arabic' for Arabic+English
    
    Returns:
        EasyOCR reader instance
    """
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

def extract_text_from_file(file_path=None, file_bytes=None, file_extension=None):
    """
    Extract text from a file based on its extension
    
    Args:
        file_path: Path to the file
        file_bytes: Bytes of the file
        file_extension: File extension
        
    Returns:
        Extracted text
    """
    if file_path:
        file_extension = os.path.splitext(file_path)[1].lower()
    elif not file_extension and file_bytes:
        # Try to detect file type from bytes if needed
        pass
    
    logger.info(f"Processing file with extension: {file_extension}")
    
    if file_extension in ['.pdf']:
        if file_path:
            return extract_text_from_pdf(file_path=file_path)
        else:
            return extract_text_from_pdf(file_bytes=file_bytes)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        if file_path:
            return extract_text_from_image(image_path=file_path)
        else:
            return extract_text_from_image(image_bytes=file_bytes)
    elif file_extension in ['.txt', '.md', '.csv', '.json', '.xml', '.html']:
        return extract_text_from_text_file(file_path, file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def extract_text_from_text_file(file_path=None, file_bytes=None):
    """
    Extract text from a text file
    
    Args:
        file_path: Path to the text file
        file_bytes: Bytes of the text file
        
    Returns:
        Extracted text
    """
    logger.info(f"Extracting text from text file {'file' if file_path else 'bytes'}")
    
    try:
        if file_path:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            # Decode bytes with UTF-8 (fallback to latin-1 if needed)
            try:
                return file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return file_bytes.decode('latin-1', errors='ignore')
    except Exception as e:
        logger.error(f"Error extracting text from text file: {str(e)}")
        raise

def extract_text_from_pdf(file_path=None, file_bytes=None):
    """
    Extract text from a PDF file using PyMuPDF and OCR if needed
    
    Args:
        file_path: Path to the PDF file
        file_bytes: Bytes of the PDF file
        
    Returns:
        Extracted text
    """
    logger.info(f"Extracting text from PDF {'file' if file_path else 'bytes'}")
    
    try:
        # Create a temporary file if we have bytes
        if file_bytes and not file_path:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
                temp.write(file_bytes)
                file_path = temp.name
        
        # Open the PDF with PyMuPDF
        pdf_document = fitz.open(file_path)
        
        # First try direct text extraction
        full_text = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            full_text.append(text)
        
        extracted_text = "\n".join(full_text)
        
        # If direct extraction yields little text, fall back to OCR
        if len(extracted_text.strip()) < config.MIN_TEXT_LENGTH:
            logger.info("Direct text extraction yielded limited text, falling back to OCR")
            
            full_text = []
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Convert the page to an image at a higher DPI for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert pixmap to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Extract text using Tesseract OCR
                text = pytesseract.image_to_string(img, lang='eng+fra+ara')
                full_text.append(text)
            
            extracted_text = "\n".join(full_text)
            
            # If Tesseract OCR still finds limited text, try EasyOCR as a last resort
            if len(extracted_text.strip()) < config.MIN_TEXT_LENGTH:
                logger.info("Tesseract found limited text, trying EasyOCR as final fallback")
                
                full_text = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Save to a temporary file for EasyOCR
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                        img.save(temp.name)
                        
                        # Try with Latin languages first (English and French)
                        latin_reader = get_easyocr_reader('latin')
                        latin_result = latin_reader.readtext(temp.name)
                        latin_text = "\n".join([item[1] for item in latin_result])
                        
                        # Then try with Arabic
                        arabic_reader = get_easyocr_reader('arabic')
                        arabic_result = arabic_reader.readtext(temp.name)
                        arabic_text = "\n".join([item[1] for item in arabic_result])
                        
                        # Combine results
                        page_text = latin_text + "\n" + arabic_text
                        
                        os.unlink(temp.name)  # Clean up temporary file
                    
                    full_text.append(page_text)
                
                extracted_text = "\n".join(full_text)
        
        # Clean up temporary file if we created one
        if file_bytes and not file_path:
            os.unlink(file_path)
        
        # Close the PDF document
        pdf_document.close()
        
        return extracted_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_from_image(image_path=None, image_bytes=None):
    """
    Extract text from an image using OCR
    
    Args:
        image_path: Path to the image file
        image_bytes: Bytes of the image file
        
    Returns:
        Extracted text
    """
    logger.info(f"Extracting text from image {'file' if image_path else 'bytes'}")
    
    try:
        # First try Tesseract
        if image_path:
            text = pytesseract.image_to_string(Image.open(image_path), lang='eng+fra+ara')
        else:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp.write(image_bytes)
                temp_path = temp.name
            
            text = pytesseract.image_to_string(Image.open(temp_path), lang='eng+fra+ara')
            os.unlink(temp_path)  # Clean up temporary file
        
        # If Tesseract doesn't find much text, try EasyOCR as a fallback
        if len(text.strip()) < 20:
            logger.info("Tesseract found limited text, trying EasyOCR as fallback")
            
            if image_path:
                # Try with Latin languages first (English and French)
                latin_reader = get_easyocr_reader('latin')
                latin_result = latin_reader.readtext(image_path)
                latin_text = "\n".join([item[1] for item in latin_result])
                
                # Then try with Arabic
                arabic_reader = get_easyocr_reader('arabic')
                arabic_result = arabic_reader.readtext(image_path)
                arabic_text = "\n".join([item[1] for item in arabic_result])
                
                # Combine results
                text = latin_text + "\n" + arabic_text
            else:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                    temp.write(image_bytes)
                    temp_path = temp.name
                
                    # Try with Latin languages first (English and French)
                    latin_reader = get_easyocr_reader('latin')
                    latin_result = latin_reader.readtext(temp_path)
                    latin_text = "\n".join([item[1] for item in latin_result])
                    
                    # Then try with Arabic
                    arabic_reader = get_easyocr_reader('arabic')
                    arabic_result = arabic_reader.readtext(temp_path)
                    arabic_text = "\n".join([item[1] for item in arabic_result])
                    
                    # Combine results
                    text = latin_text + "\n" + arabic_text
                
                os.unlink(temp_path)  # Clean up temporary file
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise 