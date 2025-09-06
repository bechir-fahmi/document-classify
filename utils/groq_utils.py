import os
import base64
from typing import Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from dateutil import parser

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    """
    Extract text from an image using Groq's vision capabilities
    Note: Groq vision models are currently being deprecated, so this falls back to Tesseract
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing the extracted text and metadata
    """
    # Groq vision models are currently deprecated/decommissioned
    # Return failure to trigger fallback to Tesseract OCR
    return {
        "success": False,
        "error": "Groq vision models are currently deprecated",
        "text": "",
        "model": "groq-vision-disabled"
    }

def analyze_document_content(image_path: str) -> Dict[str, Any]:
    """
    Analyze document content using Groq's vision capabilities
    Note: Groq vision models are currently being deprecated, so this is disabled
    
    Args:
        image_path: Path to the document image
        
    Returns:
        Dictionary containing document analysis results
    """
    # Groq vision models are currently deprecated/decommissioned
    # Return failure to skip vision analysis
    return {
        "success": False,
        "error": "Groq vision models are currently deprecated",
        "analysis": {},
        "model": "groq-vision-disabled"
    }

def extract_document_info_with_groq(text: str, doc_type: str) -> Dict[str, Any]:
    """
    Extract document information using Groq AI
    
    Args:
        text: Document text content
        doc_type: Detected document type
        
    Returns:
        Dictionary with extracted information
    """
    prompt = f"""
    Extract ONLY the following information from this document:
    1. Date (any date found in the document)
    2. Client name (found after "Bill To:" or similar)
    3. Client address (found after the client name)
    
    Document text:
    {text}
    
    Return the information in JSON format with only these fields if found:
    {{
        "date": "extracted date",
        "client_name": "extracted client name",
        "client_address": "extracted client address"
    }}
    
    If a field is not found, do not include it in the response.
    """
    
    try:
        # Call Groq AI
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Using the same model as text extraction
            messages=[
                {"role": "system", "content": "You are a document analysis expert. Extract dates and client information from documents and return it in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=1000,
            response_format={"type": "json_object"}  # Ensure JSON response
        )
        
        # Parse the response
        extracted_info = response.choices[0].message.content
        
        # Convert string response to dictionary
        import json
        try:
            extracted_data = json.loads(extracted_info)
            
            # Standardize date format
            if "date" in extracted_data and extracted_data["date"]:
                try:
                    # Attempt to parse the date string
                    date_obj = parser.parse(extracted_data["date"])
                    # Format to YYYY-MM-DD
                    extracted_data["date"] = date_obj.strftime('%Y-%m-%d')
                except (parser.ParserError, TypeError, ValueError) as e:
                    # If parsing fails, print a message and keep the original date string
                    print(f"Could not parse date: '{extracted_data['date']}'. Error: {e}")
                    # Pass to keep the original date string if it's unparseable
                    pass
            
            return extracted_data
        except json.JSONDecodeError:
            # If Groq AI doesn't return valid JSON, return empty dict
            print(f"Failed to decode JSON from Groq response: {extracted_info}")
            return {}
            
    except Exception as e:
        print(f"Error in Groq AI extraction: {str(e)}")
        return {}