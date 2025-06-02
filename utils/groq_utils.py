import os
import base64
from typing import Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

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
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing the extracted text and metadata
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        # Create the chat completion request
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Format the response as a JSON object with a 'text' field containing the extracted text and a 'confidence' field indicating your confidence in the extraction (0-1)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,  # Lower temperature for more focused extraction
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response = completion.choices[0].message.content
        
        return {
            "success": True,
            "text": response,
            "model": "groq-vision"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "model": "groq-vision"
        }

def analyze_document_content(image_path: str) -> Dict[str, Any]:
    """
    Analyze document content using Groq's vision capabilities
    
    Args:
        image_path: Path to the document image
        
    Returns:
        Dictionary containing document analysis results
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)
        
        # Create the chat completion request
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this document and provide the following information in JSON format:\n1. Document type (invoice, receipt, etc.)\n2. Key information (dates, amounts, names, etc.)\n3. Confidence score for the analysis\n4. Any notable features or patterns"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response = completion.choices[0].message.content
        
        return {
            "success": True,
            "analysis": response,
            "model": "groq-vision"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis": {},
            "model": "groq-vision"
        } 