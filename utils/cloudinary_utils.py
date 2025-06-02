import cloudinary
import cloudinary.uploader
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

def upload_document(file_path: str, folder: str = "documents") -> Dict[str, Any]:
    """
    Upload a document to Cloudinary and return the upload response
    
    Args:
        file_path: Path to the document file
        folder: Cloudinary folder to store the document in
        
    Returns:
        Dictionary containing upload response including the URL
    """
    try:
        # Upload the file
        result = cloudinary.uploader.upload(
            file_path,
            folder=folder,
            resource_type="raw",  # This allows uploading any file type
            use_filename=True,
            unique_filename=True
        )
        
        return {
            "success": True,
            "url": result.get("secure_url"),
            "public_id": result.get("public_id"),
            "format": result.get("format"),
            "resource_type": result.get("resource_type")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_document_url(public_id: str) -> Optional[str]:
    """
    Get the URL of a document from its public_id
    
    Args:
        public_id: The public_id of the document in Cloudinary
        
    Returns:
        The secure URL of the document if found, None otherwise
    """
    try:
        result = cloudinary.api.resource(public_id)
        return result.get("secure_url")
    except Exception:
        return None

def delete_document(public_id: str) -> Dict[str, Any]:
    """
    Delete a document from Cloudinary
    
    Args:
        public_id: The public_id of the document to delete
        
    Returns:
        Dictionary containing the deletion response
    """
    try:
        result = cloudinary.uploader.destroy(public_id)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 