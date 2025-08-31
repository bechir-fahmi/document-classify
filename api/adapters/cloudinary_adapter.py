"""
Cloudinary Adapter - Adapter Pattern
Adapts Cloudinary utilities to the cloud storage interface
"""
import logging
from typing import Dict, Any

from api.interfaces.document_classifier import ICloudStorageService

logger = logging.getLogger(__name__)


class CloudinaryAdapter(ICloudStorageService):
    """Adapter for Cloudinary cloud storage service"""
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload document to Cloudinary"""
        try:
            from utils.cloudinary_utils import upload_document
            return upload_document(file_path)
        except Exception as e:
            logger.error(f"Error uploading to Cloudinary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "url": None,
                "public_id": None
            }