"""
Classification Controller - Single Responsibility Principle
Handles document classification API endpoints
"""
import os
import tempfile
import logging
import uuid
import time
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Optional, Dict, Any

from api.services.document_classification_service import DocumentClassificationService
from api.models.responses import ClassificationResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["classification"])


class ClassificationController:
    """Controller for document classification endpoints"""
    
    def __init__(self, classification_service: DocumentClassificationService):
        self._classification_service = classification_service
    
    def setup_routes(self, router: APIRouter):
        """Setup classification routes"""
        
        @router.post("/", response_model=ClassificationResponse)
        async def classify_document(
            file: UploadFile = File(...),
            model_type: Optional[str] = Query(None, description="Model type to use"),
            upload_to_cloud: bool = Query(True, description="Upload to cloud storage")
        ):
            """
            Classify a document and optionally upload to cloud storage
            
            Args:
                file: Document file (PDF, image, text)
                model_type: Optional model type override
                upload_to_cloud: Whether to upload to cloud storage
                
            Returns:
                Document classification result with metadata
            """
            start_time = time.time()
            
            try:
                # Generate unique document ID
                document_id = str(uuid.uuid4())
                
                # Save uploaded file temporarily
                temp_file_path = f"temp_{document_id}_{file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                try:
                    # Classify document
                    result = self._classification_service.classify_document(
                        temp_file_path, 
                        upload_to_cloud=upload_to_cloud
                    )
                    
                    return ClassificationResponse(**result)
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error in document classification: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error classifying document: {str(e)}"
                )
        
        @router.get("/models")
        async def list_available_models():
            """List available classification models"""
            import config
            return {
                "default_model": config.DEFAULT_MODEL,
                "available_models": ["sklearn_tfidf_svm", "bert", "layoutlm"],
                "document_classes": config.DOCUMENT_CLASSES
            }
        
        @router.get("/embedding-info")
        async def get_embedding_info():
            """Get information about the embedding model"""
            try:
                from utils.embedding_utils import get_embedder
                
                embedder = get_embedder()
                return {
                    "embedding_model": embedder.model_name,
                    "embedding_dimension": embedder.get_embedding_dimension(),
                    "description": "Document embeddings using sentence-transformers"
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error getting embedding info: {str(e)}"
                )