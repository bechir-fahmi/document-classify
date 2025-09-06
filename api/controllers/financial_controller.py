"""
Financial Controller - Single Responsibility Principle
Handles financial analysis API endpoints
"""
import os
import logging
import uuid
import time
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import List, Dict, Any

from api.services.financial_analysis_service import FinancialAnalysisService
from api.models.responses import (
    FinancialTransactionResponse, 
    FinancialBilanResponse,
    DocumentFinancialAnalysisResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/financial", tags=["financial"])


class FinancialController:
    """Controller for financial analysis endpoints"""
    
    def __init__(self, financial_service: FinancialAnalysisService):
        self._financial_service = financial_service
    
    def setup_routes(self, router: APIRouter):
        """Setup financial analysis routes"""
        
        @router.post("/analyze", response_model=Dict[str, Any])
        async def analyze_financial_document(file: UploadFile = File(...)):
            """
            Analyze a document for financial information using Groq AI
            
            Args:
                file: Document file (PDF, image, text)
                
            Returns:
                Comprehensive financial analysis using Groq AI
            """
            try:
                # Generate unique document ID
                document_id = str(uuid.uuid4())
                
                # Save uploaded file temporarily
                temp_file_path = f"temp_{document_id}_{file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                try:
                    # Analyze document
                    result = self._financial_service.analyze_document_financial(temp_file_path)
                    
                    return {
                        "document_id": result["document_id"],
                        "groq_financial_analysis": result["financial_analysis"],
                        "processing_time_ms": result["processing_time_ms"]
                    }
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error in financial analysis: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error analyzing document: {str(e)}"
                )
        
        @router.post("/bilan", response_model=FinancialBilanResponse)
        async def generate_financial_bilan(
            files: List[UploadFile] = File(...),
            period_days: int = Query(30, description="Number of days to include in analysis")
        ):
            """
            Generate a financial bilan from multiple documents
            
            Args:
                files: List of document files to analyze
                period_days: Number of days to include in analysis
                
            Returns:
                Financial bilan with summary and recommendations
            """
            try:
                # Save all files temporarily
                temp_file_paths = []
                
                for file in files:
                    document_id = str(uuid.uuid4())
                    temp_file_path = f"temp_{document_id}_{file.filename}"
                    
                    with open(temp_file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)
                    
                    temp_file_paths.append(temp_file_path)
                
                try:
                    # Generate bilan
                    result = self._financial_service.generate_financial_bilan(
                        temp_file_paths, 
                        period_days
                    )
                    
                    return FinancialBilanResponse(**result)
                    
                finally:
                    # Clean up temporary files
                    for temp_file_path in temp_file_paths:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
            except Exception as e:
                logger.error(f"Error generating bilan: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error generating bilan: {str(e)}"
                )
        
        @router.post("/v2/bilan")
        async def generate_v2_bilan(request: Dict[str, Any]):
            """
            Generate V2 bilan with structured format from Cloudinary URLs
            
            Request Body:
            {
                "documents": [
                    {
                        "id": "doc-uuid",
                        "filename": "document.pdf",
                        "document_type": "invoice",
                        "cloudinaryUrl": "https://cloudinary-url",
                        "created_at": "2025-01-15T10:30:00.000Z"
                    }
                ],
                "period_days": 90,
                "business_info": {
                    "name": "Company Name",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31"
                }
            }
            
            Returns structured bilan format with dates, actifs, capitaux_propres_et_passifs, etc.
            """
            try:
                documents = request.get("documents", [])
                business_info = request.get("business_info", {})
                period_days = request.get("period_days", 90)
                
                if not documents:
                    raise HTTPException(status_code=400, detail="No documents provided")
                
                # Download and process documents
                temp_file_paths = []
                downloaded_files = []
                
                for doc in documents:
                    cloudinary_url = doc.get('cloudinaryUrl', '')
                    if not cloudinary_url:
                        continue
                    
                    # Download file
                    import requests
                    response = requests.get(cloudinary_url)
                    if response.status_code != 200:
                        continue
                    
                    # Save temporarily
                    temp_file_path = f"temp_{doc['id']}_{doc['filename']}"
                    with open(temp_file_path, "wb") as f:
                        f.write(response.content)
                    
                    temp_file_paths.append(temp_file_path)
                    downloaded_files.append({
                        'file_path': temp_file_path,
                        'original_doc': doc
                    })
                
                try:
                    # Generate V2 structured bilan
                    result = self._financial_service.generate_v2_structured_bilan(
                        downloaded_files, 
                        business_info,
                        period_days
                    )
                    
                    return result
                    
                finally:
                    # Clean up
                    for temp_file_path in temp_file_paths:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing V2 bilan: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing V2 bilan: {str(e)}"
                )

        @router.post("/bilan-from-urls")
        async def process_bilan_from_cloudinary_urls(request: Dict[str, Any]):
            """
            Generate bilan from Cloudinary URLs
            
            Request Body:
            {
                "documents": [
                    {
                        "id": "doc-uuid",
                        "filename": "document.pdf",
                        "document_type": "invoice",
                        "cloudinaryUrl": "https://cloudinary-url",
                        "created_at": "2025-01-15T10:30:00.000Z"
                    }
                ],
                "period_days": 90,
                "business_info": {
                    "name": "Company Name",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31"
                }
            }
            """
            try:
                documents = request.get("documents", [])
                business_info = request.get("business_info", {})
                period_days = request.get("period_days", 90)
                
                if not documents:
                    raise HTTPException(status_code=400, detail="No documents provided")
                
                # Download and process documents
                temp_file_paths = []
                
                for doc in documents:
                    cloudinary_url = doc.get('cloudinaryUrl', '')
                    if not cloudinary_url:
                        continue
                    
                    # Download file
                    import requests
                    response = requests.get(cloudinary_url)
                    if response.status_code != 200:
                        continue
                    
                    # Save temporarily
                    temp_file_path = f"temp_{doc['id']}_{doc['filename']}"
                    with open(temp_file_path, "wb") as f:
                        f.write(response.content)
                    
                    temp_file_paths.append(temp_file_path)
                
                try:
                    # Generate bilan
                    result = self._financial_service.generate_financial_bilan(
                        temp_file_paths, 
                        period_days
                    )
                    
                    return {
                        "business_info": business_info,
                        **result
                    }
                    
                finally:
                    # Clean up
                    for temp_file_path in temp_file_paths:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing bilan from URLs: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error processing bilan: {str(e)}"
                )