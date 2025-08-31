"""
Document Classification API - Refactored FastAPI Application
Following SOLID Principles
Author: Bachir Fahmi
Email: bachir.fahmi@example.com
Description: Clean, maintainable document classification API
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.factories.service_factory import service_factory
from api.controllers.classification_controller import router as classification_router
from api.controllers.financial_controller import router as financial_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application
    Following the Factory Pattern and Dependency Injection
    """
    
    # Create FastAPI app
    app = FastAPI(
        title="Document Classification API",
        description="Production-ready document classification system using machine learning to automatically classify business documents with 95.5% accuracy. Developed by Bachir Fahmi.",
        version="2.0.0",
        contact={
            "name": "Bachir Fahmi",
            "email": "bachir.fahmi@example.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup controllers and routes
    setup_routes(app)
    
    # Setup error handlers
    setup_error_handlers(app)
    
    return app


def setup_routes(app: FastAPI):
    """Setup API routes using dependency injection"""
    
    # Get controllers from factory
    classification_controller = service_factory.get_classification_controller()
    financial_controller = service_factory.get_financial_controller()
    
    # Setup controller routes
    classification_controller.setup_routes(classification_router)
    financial_controller.setup_routes(financial_router)
    
    # Include routers
    app.include_router(classification_router)
    app.include_router(financial_router)
    
    # Basic endpoints
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {"message": "Document Classification API v2.0 is running"}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "ok", "version": "2.0.0"}
    
    # Add backward compatibility routes
    add_legacy_routes(app)


def add_legacy_routes(app: FastAPI):
    """Add legacy routes for backward compatibility"""
    from fastapi import File, UploadFile, Query
    from typing import List, Dict, Any
    import os
    import uuid
    
    @app.post("/bilan")
    async def process_bilan_documents(request: Dict[str, Any]):
        """
        Download documents from Cloudinary URLs, extract text, and generate a financial bilan
        
        Request Body:
        {
            "documents": [
                {
                    "id": "doc-uuid",
                    "filename": "document.pdf",
                    "document_type": "invoice", // invoice, receipt, bank_statement, expense, etc.
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
        
        Returns:
            Structured financial bilan with assets, liabilities, and equity
        """
        import time
        import shutil
        import requests
        
        start_time = time.time()
        
        try:
            # Get documents and business info from request
            documents = request.get("documents", [])
            business_info = request.get("business_info", {})
            period_days = request.get("period_days", 90)
            
            if not documents:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="No documents provided")
            
            # Create unique session folder for this request
            session_id = str(uuid.uuid4())[:8]
            download_folder = f"downloaddoc/session_{session_id}"
            
            # Create session folder
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)
            
            logger.info(f"Created session folder: {download_folder}")
            
            downloaded_files = []
            
            logger.info(f"Processing {len(documents)} documents for bilan generation")
            
            # Download all documents for this session
            for doc in documents:
                try:
                    # Download document from Cloudinary
                    cloudinary_url = doc.get('cloudinaryUrl', '')
                    if not cloudinary_url:
                        logger.warning(f"No Cloudinary URL for {doc.get('filename', 'unknown')}")
                        continue
                    
                    # Download the file
                    response = requests.get(cloudinary_url)
                    if response.status_code != 200:
                        logger.warning(f"Failed to download {doc.get('filename', 'unknown')}")
                        continue
                    
                    # Save file in session folder
                    saved_file_path = os.path.join(download_folder, f"{doc['id']}_{doc['filename']}")
                    
                    with open(saved_file_path, "wb") as f:
                        f.write(response.content)
                    
                    downloaded_files.append({
                        'file_path': saved_file_path,
                        'original_doc': doc
                    })
                    logger.info(f"Downloaded: {saved_file_path}")
                    
                except Exception as e:
                    logger.error(f"Error downloading {doc.get('filename', 'unknown')}: {str(e)}")
                    continue
            
            if not downloaded_files:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail="No files could be downloaded")
            
            # Process files using the financial service
            logger.info("Processing files for bilan generation")
            
            # Generate bilan using the direct file processing method
            financial_service = service_factory.get_financial_service()
            bilan = financial_service.generate_bilan_from_files_directly(
                downloaded_files, business_info, period_days
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Cleanup: Remove session folder after processing
            try:
                shutil.rmtree(download_folder)
                logger.info(f"Cleaned up session folder: {download_folder}")
            except Exception as e:
                logger.warning(f"Could not cleanup folder: {e}")
            
            return {
                "session_id": session_id,
                "processed_documents": len(downloaded_files),
                "processing_time_ms": processing_time,
                "business_info": business_info,
                **bilan  # Spread the bilan object directly into the response
            }
            
        except Exception as e:
            logger.error(f"Error in bilan processing: {str(e)}")
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Error processing bilan documents: {str(e)}")
    
    @app.post("/generate-bilan")
    async def generate_bilan_from_files(
        files: List[UploadFile] = File(...),
        period_days: int = Query(30, description="Number of days to include in analysis")
    ):
        """Generate bilan from uploaded files (alternative endpoint)"""
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
                # Generate bilan using service
                financial_service = service_factory.get_financial_service()
                result = financial_service.generate_financial_bilan(
                    temp_file_paths, 
                    period_days
                )
                
                return result
                
            finally:
                # Clean up temporary files
                for temp_file_path in temp_file_paths:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
        except Exception as e:
            logger.error(f"Error in file-based bilan endpoint: {str(e)}")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating bilan: {str(e)}"
            )
    
    @app.post("/analyze-financial")
    async def analyze_financial_legacy(file: UploadFile = File(...)):
        """Legacy financial analysis endpoint for backward compatibility"""
        try:
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            
            # Save uploaded file temporarily
            temp_file_path = f"temp_{document_id}_{file.filename}"
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            try:
                # Analyze document using service
                financial_service = service_factory.get_financial_service()
                result = financial_service.analyze_document_financial(temp_file_path)
                
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
            logger.error(f"Error in legacy financial analysis endpoint: {str(e)}")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=500, 
                detail=f"Error analyzing document: {str(e)}"
            )


def setup_error_handlers(app: FastAPI):
    """Setup global error handlers"""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler"""
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)}
        )


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    import config
    
    logger.info(f"Starting refactored API server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)