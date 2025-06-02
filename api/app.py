import os
import tempfile
import logging
import uuid
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import config
from utils.text_extraction import extract_text_from_file
from preprocessor.text_processor import preprocess_for_model
from models import get_model
from utils.cloudinary_utils import upload_document
from utils.document_analyzer import analyze_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Classification API",
    description="API for classifying documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class ClassificationResponse(BaseModel):
    document_id: str
    prediction: str
    confidence: float
    confidence_scores: Dict[str, float]
    text_excerpt: str
    processing_time_ms: float
    cloudinary_url: Optional[str] = None
    public_id: Optional[str] = None

class CommercialDocumentResponse(BaseModel):
    document_id: str
    document_type: str
    confidence: float
    extracted_info: Dict[str, Any]
    type_scores: Dict[str, float]
    text_excerpt: str
    processing_time_ms: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Initialize model cache
model_cache = {}

def get_cached_model(model_type=None):
    """Get a cached model instance or create a new one"""
    if model_type is None:
        model_type = config.DEFAULT_MODEL
    
    if model_type not in model_cache:
        model_cache[model_type] = get_model(model_type)
    
    return model_cache[model_type]

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Document Classification API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "default_model": config.DEFAULT_MODEL,
        "available_models": ["sklearn_tfidf_svm", "bert", "layoutlm"],
        "document_classes": config.DOCUMENT_CLASSES
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(
    file: UploadFile = File(...),
    model_type: Optional[str] = Form(None)
):
    """
    Classify a document
    
    Args:
        file: Document file (PDF, image)
        model_type: Type of model to use (optional)
        
    Returns:
        Classification result
    """
    start_time = time.time()
    
    try:
        # Generate a unique ID for this document
        document_id = str(uuid.uuid4())
        
        # Create temp file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp_path = temp.name
            # Write uploaded file to temp file
            contents = await file.read()
            temp.write(contents)
        
        try:
            # Upload to Cloudinary
            upload_result = upload_document(temp_path)
            
            # Determine which model to use - default if none specified
            if model_type is None:
                model_type = config.DEFAULT_MODEL
            
            # Extract text from file
            logger.info(f"Extracting text from {file.filename}")
            text = extract_text_from_file(file_path=temp_path)
            
            # Preprocess text for the model
            logger.info("Preprocessing text")
            processed_text = preprocess_for_model(text, model_type=model_type)
            
            # Get the model
            logger.info(f"Getting model: {model_type}")
            model = get_cached_model(model_type)
            
            # Get prediction
            logger.info("Making prediction")
            # Use hybrid_predict if available
            if hasattr(model, 'hybrid_predict'):
                result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                # Add processing time and Cloudinary info
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                result['cloudinary_url'] = upload_result.get('url')
                result['public_id'] = upload_result.get('public_id')
                return JSONResponse(content=result)
            else:
                result = model.predict(processed_text)
                processing_time_ms = (time.time() - start_time) * 1000
                return ClassificationResponse(
                    document_id=document_id,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    confidence_scores=result["confidence_scores"],
                    text_excerpt=text[:500] + "..." if len(text) > 500 else text,
                    processing_time_ms=processing_time_ms,
                    cloudinary_url=upload_result.get('url'),
                    public_id=upload_result.get('public_id')
                )
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error classifying document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error classifying document: {str(e)}"
        )

@app.post("/classify-commercial", response_model=CommercialDocumentResponse)
async def classify_commercial_document(
    file: UploadFile = File(...),
    model_type: Optional[str] = Form(None)
):
    """
    Classify a commercial document (invoice, quote, purchase order, etc.)
    
    Args:
        file: Document file (PDF, image)
        model_type: Type of model to use for additional verification (optional)
        
    Returns:
        Commercial document classification result with extracted information
    """
    start_time = time.time()
    
    try:
        # Generate a unique ID for this document
        document_id = str(uuid.uuid4())
        
        # Create temp file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp_path = temp.name
            # Write uploaded file to temp file
            contents = await file.read()
            temp.write(contents)
        
        try:
            # Use our specialized document analyzer
            logger.info(f"Analyzing commercial document: {file.filename}")
            analysis = analyze_document(temp_path)
            
            # Get model prediction if requested
            if model_type:
                # Get the model
                logger.info(f"Getting model: {model_type}")
                model = get_cached_model(model_type)
                
                # Extract text from file
                text = extract_text_from_file(file_path=temp_path)
                
                # Preprocess text for the model
                processed_text = preprocess_for_model(text, model_type=model_type)
                
                # Use hybrid_predict if available
                if hasattr(model, 'hybrid_predict'):
                    model_result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                    # Add model hybrid fields to extracted_info
                    analysis['extracted_info'].update({
                        'model_prediction': model_result['model_prediction'],
                        'model_confidence': model_result['model_confidence'],
                        'rule_based_prediction': model_result['rule_based_prediction'],
                        'final_prediction': model_result['final_prediction'],
                        'confidence_flag': model_result['confidence_flag'],
                        'confidence_scores': model_result['confidence_scores'],
                    })
                    # Optionally, override document_type/confidence if hybrid logic is more confident
                    if model_result['confidence_flag'] == 'ok':
                        analysis['document_type'] = model_result['final_prediction']
                        analysis['confidence'] = model_result['model_confidence']
                else:
                    model_result = model.predict(processed_text)
                    analysis['extracted_info']['model_prediction'] = model_result['prediction']
                    analysis['extracted_info']['model_confidence'] = model_result['confidence']
                    
                    # If document_type is unknown, try using the model's prediction
                    # Only if the model prediction is one of our commercial document types
                    commercial_types = ["invoice", "quote", "purchase_order", "delivery_note", 
                                       "receipt", "bank_statement", "expense_report", "payslip"]
                    if (analysis['document_type'] == "unknown" and 
                        model_result['prediction'] in commercial_types and
                        model_result['confidence'] >= 0.2):
                        analysis['document_type'] = model_result['prediction']
                        analysis['confidence'] = model_result['confidence']
                        logger.info(f"Using model prediction as document type: {model_result['prediction']}")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            return CommercialDocumentResponse(
                document_id=document_id,
                document_type=analysis['document_type'],
                confidence=analysis['confidence'],
                extracted_info=analysis['extracted_info'],
                type_scores=analysis['type_scores'],
                text_excerpt=analysis['text'][:500] + "..." if len(analysis['text']) > 500 else analysis['text'],
                processing_time_ms=processing_time_ms
            )
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"Error analyzing commercial document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing commercial document: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Error processing request", "detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler for general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )