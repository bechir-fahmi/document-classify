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
from utils.document_analyzer import analyze_document, extract_document_info
from utils.groq_utils import extract_document_info_with_groq
from utils.embedding_utils import embed_document
from utils.financial_analyzer import FinancialReportGenerator, analyze_document_for_bilan, FinancialTransaction
from utils.groq_financial_analyzer import analyze_document_with_groq, GroqFinancialTransaction
import cloudinary
import cloudinary.uploader

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
    model_prediction: str
    model_confidence: float
    rule_based_prediction: str
    final_prediction: str
    confidence_flag: str
    confidence_scores: Dict[str, float]
    text_excerpt: str
    processing_time_ms: float
    cloudinary_url: str
    public_id: str
    extracted_info: Optional[Dict[str, Any]] = None
    document_embedding: List[float]
    embedding_model: str

class CommercialDocumentResponse(BaseModel):
    document_id: str
    document_type: str
    confidence: float
    extracted_info: Dict[str, Any]
    type_scores: Dict[str, float]
    text_excerpt: str
    processing_time_ms: float
    document_embedding: List[float]
    embedding_model: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class FinancialTransactionResponse(BaseModel):
    document_type: str
    amount: float
    currency: str
    date: Optional[str] = None
    description: str
    category: str
    subcategory: str
    document_id: str
    confidence: float

class GroqFinancialTransactionResponse(BaseModel):
    document_type: str
    amount: float
    currency: str
    date: Optional[str] = None
    description: str
    category: str
    subcategory: str
    document_id: str
    confidence: float
    raw_groq_response: Dict[str, Any]
    line_items: Optional[List[Dict[str, Any]]] = None
    tax_amount: Optional[float] = None
    subtotal: Optional[float] = None
    payment_terms: Optional[str] = None
    vendor_customer: Optional[str] = None

class FinancialBilanResponse(BaseModel):
    period: Dict[str, Any]
    summary: Dict[str, float]
    currency_breakdown: Dict[str, Dict[str, float]]
    document_analysis: Dict[str, Dict[str, Any]]
    transaction_count: int
    recommendations: List[str]
    generated_at: str

class DocumentFinancialAnalysisResponse(BaseModel):
    document_id: str
    financial_transaction: FinancialTransactionResponse
    document_classification: Dict[str, Any]
    document_embedding: List[float]
    embedding_model: str
    processing_time_ms: float

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

@app.get("/embedding-info")
async def get_embedding_info():
    """Get information about the embedding model"""
    from utils.embedding_utils import get_embedder
    
    try:
        embedder = get_embedder()
        return {
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.get_embedding_dimension(),
            "description": "Document embeddings using sentence-transformers"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding info: {str(e)}")

@app.post("/analyze-financial", response_model=Dict[str, Any])
async def analyze_document_financial_groq(file: UploadFile = File(...)):
    """
    Analyze a document for financial information using Groq AI
    
    Args:
        file: Document file (PDF, image, text)
        
    Returns:
        Comprehensive financial analysis using Groq AI
    """
    start_time = time.time()
    
    try:
        # Generate a unique ID for this document
        document_id = str(uuid.uuid4())
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{document_id}_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Extract text from the document
            text = extract_text_from_file(temp_file_path)
            
            # Classify the document first
            doc_type = analyze_document(temp_file_path)
            model = get_cached_model()
            model_result = model.predict(text)
            
            # Use improved classification logic
            final_doc_type = doc_type.replace("❓ Unknown Document Type", "unknown")
            if final_doc_type == "unknown":
                if hasattr(model, 'hybrid_predict'):
                    hybrid_result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                    final_doc_type = hybrid_result['final_prediction']
                elif model_result["confidence"] > 0.2:
                    final_doc_type = model_result["prediction"]
                else:
                    best_prediction = max(model_result["confidence_scores"].items(), key=lambda x: x[1])
                    if best_prediction[1] > 0.25:
                        final_doc_type = best_prediction[0]
            
            # Use Groq for financial analysis
            logger.info(f"Using Groq for financial analysis of {final_doc_type}")
            groq_financial = analyze_document_with_groq(text, final_doc_type, document_id)
            
            # Generate document embedding
            embedding_model_name = "all-MiniLM-L6-v2"
            document_embedding = embed_document(text, embedding_model_name)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare comprehensive response
            return {
                "document_id": document_id,
                "groq_financial_analysis": {
                    "document_type": groq_financial.document_type,
                    "amount": groq_financial.amount,
                    "currency": groq_financial.currency,
                    "date": groq_financial.date.isoformat() if groq_financial.date else None,
                    "description": groq_financial.description,
                    "category": groq_financial.category,
                    "subcategory": groq_financial.subcategory,
                    "confidence": groq_financial.confidence,
                    "raw_groq_response": groq_financial.raw_groq_response
                },
                # "document_classification": {
                #     "rule_based_prediction": doc_type,
                #     "model_prediction": model_result["prediction"],
                #     "model_confidence": model_result["confidence"],
                #     "final_prediction": final_doc_type,
                #     "confidence_scores": model_result["confidence_scores"]
                # },
                # "document_embedding": document_embedding,
                # "embedding_model": embedding_model_name,
                "processing_time_ms": processing_time,
                # "extraction_method": "groq_ai"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in Groq financial analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document with Groq: {str(e)}")
# need to delete after fix
@app.post("/test-analyze-financial", response_model=DocumentFinancialAnalysisResponse)
async def analyze_document_financial(file: UploadFile = File(...)):
    """
    Analyze a document for both classification and financial information
    
    Args:
        file: Document file (PDF, image, text)
        
    Returns:
        Combined document classification and financial analysis
    """
    start_time = time.time()
    
    try:
        # Generate a unique ID for this document
        document_id = str(uuid.uuid4())
        
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{document_id}_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Extract text from the document
            text = extract_text_from_file(temp_file_path)
            
            # Classify the document
            doc_type = analyze_document(temp_file_path)
            
            # Get model prediction for better classification
            model = get_cached_model()
            model_result = model.predict(text)
            
            # Use the better prediction - same logic as /classify endpoint
            final_doc_type = doc_type.replace("❓ Unknown Document Type", "unknown")
            
            if final_doc_type == "unknown":
                # Use the model's hybrid_predict if available (same as /classify)
                if hasattr(model, 'hybrid_predict'):
                    hybrid_result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                    final_doc_type = hybrid_result['final_prediction']
                    logger.info(f"Using hybrid prediction: {final_doc_type}")
                elif model_result["confidence"] > 0.2:  # Lower threshold like /classify
                    final_doc_type = model_result["prediction"]
                    logger.info(f"Using model prediction: {final_doc_type} (confidence: {model_result['confidence']:.3f})")
                else:
                    # Find the highest confidence prediction
                    best_prediction = max(model_result["confidence_scores"].items(), key=lambda x: x[1])
                    if best_prediction[1] > 0.25:
                        final_doc_type = best_prediction[0]
                        logger.info(f"Using best confidence prediction: {final_doc_type} (confidence: {best_prediction[1]:.3f})")
            
            logger.info(f"Final document type for financial analysis: {final_doc_type}")
            
            # Analyze financial information
            financial_transaction = analyze_document_for_bilan(text, final_doc_type, document_id)
            
            # Generate document embedding
            embedding_model_name = "all-MiniLM-L6-v2"
            document_embedding = embed_document(text, embedding_model_name)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare response
            return DocumentFinancialAnalysisResponse(
                document_id=document_id,
                financial_transaction=FinancialTransactionResponse(
                    document_type=financial_transaction.document_type,
                    amount=financial_transaction.amount,
                    currency=financial_transaction.currency,
                    date=financial_transaction.date.isoformat() if financial_transaction.date else None,
                    description=financial_transaction.description,
                    category=financial_transaction.category,
                    subcategory=financial_transaction.subcategory,
                    document_id=financial_transaction.document_id,
                    confidence=financial_transaction.confidence
                ),
                document_classification={
                    "rule_based_prediction": doc_type,
                    "model_prediction": model_result["prediction"],
                    "model_confidence": model_result["confidence"],
                    "final_prediction": final_doc_type,
                    "confidence_scores": model_result["confidence_scores"]
                },
                document_embedding=document_embedding,
                embedding_model=embedding_model_name,
                processing_time_ms=processing_time
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in financial analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

@app.post("/generate-bilan", response_model=FinancialBilanResponse)
async def generate_financial_bilan_groq(
    files: List[UploadFile] = File(...),
    period_days: int = Query(30, description="Number of days to include in the analysis")):
    """
    Generate a financial bilan using Groq AI for accurate data extraction
    
    Args:
        files: List of document files to analyze
        period_days: Number of days to include in the analysis (default: 30)
        
    Returns:
        Financial bilan with Groq-powered accurate data extraction
    """
    start_time = time.time()
    
    try:
        transactions = []
        processed_files = 0
        
        logger.info(f"Processing {len(files)} files for Groq-powered financial bilan")
        
        for file in files:
            try:
                # Generate a unique ID for this document
                document_id = str(uuid.uuid4())
                
                # Save the uploaded file temporarily
                temp_file_path = f"temp_{document_id}_{file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                try:
                    # Extract text and classify document
                    text = extract_text_from_file(temp_file_path)
                    doc_type = analyze_document(temp_file_path)
                    
                    # Improve classification
                    final_doc_type = doc_type.replace("❓ Unknown Document Type", "unknown")
                    if final_doc_type == "unknown":
                        model = get_cached_model()
                        model_result = model.predict(text)
                        
                        if hasattr(model, 'hybrid_predict'):
                            hybrid_result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                            final_doc_type = hybrid_result['final_prediction']
                        elif model_result["confidence"] > 0.2:
                            final_doc_type = model_result["prediction"]
                        else:
                            best_prediction = max(model_result["confidence_scores"].items(), key=lambda x: x[1])
                            if best_prediction[1] > 0.25:
                                final_doc_type = best_prediction[0]
                    
                    # Use Groq for financial analysis
                    groq_financial = analyze_document_with_groq(text, final_doc_type, document_id)
                    
                    # Convert to standard FinancialTransaction for compatibility
                    financial_transaction = FinancialTransaction(
                        document_type=groq_financial.document_type,
                        amount=groq_financial.amount,
                        currency=groq_financial.currency,
                        date=groq_financial.date,
                        description=groq_financial.description,
                        category=groq_financial.category,
                        subcategory=groq_financial.subcategory,
                        document_id=groq_financial.document_id,
                        confidence=groq_financial.confidence
                    )
                    
                    transactions.append(financial_transaction)
                    processed_files += 1
                    
                    logger.info(f"Processed {file.filename} with Groq: {final_doc_type}, Amount: {groq_financial.amount} {groq_financial.currency}")
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue  # Skip this file and continue with others
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No valid financial documents could be processed with Groq")
        
        # Generate the financial bilan
        report_generator = FinancialReportGenerator()
        bilan = report_generator.generate_bilan(transactions, period_days)
        
        logger.info(f"Generated Groq-powered bilan from {processed_files} documents in {(time.time() - start_time)*1000:.1f}ms")
        
        return FinancialBilanResponse(**bilan)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating Groq financial bilan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating Groq financial bilan: {str(e)}")
# need to delete after fix
@app.post("/test-generate-bilan", response_model=FinancialBilanResponse)
async def generate_financial_bilan(
    files: List[UploadFile] = File(...),
    period_days: int = Query(30, description="Number of days to include in the analysis")):
    """
    Generate a financial bilan (balance sheet) from multiple documents
    
    Args:
        files: List of document files to analyze
        period_days: Number of days to include in the analysis (default: 30)
        
    Returns:
        Financial bilan with summary, breakdown, and recommendations
    """
    start_time = time.time()
    
    try:
        transactions = []
        processed_files = 0
        
        logger.info(f"Processing {len(files)} files for financial bilan")
        
        for file in files:
            try:
                # Generate a unique ID for this document
                document_id = str(uuid.uuid4())
                
                # Save the uploaded file temporarily
                temp_file_path = f"temp_{document_id}_{file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                try:
                    # Extract text and classify document
                    text = extract_text_from_file(temp_file_path)
                    doc_type = analyze_document(temp_file_path)
                    
                    # Improve classification with model if needed - same logic as /classify
                    final_doc_type = doc_type.replace("❓ Unknown Document Type", "unknown")
                    if final_doc_type == "unknown":
                        model = get_cached_model()
                        model_result = model.predict(text)
                        
                        # Use hybrid_predict if available (same as /classify)
                        if hasattr(model, 'hybrid_predict'):
                            hybrid_result = model.hybrid_predict(text, document_id=document_id, text_excerpt=text[:500])
                            final_doc_type = hybrid_result['final_prediction']
                        elif model_result["confidence"] > 0.2:  # Lower threshold
                            final_doc_type = model_result["prediction"]
                        else:
                            # Find the highest confidence prediction
                            best_prediction = max(model_result["confidence_scores"].items(), key=lambda x: x[1])
                            if best_prediction[1] > 0.25:
                                final_doc_type = best_prediction[0]
                    
                    # Analyze financial information
                    financial_transaction = analyze_document_for_bilan(text, final_doc_type, document_id)
                    transactions.append(financial_transaction)
                    processed_files += 1
                    
                    logger.info(f"Processed {file.filename}: {final_doc_type}, Amount: {financial_transaction.amount} {financial_transaction.currency}")
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue  # Skip this file and continue with others
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No valid financial documents could be processed")
        
        # Generate the financial bilan
        report_generator = FinancialReportGenerator()
        bilan = report_generator.generate_bilan(transactions, period_days)
        
        logger.info(f"Generated bilan from {processed_files} documents in {(time.time() - start_time)*1000:.1f}ms")
        
        return FinancialBilanResponse(**bilan)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating financial bilan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating financial bilan: {str(e)}")

@app.get("/financial-summary")
async def get_financial_summary():
    """Get information about financial analysis capabilities"""
    return {
        "supported_document_types": [
            "invoice", "quote", "purchase_order", "receipt", 
            "bank_statement", "expense_report", "payslip", "delivery_note"
        ],
        "supported_currencies": ["EUR", "USD", "TND", "GBP"],
        "analysis_features": [
            "Amount extraction",
            "Currency detection", 
            "Date extraction",
            "Transaction categorization",
            "Financial recommendations",
            "Multi-currency support",
            "Period-based analysis"
        ],
        "bilan_metrics": [
            "Total income",
            "Total expenses", 
            "Net result",
            "Profit margin",
            "Currency breakdown",
            "Document type analysis"
        ]
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from the document
        text = extract_text_from_file(temp_file_path)
        
        # Analyze document type
        doc_type = analyze_document(temp_file_path)
        
        # Get model prediction
        model = get_cached_model()
        model_result = model.predict(text)
        
        # Extract information using Groq AI
        extracted_info = extract_document_info_with_groq(text, doc_type)
        
        # Generate document embedding
        embedding_model_name = "all-MiniLM-L6-v2"
        document_embedding = embed_document(text, embedding_model_name)
        
        # Upload to Cloudinary
        cloudinary_response = cloudinary.uploader.upload(
            temp_file_path,
            resource_type="raw",
            folder="documents"
        )
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Return response
        return {
            "document_id": cloudinary_response["public_id"],
            "model_prediction": model_result["prediction"],
            "model_confidence": model_result["confidence"],
            "rule_based_prediction": doc_type,
            "final_prediction": doc_type if doc_type != "❓ Unknown Document Type" else model_result["prediction"],
            "confidence_flag": "high" if model_result["confidence"] >= 0.7 else "low",
            "confidence_scores": model_result["confidence_scores"],
            "text_excerpt": text,
            "processing_time_ms": processing_time,
            "cloudinary_url": cloudinary_response["secure_url"],
            "public_id": cloudinary_response["public_id"],
            "extracted_info": extracted_info,
            "document_embedding": document_embedding,
            "embedding_model": embedding_model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            # Extract text from file first
            text = extract_text_from_file(file_path=temp_path)
            
            # Use our specialized document analyzer
            logger.info(f"Analyzing commercial document: {file.filename}")
            doc_type = analyze_document(temp_path)
            
            # Create analysis structure
            analysis = {
                'document_type': doc_type.replace("❓ Unknown Document Type", "unknown"),
                'confidence': 0.8 if doc_type != "❓ Unknown Document Type" else 0.1,
                'extracted_info': {},
                'type_scores': {doc_type: 0.8} if doc_type != "❓ Unknown Document Type" else {"unknown": 0.1},
                'text': text
            }
            
            # Get model prediction if requested
            if model_type:
                # Get the model
                logger.info(f"Getting model: {model_type}")
                model = get_cached_model(model_type)
                
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
            
            # Generate document embedding
            embedding_model_name = "all-MiniLM-L6-v2"
            document_embedding = embed_document(analysis['text'], embedding_model_name)
            
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
                processing_time_ms=processing_time_ms,
                document_embedding=document_embedding,
                embedding_model=embedding_model_name
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