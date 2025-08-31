"""
Document Classification API - FastAPI Application
Author: Bachir Fahmi
Email: bachir.fahmi@example.com
Description: FastAPI endpoints for document classification and financial analysis
"""

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
from datetime import datetime
import cloudinary
import cloudinary.uploader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Classification API",
    description="Production-ready document classification system using machine learning to automatically classify business documents with 95.5% accuracy. Developed by Bachir Fahmi.",
    version="1.0.0",
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
    details_transactions: Optional[List[Dict[str, Any]]] = []

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
        
        # Detailed console log showing ALL files received
        print("=" * 100)
        print(f"📄 GENERATE-BILAN API CALLED WITH {len(files)} FILES:")
        print("=" * 100)
        for i, file in enumerate(files, 1):
            print(f"🔸 FILE {i}: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'} bytes)")
        print(f"📅 PERIOD: {period_days} days")
        print("=" * 100)
        
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
                    
                    # Log extracted text for debugging
                    print(f"\n🔸 PROCESSING FILE: {file.filename}")
                    print(f"   Document type: {doc_type}")
                    print(f"   Extracted text preview: {text[:300]}...")
                    print(f"   Full text length: {len(text)} characters")
                    print("-" * 80)
                    
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
    start_time = time.time()
    
    try:
        # Get documents and business info from request
        documents = request.get("documents", [])
        business_info = request.get("business_info", {})
        period_days = request.get("period_days", 90)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Create unique session folder for this request
        session_id = str(uuid.uuid4())[:8]
        download_folder = f"downloaddoc/session_{session_id}"
        
        # Create session folder
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        
        print(f"📁 Created session folder: {download_folder}")
        
        extracted_documents = []
        downloaded_files = []
        
        logger.info(f"Processing {len(documents)} documents for bilan generation")
        
        # Download all documents for this session
        for doc in documents:
            try:
                # Download document from Cloudinary
                cloudinary_url = doc.get('cloudinaryUrl', '')
                if not cloudinary_url:
                    print(f"⚠️ No Cloudinary URL for {doc.get('filename', 'unknown')}")
                    continue
                
                # Download the file
                import requests
                response = requests.get(cloudinary_url)
                if response.status_code != 200:
                    print(f"⚠️ Failed to download {doc.get('filename', 'unknown')}")
                    continue
                
                # Save file in session folder
                saved_file_path = os.path.join(download_folder, f"{doc['id']}_{doc['filename']}")
                
                with open(saved_file_path, "wb") as f:
                    f.write(response.content)
                
                downloaded_files.append({
                    'file_path': saved_file_path,
                    'original_doc': doc
                })
                print(f"💾 Downloaded: {saved_file_path}")
                
            except Exception as e:
                print(f"❌ Error downloading {doc.get('filename', 'unknown')}: {str(e)}")
                continue
        
        if not downloaded_files:
            raise HTTPException(status_code=400, detail="No files could be downloaded")
        
        # Send files directly to Groq for bilan generation
        print("🔥 Sending files directly to Groq for bilan generation")
        logger.info("🔥 Sending files directly to Groq for bilan generation")
        
        bilan = await generate_bilan_from_files_directly(downloaded_files, business_info, period_days)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Cleanup: Remove session folder after processing
        try:
            import shutil
            shutil.rmtree(download_folder)
            print(f"🗑️ Cleaned up session folder: {download_folder}")
        except Exception as e:
            print(f"⚠️ Could not cleanup folder: {e}")
        
        return {
            "session_id": session_id,
            "processed_documents": len(downloaded_files),
            "processing_time_ms": processing_time,
            "business_info": business_info,
            **bilan  # Spread the bilan object directly into the response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bilan processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing bilan documents: {str(e)}")

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

# @app.post("/bilan")
# async def generate_tunisian_bilan(
#     request: Dict[str, Any]
# ):
#     """
#     Generate Tunisian accounting bilan (Bilan Comptable + Compte de Résultat) using Groq AI
    
#     Request Body:
#     {
#         "documents": [
#             {
#                 "id": "14edca9d-4f10-4afb-9e9d-7e11ce6d9544",
#                 "filename": "FACTURE_01-03-2025_NÂ°2220000032299099.pdf",
#                 "document_type": "invoice",
#                 "confidence": 0.959465650679308,
#                 "extracted_text": "Full OCR text...",
#                 "extracted_info": "{\"date\": \"2025-03-31\", \"client_name\": \"M. BACHIR FAHMI\"}",
#                 "created_at": "2025-07-27 03:22:38.85642"
#             }
#         ],
#         "period_days": 30
#     }
    
#     Returns: Complete Tunisian accounting bilan following Plan Comptable Tunisien
#     """
#     start_time = time.time()
    
#     try:
#         # Extract parameters
#         documents = request.get("documents", [])
#         period_days = request.get("period_days", 365)  # Default to annual bilan
#         document_only = request.get("document_only", False)  # New parameter for document-only mode
        
#         # Validate period_days - only quarterly or annual allowed
#         if period_days not in [90, 365]:
#             raise HTTPException(
#                 status_code=400, 
#                 detail="Invalid period_days. Only quarterly (90 days) or annual (365 days) bilans are supported."
#             )
        
#         if not documents:
#             raise HTTPException(status_code=400, detail="No documents provided")
        
#         # Check if documents have Cloudinary URLs (new format) or extracted_text (old format)
#         has_cloudinary_urls = any(doc.get('cloudinaryUrl') for doc in documents)
        
#         if has_cloudinary_urls:
#             # NEW FORMAT: Process Cloudinary URLs
#             print(f"📄 Bilan API called with {len(documents)} Cloudinary documents:")
#             for i, doc in enumerate(documents, 1):
#                 print(f"  {i}. {doc.get('document_type', 'unknown')} - {doc.get('filename', 'unknown')}")
            
#             logger.info(f"Processing {len(documents)} documents from Cloudinary URLs")
            
#             # Download documents and send actual files to Groq
#             downloaded_files = []
#             documents_data = []
            
#             # Create unique session folder for this request
#             import uuid
#             session_id = str(uuid.uuid4())[:8]
#             download_folder = f"downloaddoc/session_{session_id}"
            
#             try:
#                 # Create session folder
#                 if not os.path.exists(download_folder):
#                     os.makedirs(download_folder)
                
#                 print(f"📁 Created session folder: {download_folder}")
                
#                 # Download all documents for this session
#                 for doc in documents:
#                     try:
#                         # Download document from Cloudinary
#                         cloudinary_url = doc.get('cloudinaryUrl', '')
#                         if not cloudinary_url:
#                             print(f"⚠️ No Cloudinary URL for {doc.get('filename', 'unknown')}")
#                             continue
                        
#                         # Download the file
#                         import requests
#                         response = requests.get(cloudinary_url)
#                         if response.status_code != 200:
#                             print(f"⚠️ Failed to download {doc.get('filename', 'unknown')}")
#                             continue
                        
#                         # Save file in session folder
#                         saved_file_path = os.path.join(download_folder, f"{doc['id']}_{doc['filename']}")
                        
#                         with open(saved_file_path, "wb") as f:
#                             f.write(response.content)
                        
#                         downloaded_files.append(saved_file_path)
                        
#                         # Extract text for Groq processing
#                         extracted_text = extract_text_from_file(saved_file_path)
                        
#                         print(f"💾 Downloaded: {saved_file_path}")
                        
#                         if extracted_text.strip():
#                             documents_data.append({
#                                 'id': doc['id'],
#                                 'filename': doc['filename'],
#                                 'document_type': doc['document_type'],
#                                 'extracted_text': extracted_text,
#                                 'date': doc['created_at'],
#                                 'file_path': saved_file_path
#                             })
                            
#                             print(f"✅ Processed: {doc['filename']} ({len(extracted_text)} chars)")
#                         else:
#                             print(f"⚠️ No text extracted from: {doc['filename']}")
                            
#                     except Exception as e:
#                         print(f"❌ Error processing {doc.get('filename', 'unknown')}: {str(e)}")
#                         continue
                
#                 if not documents_data:
#                     raise HTTPException(status_code=400, detail="No text could be extracted from any documents")
                
#                 print(f"🚀 Sending {len(documents_data)} downloaded files to Groq for bilan generation")
                
#                 # Send downloaded files data to Groq for bilan generation
#                 bilan_result = await generate_bilan_from_downloaded_files(documents_data, period_days, session_id)
                
#             finally:
#                 # Clean up: Delete only files from this session
#                 print(f"🧹 Cleaning up session files from: {download_folder}")
#                 try:
#                     import shutil
#                     if os.path.exists(download_folder):
#                         shutil.rmtree(download_folder)
#                         print(f"✅ Deleted session folder: {download_folder}")
#                 except Exception as e:
#                     print(f"⚠️ Error cleaning up session folder: {e}")
            
#         else:
#             # OLD FORMAT: Process documents with extracted_text
#             print("=" * 100)
#             print(f"📄 BILAN API CALLED WITH {len(documents)} DOCUMENTS:")
#             print("=" * 100)
#             for i, doc in enumerate(documents, 1):
#                 print(f"\n🔸 DOCUMENT {i}:")
#                 print(f"   Type: {doc.get('document_type', 'unknown')}")
#                 print(f"   Filename: {doc.get('filename', 'unknown')}")
#                 print(f"   Date: {doc.get('created_at', 'unknown')}")
#                 print(f"   ID: {doc.get('id', 'unknown')}")
                
#                 # Show extracted text (first 300 chars)
#                 extracted_text = doc.get('extracted_text', '')
#                 if extracted_text:
#                     print(f"   Text preview: {extracted_text[:300]}...")
#                     print(f"   Full text length: {len(extracted_text)} characters")
#                 else:
#                     print("   Text preview: NO TEXT FOUND")
                
#                 # Show extracted info
#                 extracted_info = doc.get('extracted_info', '{}')
#                 print(f"   Extracted info: {extracted_info}")
#                 print("-" * 80)
            
#             print(f"\n📊 TOTAL DOCUMENTS: {len(documents)}")
#             print(f"📅 PERIOD: {period_days} days")
#             print(f"🔧 MODE: {'document-only' if document_only else 'full accounting'}")
#             print("=" * 100)
            
#             # Use existing logic for old format
#             mode = "document-only" if document_only else "full accounting"
#             logger.info(f"Generating Tunisian bilan ({mode}) from {len(documents)} documents for {period_days} days")
            
#             # Prepare all document texts for Groq analysis
#             documents_data = []
#             for doc in documents:
#                 # Clean extracted_text - handle nested JSON
#                 extracted_text = doc.get("extracted_text", "")
#                 cleaned_text = clean_extracted_text(extracted_text)
                
#                 documents_data.append({
#                     "id": doc["id"],
#                     "filename": doc.get("filename", "unknown"),
#                     "document_type": doc.get("document_type", "unknown"),
#                     "extracted_text": cleaned_text,
#                     "date": doc.get("created_at", ""),
#                     "extracted_info": doc.get("extracted_info", "{}")
#                 })
            
#             # Use Groq to generate bilan (document-only or full accounting mode)
#             if document_only:
#                 bilan_result = await generate_document_only_bilan(documents_data, period_days)
#             else:
#                 bilan_result = await generate_tunisian_bilan_with_groq(documents_data, period_days)
        
#         # FINAL VALIDATION: Ensure we never return default/fake data
#         if 'bilan_comptable' in bilan_result:
#             ca = bilan_result.get('compte_de_resultat', {}).get('chiffre_affaires', 0)
#             total_actif = bilan_result.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 0)
            
#             if ca == 0 and total_actif == 0:
#                 error_msg = f"Final validation failed: Bilan contains only zeros/default data. This means no real financial data was extracted from {len(documents)} documents."
#                 logger.error(error_msg)
#                 raise HTTPException(status_code=500, detail=error_msg)
        
#         # Add metadata only if we have real data
#         if "metadata" not in bilan_result:
#             bilan_result["metadata"] = {}
        
#         bilan_result["metadata"].update({
#             "documents_processed": len(documents),
#             "period_days": period_days,
#             "processing_time_ms": (time.time() - start_time) * 1000,
#             "generated_at": datetime.now().isoformat(),
#             "standard": "Plan Comptable Tunisien",
#             "source": "cloudinary_documents" if has_cloudinary_urls else "extracted_text"
#         })
        
#         logger.info(f"Generated Tunisian bilan with real data in {(time.time() - start_time)*1000:.1f}ms")
        
#         return bilan_result
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating Tunisian bilan: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error generating bilan: {str(e)}")

async def generate_document_only_bilan(documents_data: List[Dict], period_days: int) -> Dict[str, Any]:
    """Generate bilan using ONLY data found in documents - no artificial additions"""
    
    try:
        # Create document-only prompt
        prompt = create_document_only_bilan_prompt(documents_data, period_days)
        

        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a document data extraction expert. Extract ONLY the financial information explicitly stated in documents. Never add estimated, calculated, or assumed values."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        # Parse the response
        groq_response = response.choices[0].message.content
        logger.info(f"Groq document-only response received: {len(groq_response)} characters")
        

        
        # Parse JSON response
        bilan_data = parse_groq_bilan_response(groq_response)
        
        # Add document-only metadata
        if "metadata" not in bilan_data:
            bilan_data["metadata"] = {}
        bilan_data["metadata"]["extraction_mode"] = "document_only"
        bilan_data["metadata"]["artificial_data_added"] = False
        

        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error in document-only bilan generation: {str(e)}")
        raise

def parse_groq_response_simple(response: str) -> Dict[str, Any]:
    """Simple JSON parser - returns Groq's exact response without modifications"""
    import json
    import re
    
    try:
        # Remove markdown code blocks if present
        if '```' in response:
            # Extract JSON from markdown blocks
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL | re.IGNORECASE)
            if json_match:
                response = json_match.group(1)
        
        # Find JSON object in response
        if '{' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            # Try multiple approaches to parse the JSON
            try:
                # First try: direct parsing
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    # Second try: fix formatting and parse
                    # Fix commas in numbers and spacing issues
                    fixed_json = re.sub(r':\s*(\d{1,3}),(\d{3})\b', r': \1\2', json_str)
                    fixed_json = re.sub(r':\s*(\d{1,3}),(\d{3}),(\d{3})\b', r': \1\2\3', fixed_json)
                    fixed_json = re.sub(r'(\d+),"([a-zA-Z_])', r'\1, "\2', fixed_json)
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    # Third try: aggressive number fixing
                    import re
                    aggressive_fix = re.sub(r'(\d+),(\d+)', r'\1\2', json_str)
                    return json.loads(aggressive_fix)
        
        # If no JSON found, return error
        return {"error": "No JSON found in response"}
        
    except json.JSONDecodeError as e:
        logger.error(f"Simple JSON parsing failed: {e}")
        return {"error": "JSON parsing failed", "raw_response": response[:500]}

def extract_basic_data_from_response(groq_response: str) -> Dict[str, Any]:
    """Extract basic data from Groq response when JSON parsing fails"""
    import re
    
    logger.info("Attempting basic data extraction from Groq response")
    
    # Try to extract key values using regex
    result = {
        "bilan_comptable": {
            "actif": {"total_actif": 0},
            "passif": {"total_passif": 0}
        },
        "compte_de_resultat": {
            "resultat_net": 0,
            "chiffre_affaires": 0
        },
        "ratios_financiers": {},
        "analyse_financiere": {"points_forts": [], "points_faibles": [], "recommandations": []},
        "details_transactions": []
    }
    
    try:
        # Extract chiffre_affaires
        ca_match = re.search(r'"chiffre_affaires":\s*(\d+(?:[,\.]\d+)?)', groq_response)
        if ca_match:
            ca = float(ca_match.group(1).replace(',', ''))
            result["compte_de_resultat"]["chiffre_affaires"] = ca
            logger.info(f"Extracted revenue: {ca}")
        
        # Extract total_actif
        actif_match = re.search(r'"total_actif":\s*(\d+(?:[,\.]\d+)?)', groq_response)
        if actif_match:
            total_actif = float(actif_match.group(1).replace(',', ''))
            result["bilan_comptable"]["actif"]["total_actif"] = total_actif
            logger.info(f"Extracted total actif: {total_actif}")
        
        # Extract total_passif
        passif_match = re.search(r'"total_passif":\s*(\d+(?:[,\.]\d+)?)', groq_response)
        if passif_match:
            total_passif = float(passif_match.group(1).replace(',', ''))
            result["bilan_comptable"]["passif"]["total_passif"] = total_passif
            logger.info(f"Extracted total passif: {total_passif}")
        
        # Extract resultat_net
        resultat_match = re.search(r'"resultat_net":\s*(\d+(?:[,\.]\d+)?)', groq_response)
        if resultat_match:
            resultat_net = float(resultat_match.group(1).replace(',', ''))
            result["compte_de_resultat"]["resultat_net"] = resultat_net
            logger.info(f"Extracted resultat net: {resultat_net}")
        
        # Extract transaction details
        transaction_pattern = r'"montant":\s*(\d+(?:[,\.]\d+)?)[^}]*"libelle":\s*"([^"]+)"'
        transactions = []
        for i, match in enumerate(re.finditer(transaction_pattern, groq_response), 1):
            amount = float(match.group(1).replace(',', ''))
            description = match.group(2)
            transactions.append({
                "document_id": f"doc{i}",
                "type": "transaction",
                "montant": amount,
                "compte_comptable": "701",
                "libelle": description
            })
        
        if transactions:
            result["details_transactions"] = transactions
            logger.info(f"Extracted {len(transactions)} transactions")
        
        # Extract ratios
        ratios = {}
        ratio_patterns = [
            ('marge_brute_percent', r'"marge_brute_percent":\s*(\d+(?:\.\d+)?)'),
            ('marge_nette_percent', r'"marge_nette_percent":\s*(\d+(?:\.\d+)?)'),
            ('rentabilite_actif_percent', r'"rentabilite_actif_percent":\s*(\d+(?:\.\d+)?)'),
            ('liquidite_generale', r'"liquidite_generale":\s*(\d+(?:\.\d+)?)'),
            ('autonomie_financiere_percent', r'"autonomie_financiere_percent":\s*(\d+(?:\.\d+)?)')
        ]
        
        for ratio_name, pattern in ratio_patterns:
            match = re.search(pattern, groq_response)
            if match:
                ratios[ratio_name] = float(match.group(1))
        
        if ratios:
            result["ratios_financiers"] = ratios
            logger.info(f"Extracted {len(ratios)} financial ratios")
        
        # Extract analysis points
        points_forts = re.findall(r'"points_forts":\s*\[(.*?)\]', groq_response, re.DOTALL)
        if points_forts:
            # Extract individual points
            points = re.findall(r'"([^"]+)"', points_forts[0])
            result["analyse_financiere"]["points_forts"] = points[:3]  # Limit to 3
            logger.info(f"Extracted {len(points)} strengths")
        
        # Extract weaknesses
        points_faibles = re.findall(r'"points_faibles":\s*\[(.*?)\]', groq_response, re.DOTALL)
        if points_faibles:
            points = re.findall(r'"([^"]+)"', points_faibles[0])
            result["analyse_financiere"]["points_faibles"] = points[:3]
            logger.info(f"Extracted {len(points)} weaknesses")
        
        # Extract recommendations
        recommandations = re.findall(r'"recommandations":\s*\[(.*?)\]', groq_response, re.DOTALL)
        if recommandations:
            points = re.findall(r'"([^"]+)"', recommandations[0])
            result["analyse_financiere"]["recommandations"] = points[:3]
            logger.info(f"Extracted {len(points)} recommendations")
        
        return result
        
    except Exception as e:
        logger.error(f"Basic extraction failed: {e}")
        return result

def extract_data_from_groq_response(groq_response: str, documents_data: List[Dict]) -> Dict[str, Any]:
    """Extract data from Groq response when JSON parsing fails"""
    logger.info("Attempting manual data extraction from Groq response")
    return extract_json_manually(groq_response)

async def generate_bilan_from_files_directly(downloaded_files: List[Dict], business_info: Dict, period_days: int) -> Dict:
    """Send files directly to Groq for bilan generation - no text extraction"""
    try:
        from utils.groq_utils import client as groq_client
        import base64
        
        # Prepare files for Groq - handle both images and PDFs
        files_for_groq = []
        extracted_texts = []
        
        for file_info in downloaded_files:
            file_path = file_info['file_path']
            original_doc = file_info['original_doc']
            
            # Determine file type
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                # For PDFs, extract text since vision models may not handle PDFs well
                try:
                    from utils.text_extraction import extract_text_from_file
                    text_content = extract_text_from_file(file_path)
                    extracted_texts.append({
                        "filename": original_doc.get('filename', 'unknown'),
                        "document_type": original_doc.get('document_type', 'unknown'),
                        "text": text_content[:3000]  # Limit text length
                    })
                    print(f"📄 Extracted text from PDF: {original_doc.get('filename', 'unknown')}")
                except Exception as e:
                    print(f"⚠️ Could not extract text from PDF {file_path}: {e}")
                    continue
            
            elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                # For images, use vision API
                try:
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    
                    # Determine MIME type
                    if file_extension in ['jpg', 'jpeg']:
                        mime_type = "image/jpeg"
                    elif file_extension == 'png':
                        mime_type = "image/png"
                    elif file_extension == 'gif':
                        mime_type = "image/gif"
                    elif file_extension == 'webp':
                        mime_type = "image/webp"
                    
                    files_for_groq.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_content}"
                        }
                    })
                    print(f"🖼️ Added image to vision processing: {original_doc.get('filename', 'unknown')}")
                except Exception as e:
                    print(f"⚠️ Could not process image {file_path}: {e}")
                    continue
            else:
                print(f"⚠️ Unsupported file type: {file_extension} for {file_path}")
                continue
        
        # Check if we have any processable content
        if not extracted_texts and not files_for_groq:
            raise Exception("No processable documents found. Please ensure files are PDFs or images (JPG, PNG, GIF, WebP).")
        
        print(f"📊 Processing summary: {len(extracted_texts)} PDFs + {len(files_for_groq)} images = {len(extracted_texts) + len(files_for_groq)} total documents")
        
        # Create comprehensive prompt including extracted text
        text_section = ""
        if extracted_texts:
            text_section = "\n\nEXTRACTED TEXT FROM PDF DOCUMENTS:\n"
            for i, doc in enumerate(extracted_texts, 1):
                text_section += f"\nDocument {i}: {doc['filename']} (Type: {doc['document_type']})\n"
                text_section += f"Content: {doc['text']}\n"
                text_section += "-" * 80 + "\n"
        
        image_count = len(files_for_groq)
        total_docs = len(extracted_texts) + image_count
        
        # Create prompt for Groq to analyze files and generate bilan
        prompt = f"""Analyze these {total_docs} financial documents ({len(extracted_texts)} PDFs with extracted text + {image_count} images) and generate a complete Tunisian accounting bilan following the Plan Comptable Tunisien.

{text_section}

Business Information:
- Company: {business_info.get('name', 'N/A')}
- Period: {business_info.get('period_start', 'N/A')} to {business_info.get('period_end', 'N/A')}
- Analysis Period: Last {period_days} days

Generate EXACTLY this JSON structure:
{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": 0,
                "immobilisations_incorporelles": 0,
                "immobilisations_financieres": 0,
                "total_actif_non_courant": 0
            }},
            "actif_courant": {{
                "stocks_et_en_cours": 0,
                "clients_et_comptes_rattaches": 0,
                "autres_creances": 0,
                "disponibilites": 0,
                "total_actif_courant": 0
            }},
            "total_actif": 0
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": 0,
                "reserves": 0,
                "resultat_net_exercice": 0,
                "total_capitaux_propres": 0
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": 0,
                "provisions_lt": 0,
                "total_passif_non_courant": 0
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": 0,
                "dettes_fiscales_et_sociales": 0,
                "autres_dettes_ct": 0,
                "total_passif_courant": 0
            }},
            "total_passif": 0
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": 0,
            "production_immobilisee": 0,
            "subventions_exploitation": 0,
            "autres_produits_exploitation": 0,
            "total_produits_exploitation": 0
        }},
        "charges_exploitation": {{
            "achats_consommes": 0,
            "charges_personnel": 0,
            "dotations_amortissements": 0,
            "autres_charges_exploitation": 0,
            "total_charges_exploitation": 0
        }},
        "resultat_exploitation": 0,
        "resultat_financier": 0,
        "resultat_exceptionnel": 0,
        "resultat_avant_impot": 0,
        "impot_sur_benefices": 0,
        "resultat_net": 0
    }},
    "ratios_financiers": {{
        "marge_brute_percent": 0.0,
        "marge_nette_percent": 0.0,
        "rentabilite_actif_percent": 0.0,
        "liquidite_generale": 0.0,
        "autonomie_financiere_percent": 0.0
    }},
    "analyse_financiere": {{
        "points_forts": [],
        "points_faibles": [],
        "recommandations": []
    }},
    "details_transactions": []
}}

INSTRUCTIONS:
1. Read and analyze each document image
2. Extract financial data and classify according to Tunisian accounting standards
3. Calculate totals and ratios
4. Return ONLY valid JSON, no additional text
"""

        # Prepare messages for Groq API
        if files_for_groq:
            # Mixed content: text + images
            messages = [
                {
                    "role": "system",
                    "content": "You are a Tunisian certified accountant. Analyze document text and images to generate accurate bilans following Plan Comptable Tunisien."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *files_for_groq
                    ]
                }
            ]
        else:
            # Text only (PDFs)
            messages = [
                {
                    "role": "system",
                    "content": "You are a Tunisian certified accountant. Analyze document text to generate accurate bilans following Plan Comptable Tunisien."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        
        # Call Groq Vision API
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to supported model
            messages=messages,
            temperature=0.1,
            max_tokens=4000
        )
        
        groq_response = response.choices[0].message.content
        logger.info(f"Groq direct file response: {groq_response[:500]}...")
        print(f"🔥 Groq direct file response: {groq_response[:500]}...")
        
        # Parse response
        bilan = parse_bilan_response(groq_response)
        
        return bilan
        
    except Exception as e:
        logger.error(f"Error in direct file Groq bilan generation: {str(e)}")
        raise Exception(f"Failed to generate bilan from files: {str(e)}")

async def generate_bilan_from_documents(documents: List[Dict], business_info: Dict, period_days: int) -> Dict:
    """Generate a financial bilan (balance sheet) from extracted document texts using Groq AI"""
    try:
        # Prepare the context for Groq AI
        documents_context = ""
        for doc in documents:
            doc_type = doc.get('document_type', 'unknown')
            filename = doc.get('filename', 'unknown')
            text = doc.get('extracted_text', '')
            documents_context += f"\n--- {doc_type.upper()}: {filename} ---\n{text}\n"
        
        # Create the prompt for generating bilan according to Tunisian Accounting Plan
        prompt = f"""You are a Tunisian certified accountant expert. Based on the following business documents, generate a complete financial analysis following the Plan Comptable Tunisien (Tunisian Accounting Plan).

Business Information:
- Company: {business_info.get('name', 'N/A')}
- Period: {business_info.get('period_start', 'N/A')} to {business_info.get('period_end', 'N/A')}
- Analysis Period: Last {period_days} days

Documents to analyze:
{documents_context}

Generate a complete financial report with EXACTLY this JSON structure (in French, following Tunisian standards):

{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": 0,
                "immobilisations_incorporelles": 0,
                "immobilisations_financieres": 0,
                "total_actif_non_courant": 0
            }},
            "actif_courant": {{
                "stocks_et_en_cours": 0,
                "clients_et_comptes_rattaches": 0,
                "autres_creances": 0,
                "disponibilites": 0,
                "total_actif_courant": 0
            }},
            "total_actif": 0
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": 0,
                "reserves": 0,
                "resultat_net_exercice": 0,
                "total_capitaux_propres": 0
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": 0,
                "provisions_lt": 0,
                "total_passif_non_courant": 0
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": 0,
                "dettes_fiscales_et_sociales": 0,
                "autres_dettes_ct": 0,
                "total_passif_courant": 0
            }},
            "total_passif": 0
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": 0,
            "production_immobilisee": 0,
            "subventions_exploitation": 0,
            "autres_produits_exploitation": 0,
            "total_produits_exploitation": 0
        }},
        "charges_exploitation": {{
            "achats_consommes": 0,
            "charges_personnel": 0,
            "dotations_amortissements": 0,
            "autres_charges_exploitation": 0,
            "total_charges_exploitation": 0
        }},
        "resultat_exploitation": 0,
        "resultat_financier": 0,
        "resultat_exceptionnel": 0,
        "resultat_avant_impot": 0,
        "impot_sur_benefices": 0,
        "resultat_net": 0
    }},
    "ratios_financiers": {{
        "marge_brute_percent": 0.0,
        "marge_nette_percent": 0.0,
        "rentabilite_actif_percent": 0.0,
        "liquidite_generale": 0.0,
        "autonomie_financiere_percent": 0.0
    }},
    "analyse_financiere": {{
        "points_forts": [],
        "points_faibles": [],
        "recommandations": []
    }},
    "details_transactions": [{{
        "document_id": "DOC_ID",
        "type": "document_type",
        "montant": 0,
        "compte_comptable": "XXX - Libellé compte",
        "libelle": "Description transaction"
    }}]
}}

IMPORTANT INSTRUCTIONS:
1. Extract ALL financial data from the documents and classify according to Tunisian chart of accounts
2. For each document, create a transaction entry in "details_transactions"
3. Calculate ALL totals and ensure bilan balance (total_actif = total_passif)
4. Compute financial ratios accurately
5. Provide meaningful financial analysis in French
6. Return ONLY valid JSON, no additional text

Analyze each document type:
- Invoices → Clients et comptes rattachés (411) + Chiffre d'affaires (70X)
- Purchases → Fournisseurs (401) + Achats (60X)
- Bank statements → Disponibilités (512)
- Payslips → Charges personnel (64X) + Dettes sociales (43X)
- Receipts → Various expense accounts (60X-65X)
"""

        # Call Groq AI to generate the bilan
        bilan_response = await call_groq_for_bilan_analysis(prompt)
        
        # Log Groq response for debugging
        logger.info(f"Groq bilan response: {bilan_response[:500]}...")
        print(f"🔥 Groq bilan response preview: {bilan_response[:500]}...")
        
        # Parse and structure the response
        bilan = parse_bilan_response(bilan_response)
        
        return bilan
        
    except Exception as e:
        logger.error(f"Error generating bilan: {str(e)}")
        # Return a basic structure if AI analysis fails
        return {
            "bilan_comptable": {"actif": {"actif_non_courant": {"immobilisations_corporelles": 0,"immobilisations_incorporelles": 0,"immobilisations_financieres": 0,"total_actif_non_courant": 0},"actif_courant": {"stocks_et_en_cours": 0,"clients_et_comptes_rattaches": 0,"autres_creances": 0,"disponibilites": 0,"total_actif_courant": 0},"total_actif": 0},"passif": {"capitaux_propres": {"capital_social": 0,"reserves": 0,"resultat_net_exercice": 0,"total_capitaux_propres": 0},"passif_non_courant": {"emprunts_dettes_financieres_lt": 0,"provisions_lt": 0,"total_passif_non_courant": 0},"passif_courant": {"fournisseurs_et_comptes_rattaches": 0,"dettes_fiscales_et_sociales": 0,"autres_dettes_ct": 0,"total_passif_courant": 0},"total_passif": 0}},
            "compte_de_resultat": {"produits_exploitation": {"chiffre_affaires": 0,"production_immobilisee": 0,"subventions_exploitation": 0,"autres_produits_exploitation": 0,"total_produits_exploitation": 0},"charges_exploitation": {"achats_consommes": 0,"charges_personnel": 0,"dotations_amortissements": 0,"autres_charges_exploitation": 0,"total_charges_exploitation": 0},"resultat_exploitation": 0,"resultat_financier": 0,"resultat_exceptionnel": 0,"resultat_avant_impot": 0,"impot_sur_benefices": 0,"resultat_net": 0},
            "ratios_financiers": {"marge_brute_percent": 0.0,"marge_nette_percent": 0.0,"rentabilite_actif_percent": 0.0,"liquidite_generale": 0.0,"autonomie_financiere_percent": 0.0},
            "analyse_financiere": {"points_forts": [],"points_faibles": [],"recommandations": []},
            "details_transactions": [],
            "error": f"Could not generate complete bilan: {str(e)}"
        }

async def call_groq_for_bilan_analysis(prompt: str) -> str:
    """Call Groq AI API to analyze documents and generate bilan"""
    try:
        from utils.groq_utils import client as groq_client
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional accountant and financial analyst expert in creating balance sheets from business documents."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise Exception(f"Failed to analyze documents with AI: {str(e)}")

def parse_bilan_response(response: str) -> Dict:
    """Parse the AI response and structure it into Tunisian accounting format"""
    try:
        import json
        
        logger.info(f"Parsing bilan response: {response[:200]}...")
        print(f"🔥 Parsing bilan response: {response[:200]}...")
        
        # Try to extract JSON from the response
        original_response = response
        
        if "Here is the JSON format:" in response:
            json_start = response.find("Here is the JSON format:") + len("Here is the JSON format:")
            response = response[json_start:].strip()
            logger.info("Found 'Here is the JSON format:' marker")
        elif "{" in response:
            json_start = response.find("{")
            response = response[json_start:]
            logger.info("Found JSON starting with '{'")
        
        # Remove markdown formatting
        if response.startswith('```json'):
            response = response[7:]
            logger.info("Removed ```json marker")
        if response.endswith('```'):
            response = response[:-3]
            logger.info("Removed ``` marker")
        response = response.strip()
        
        logger.info(f"Cleaned JSON for parsing: {response[:300]}...")
        print(f"🔥 Cleaned JSON: {response[:300]}...")
        
        try:
            bilan_data = json.loads(response)
            logger.info("Successfully parsed JSON from Groq response")
            print("🔥 Successfully parsed JSON from Groq response")
            
            # Add metadata section if not present
            if "metadata" not in bilan_data:
                bilan_data["metadata"] = {
                    "documents_processed": len(bilan_data.get("details_transactions", [])),
                    "period_days": 90,
                    "processing_time_ms": 0,
                    "generated_at": datetime.now().isoformat(),
                    "standard": "Plan Comptable Tunisien"
                }
            
            return bilan_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Failed to parse cleaned response: {response}")
            print(f"🔥 JSON decode error: {str(e)}")
            print(f"🔥 Failed response: {response[:500]}...")
        
        # If JSON parsing fails, return the Tunisian standard structure
        return {
            "bilan_comptable": {"actif": {"actif_non_courant": {"immobilisations_corporelles": 0,"immobilisations_incorporelles": 0,"immobilisations_financieres": 0,"total_actif_non_courant": 0},"actif_courant": {"stocks_et_en_cours": 0,"clients_et_comptes_rattaches": 0,"autres_creances": 0,"disponibilites": 0,"total_actif_courant": 0},"total_actif": 0},"passif": {"capitaux_propres": {"capital_social": 0,"reserves": 0,"resultat_net_exercice": 0,"total_capitaux_propres": 0},"passif_non_courant": {"emprunts_dettes_financieres_lt": 0,"provisions_lt": 0,"total_passif_non_courant": 0},"passif_courant": {"fournisseurs_et_comptes_rattaches": 0,"dettes_fiscales_et_sociales": 0,"autres_dettes_ct": 0,"total_passif_courant": 0},"total_passif": 0}},
            "compte_de_resultat": {"produits_exploitation": {"chiffre_affaires": 0,"production_immobilisee": 0,"subventions_exploitation": 0,"autres_produits_exploitation": 0,"total_produits_exploitation": 0},"charges_exploitation": {"achats_consommes": 0,"charges_personnel": 0,"dotations_amortissements": 0,"autres_charges_exploitation": 0,"total_charges_exploitation": 0},"resultat_exploitation": 0,"resultat_financier": 0,"resultat_exceptionnel": 0,"resultat_avant_impot": 0,"impot_sur_benefices": 0,"resultat_net": 0},
            "ratios_financiers": {"marge_brute_percent": 0.0,"marge_nette_percent": 0.0,"rentabilite_actif_percent": 0.0,"liquidite_generale": 0.0,"autonomie_financiere_percent": 0.0},
            "analyse_financiere": {"points_forts": ["Aucune analyse disponible - données insuffisantes"],"points_faibles": ["Données non disponibles pour l'analyse"],"recommandations": ["Fournir plus de documents comptables pour une analyse complète"]},
            "details_transactions": [],
            "metadata": {"documents_processed": 0,"period_days": 90,"processing_time_ms": 0,"generated_at": datetime.now().isoformat(),"standard": "Plan Comptable Tunisien"},
            "parsing_note": "Structure par défaut - analyse IA non disponible"
        }
        
    except Exception as e:
        logger.error(f"Error parsing bilan response: {str(e)}")
        raise Exception(f"Failed to parse AI response: {str(e)}")

async def generate_bilan_from_files_with_groq(downloaded_files: List[Dict], period_days: int) -> Dict[str, Any]:
    """
    Send files directly to Groq for bilan generation - let Groq handle everything
    """
    try:
        import json
        from utils.groq_utils import client as groq_client
        
        # Extract text from all files and send to Groq
        all_texts = ""
        for file_info in downloaded_files:
            file_path = file_info['file_path']
            original_doc = file_info['original_doc']
            
            # Extract text using existing function (which uses Groq for images)
            text = extract_text_from_file(file_path)
            all_texts += f"\n--- {original_doc['filename']} ({original_doc.get('document_type', 'unknown')}) ---\n"
            all_texts += text + "\n" + "="*80 + "\n"
        
        # Create prompt for Groq to analyze and generate bilan
        prompt = f"""
Analyze these financial documents and generate a complete Tunisian accounting bilan following the Plan Comptable Tunisien.

DOCUMENTS CONTENT:
{all_texts}

REQUIREMENTS:
1. Convert ALL amounts to Tunisian Dinar (TND): EUR=3.3 TND, USD=3.1 TND
2. Generate complete bilan following Plan Comptable Tunisien

Return ONLY this JSON format:
{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": 0,
                "immobilisations_incorporelles": 0,
                "immobilisations_financieres": 0,
                "total_actif_non_courant": 0
            }},
            "actif_courant": {{
                "stocks_et_en_cours": 0,
                "clients_et_comptes_rattaches": 0,
                "autres_creances": 0,
                "disponibilites": 0,
                "total_actif_courant": 0
            }},
            "total_actif": 0
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": 0,
                "reserves": 0,
                "resultat_net_exercice": 0,
                "total_capitaux_propres": 0
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": 0,
                "provisions_lt": 0,
                "total_passif_non_courant": 0
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": 0,
                "dettes_fiscales_et_sociales": 0,
                "autres_dettes_ct": 0,
                "total_passif_courant": 0
            }},
            "total_passif": 0
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": 0,
            "production_immobilisee": 0,
            "subventions_exploitation": 0,
            "autres_produits_exploitation": 0,
            "total_produits_exploitation": 0
        }},
        "charges_exploitation": {{
            "achats_consommes": 0,
            "charges_personnel": 0,
            "dotations_amortissements": 0,
            "autres_charges_exploitation": 0,
            "total_charges_exploitation": 0
        }},
        "resultat_exploitation": 0,
        "resultat_financier": 0,
        "resultat_exceptionnel": 0,
        "resultat_avant_impot": 0,
        "impot_sur_benefices": 0,
        "resultat_net": 0
    }},
    "ratios_financiers": {{
        "marge_brute_percent": 0,
        "marge_nette_percent": 0,
        "rentabilite_actif_percent": 0,
        "liquidite_generale": 0,
        "autonomie_financiere_percent": 0
    }},
    "analyse_financiere": {{
        "points_forts": [],
        "points_faibles": [],
        "recommandations": []
    }},
    "details_transactions": [],
    "metadata": {{
        "documents_processed": {len(downloaded_files)},
        "period_days": {period_days},
        "generated_at": "{datetime.now().isoformat()}",
        "standard": "Plan Comptable Tunisien"
    }}
}}
"""

        # Call Groq API
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Tunisian accounting expert. Analyze documents and generate accurate bilans following Plan Comptable Tunisien."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Parse response
        groq_response = response.choices[0].message.content
        logger.info(f"Raw Groq response: {groq_response}")
        print(f"🔥 Raw Groq response: {groq_response}")
        
        if not groq_response or not groq_response.strip():
            raise HTTPException(status_code=500, detail="Groq returned empty response")
        
        # Clean and parse JSON - extract JSON from Groq's response
        cleaned_response = groq_response.strip()
        
        # Look for JSON in the response
        if "Here is the JSON format:" in cleaned_response:
            # Extract everything after "Here is the JSON format:"
            json_start = cleaned_response.find("Here is the JSON format:") + len("Here is the JSON format:")
            cleaned_response = cleaned_response[json_start:].strip()
        elif "{" in cleaned_response:
            # Find the first { and extract from there
            json_start = cleaned_response.find("{")
            cleaned_response = cleaned_response[json_start:]
        
        # Remove markdown formatting
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        logger.info(f"Cleaned response: {cleaned_response}")
        print(f"🔥 Cleaned response: {cleaned_response}")
        
        if not cleaned_response:
            raise HTTPException(status_code=500, detail="Groq response is empty after cleaning")
        
        try:
            bilan_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Failed to parse: {cleaned_response}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON from Groq: {str(e)}")
        
        # Check for error
        if "error" in bilan_data:
            raise HTTPException(status_code=400, detail=f"Groq analysis error: {bilan_data['error']}")
        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error in direct Groq bilan generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating bilan: {str(e)}")

async def generate_tunisian_bilan_with_groq(documents_data: List[Dict], period_days: int) -> Dict[str, Any]:
    """Generate Tunisian accounting bilan using Groq AI"""
    
    try:
        # Create comprehensive prompt for Tunisian bilan generation
        prompt = create_tunisian_bilan_prompt(documents_data, period_days)
        

        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Tunisian certified accountant expert in Plan Comptable Tunisien. Generate professional accounting reports following Tunisian accounting standards."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent accounting
            max_tokens=4000
        )
        
        # Parse the response
        groq_response = response.choices[0].message.content
        logger.info(f"Groq bilan response received: {len(groq_response)} characters")
        

        
        # Log the FULL raw response for debugging
        logger.info(f"FULL Groq response: {groq_response}")
        
        # Parse and return Groq's response
        logger.info("Parsing Groq response")
        
        # Try simple JSON parsing first
        try:
            import json
            import re
            
            # Remove markdown code blocks if present
            clean_response = groq_response
            if '```' in clean_response:
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', clean_response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    clean_response = json_match.group(1)
            
            # Fix number formatting issues (commas in numbers)
            clean_response = re.sub(r'(\d+),(\d+)', r'\1\2', clean_response)
            
            # Try direct JSON parsing
            bilan_data = json.loads(clean_response)
            logger.info("✅ Simple JSON parsing successful")
            

            
        except Exception as e:
            logger.info(f"Simple JSON parsing failed: {e}, trying complex parser")
            bilan_data = parse_groq_bilan_response(groq_response)
        

        
        # If parsing fails, raise an error instead of returning fake data
        if "error" in bilan_data:
            error_msg = f"Failed to parse Groq response: {bilan_data.get('error', 'Unknown parsing error')}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        

        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error in Groq bilan generation: {str(e)}")
        raise

def create_working_bilan_prompt(documents_data: List[Dict], period_days: int) -> str:
    """Create a working prompt that generates real financial analysis"""
    
    # Prepare documents summary
    documents_summary = ""
    for i, doc in enumerate(documents_data[:5], 1):  # Limit to 5 docs for shorter prompt
        documents_summary += f"Doc{i}: {doc['document_type']} - {doc['extracted_text'][:200]}...\n"
    
    prompt = f"""You are a Tunisian accountant. Analyze these business documents and create a complete bilan comptable using ONLY the data found in the documents.

DOCUMENTS TO ANALYZE:
{documents_summary}

CRITICAL RULES:
1. Extract ONLY real amounts from the documents
2. Convert currencies: EUR×3.3, USD×3.1 to TND
3. Do NOT add any equipment, assets, or values not mentioned in documents
4. Create realistic accounting entries based on what you find
5. If no assets are mentioned, set immobilisations_corporelles to 0

Return complete JSON structure:
{{
"bilan_comptable": {{
"actif": {{
"actif_non_courant": {{
"immobilisations_corporelles": 0,
"immobilisations_incorporelles": 0,
"immobilisations_financieres": 0,
"total_actif_non_courant": 0
}},
"actif_courant": {{
"stocks_et_en_cours": 0,
"clients_et_comptes_rattaches": <amount_based_on_invoices>,
"autres_creances": 0,
"disponibilites": <amount_based_on_bank_statement>,
"total_actif_courant": <sum>
}},
"total_actif": <total>
}},
"passif": {{
"capitaux_propres": {{
"capital_social": <realistic_amount>,
"reserves": 0,
"resultat_net_exercice": <calculated_profit>,
"total_capitaux_propres": <sum>
}},
"passif_non_courant": {{
"emprunts_dettes_financieres_lt": 0,
"provisions_lt": 0,
"total_passif_non_courant": 0
}},
"passif_courant": {{
"fournisseurs_et_comptes_rattaches": <based_on_expenses>,
"dettes_fiscales_et_sociales": <based_on_invoices>,
"autres_dettes_ct": 0,
"total_passif_courant": <sum>
}},
"total_passif": <must_equal_total_actif>
}}
}},
"compte_de_resultat": {{
"produits_exploitation": {{
"chiffre_affaires": <sum_of_all_invoices_converted_to_TND>,
"production_immobilisee": 0,
"subventions_exploitation": 0,
"autres_produits_exploitation": 0,
"total_produits_exploitation": <total_revenue>
}},
"charges_exploitation": {{
"achats_consommes": <based_on_bank_statement_expenses>,
"charges_personnel": <salary_payments_from_bank>,
"dotations_amortissements": 0,
"autres_charges_exploitation": <other_bank_expenses>,
"total_charges_exploitation": <sum>
}},
"resultat_exploitation": <revenue_minus_charges>,
"resultat_financier": 0,
"resultat_exceptionnel": 0,
"resultat_avant_impot": <operating_result>,
"impot_sur_benefices": 0,
"resultat_net": <final_result>
}},
"ratios_financiers": {{
"marge_brute_percent": <calculated>,
"marge_nette_percent": <calculated>,
"rentabilite_actif_percent": <calculated>,
"liquidite_generale": <calculated>,
"autonomie_financiere_percent": <calculated>
}},
"analyse_financiere": {{
"points_forts": ["<real_analysis_based_on_data>"],
"points_faibles": ["<real_weaknesses>"],
"recommandations": ["<actionable_advice>"]
}},
"details_transactions": [
<create_entry_for_each_document_with_real_amounts>
]
}}

Extract ONLY what exists in the documents. Return ONLY the JSON."""
    
    return prompt

def create_document_only_bilan_prompt(documents_data: List[Dict], period_days: int) -> str:
    """Create prompt for document-only bilan generation - no artificial data"""
    
    # Prepare documents summary
    documents_summary = ""
    for i, doc in enumerate(documents_data[:8], 1):  # Limit to 8 docs to reduce tokens
        # Send more text but not full content to avoid token limits
        text_content = doc['extracted_text']
        # Send first 800 characters which should capture most amounts
        content_preview = text_content[:800] if len(text_content) > 800 else text_content
        
        documents_summary += f"""
Document {i}:
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Date: {doc['date']}
- EXTENDED CONTENT: {content_preview}
- Full text length: {len(text_content)} characters
- Info: {doc['extracted_info']}
"""
    
    prompt = f"""
You are a financial document analyst. Extract ONLY the financial information that is explicitly stated in these {len(documents_data)} documents. DO NOT add any estimated, calculated, or assumed values.

CRITICAL RULES:
1. Extract ONLY information explicitly written in the documents
2. Do NOT add any estimated assets, liabilities, or balance sheet items
3. Do NOT calculate or assume any values not stated in documents
4. Do NOT create artificial accounting entries to balance the books
5. If information is missing, leave it as 0 or omit it entirely
6. Focus on actual transactions and amounts found in documents

Documents to analyze:
{documents_summary}

Generate a financial report in JSON format with ONLY the data found in documents:

{{
    "details_transactions": [
        {{
            "document_id": "doc_id",
            "type": "document_type",
            "montant": <amount_if_found>,
            "compte_comptable": "account_code_if_applicable",
            "libelle": "description_from_document",
            "date": "date_if_found",
            "currency": "currency_if_found"
        }}
    ],
    "financial_summary": {{
        "total_revenue_found": <sum_of_revenue_documents>,
        "total_expenses_found": <sum_of_expense_documents>,
        "total_tax_found": <sum_of_tax_amounts>,
        "currencies_found": ["TND", "EUR", "USD"],
        "document_types_processed": ["invoice", "receipt", "etc"]
    }},
    "document_only_bilan": {{
        "note": "This bilan contains ONLY data explicitly found in documents",
        "actif": {{
            "amounts_receivable": <only_if_explicitly_mentioned>,
            "cash_mentioned": <only_if_bank_statements_provided>,
            "assets_mentioned": <only_if_asset_documents_provided>,
            "total_actif_documented": <sum_of_documented_assets>
        }},
        "passif": {{
            "debts_mentioned": <only_if_debt_documents_provided>,
            "capital_mentioned": <only_if_capital_documents_provided>,
            "total_passif_documented": <sum_of_documented_liabilities>
        }},
        "balance_status": "unbalanced_insufficient_data" or "balanced_from_documents"
    }},
    "missing_for_complete_bilan": [
        "bank_statements_for_cash_position",
        "asset_records_for_equipment_property",
        "debt_documents_for_liabilities",
        "capital_investment_records",
        "etc"
    ],
    "recommendations": [
        "Provide bank statements to show actual cash position",
        "Include asset purchase records for equipment/property",
        "Add loan/debt documents for complete liability picture"
    ]
}}

IMPORTANT: 
- Only include amounts that are explicitly stated in documents
- Do not estimate or calculate missing values
- Do not force the accounting equation to balance
- Be honest about missing information
- Return ONLY the JSON structure
"""
    
    return prompt

def create_tunisian_bilan_prompt(documents_data: List[Dict], period_days: int) -> str:
    """Create comprehensive prompt for Tunisian bilan generation"""
    
    # Prepare documents summary
    documents_summary = ""
    for i, doc in enumerate(documents_data[:8], 1):  # Limit to 8 docs to reduce tokens
        # Send more text but not full content to avoid token limits
        text_content = doc['extracted_text']
        # Send first 800 characters which should capture most amounts
        content_preview = text_content[:800] if len(text_content) > 800 else text_content
        
        documents_summary += f"""
Document {i}:
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Date: {doc['date']}
- EXTENDED CONTENT: {content_preview}
- Full text length: {len(text_content)} characters
- Info: {doc['extracted_info']}
"""
    
    # Determine bilan type
    bilan_type = "trimestriel" if period_days == 90 else "annuel"
    period_label = "trimestre" if period_days == 90 else "année"
    
    prompt = f"""
You are a Tunisian certified accountant. Analyze these {len(documents_data)} business documents from various sources and generate a complete accounting bilan {bilan_type} following the Plan Comptable Tunisien standards.

IMPORTANT CONTEXT:
- Documents may be in multiple languages (French, English, Arabic, Tunisian)
- Documents may have different currencies (TND, EUR, USD, etc.) - convert all to TND
- Documents may have various date formats and ranges - focus on the reporting period
- Some extracted_text may contain nested JSON - extract the actual text content
- Documents come from international clients and various companies with different formats

Type de bilan: Bilan {bilan_type} ({period_label})
Période d'analyse: {period_days} jours

Documents to analyze:
{documents_summary}

🔍 CRITICAL: EXTRACT REAL AMOUNTS FROM DOCUMENT TEXT
The "extracted_info" shows amount:0 for all documents - IGNORE THIS!
Instead, read the actual text content and find the real amounts:
- Ooredoo invoices: Look for "25,000" or similar amounts in TND
- Bank statements: Extract transaction amounts like "231,109", "764,081" 
- Foreign invoices: Find amounts like "$51.94" or "174.00 €" and convert to TND
- Purchase orders: Extract order values and convert currencies

Generate a comprehensive Tunisian accounting report in JSON format with the following structure:

{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": <amount_in_TND>,
                "immobilisations_incorporelles": <amount_in_TND>,
                "immobilisations_financieres": <amount_in_TND>,
                "total_actif_non_courant": <total_amount>
            }},
            "actif_courant": {{
                "stocks_et_en_cours": <amount_in_TND>,
                "clients_et_comptes_rattaches": <amount_in_TND>,
                "autres_creances": <amount_in_TND>,
                "disponibilites": <amount_in_TND>,
                "total_actif_courant": <total_amount>
            }},
            "total_actif": <total_amount>
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": <amount_in_TND>,
                "reserves": <amount_in_TND>,
                "resultat_net_exercice": <amount_in_TND>,
                "total_capitaux_propres": <total_amount>
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": <amount_in_TND>,
                "provisions_lt": <amount_in_TND>,
                "total_passif_non_courant": <total_amount>
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": <amount_in_TND>,
                "dettes_fiscales_et_sociales": <amount_in_TND>,
                "autres_dettes_ct": <amount_in_TND>,
                "total_passif_courant": <total_amount>
            }},
            "total_passif": <total_amount>
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": <amount_in_TND>,
            "production_immobilisee": <amount_in_TND>,
            "subventions_exploitation": <amount_in_TND>,
            "autres_produits_exploitation": <amount_in_TND>,
            "total_produits_exploitation": <total_amount>
        }},
        "charges_exploitation": {{
            "achats_consommes": <amount_in_TND>,
            "charges_personnel": <amount_in_TND>,
            "dotations_amortissements": <amount_in_TND>,
            "autres_charges_exploitation": <amount_in_TND>,
            "total_charges_exploitation": <total_amount>
        }},
        "resultat_exploitation": <amount_in_TND>,
        "resultat_financier": <amount_in_TND>,
        "resultat_exceptionnel": <amount_in_TND>,
        "resultat_avant_impot": <amount_in_TND>,
        "impot_sur_benefices": <amount_in_TND>,
        "resultat_net": <amount_in_TND>
    }},
    "ratios_financiers": {{
        "marge_brute_percent": <percentage>,
        "marge_nette_percent": <percentage>,
        "rentabilite_actif_percent": <percentage>,
        "liquidite_generale": <ratio>,
        "autonomie_financiere_percent": <percentage>
    }},
    "analyse_financiere": {{
        "points_forts": [
            "Analyze the actual financial data and provide specific strengths based on the documents",
            "Example: 'Chiffre d'affaires stable de X TND' or 'Marge bénéficiaire positive'"
        ],
        "points_faibles": [
            "Identify actual weaknesses from the financial data",
            "Example: 'Charges élevées représentant X% du CA' or 'Liquidités insuffisantes'"
        ],
        "recommandations": [
            "Provide specific actionable recommendations based on the analysis",
            "Example: 'Optimiser les coûts d'exploitation' or 'Améliorer la gestion de trésorerie'"
        ]
    }},
    "details_transactions": [
        {{
            "document_id": "doc_id",
            "type": "document_type",
            "montant": <amount>,
            "compte_comptable": "account_number",
            "libelle": "description"
        }}
    ]
}}

CRITICAL ACCOUNTING REQUIREMENTS:
1. ACCOUNTING EQUATION: Actif total MUST EQUAL Passif total (fundamental accounting principle)
2. REALISTIC BALANCE SHEET: If there's revenue, there must be corresponding assets (cash, receivables, or inventory)
3. LOGICAL BUSINESS STRUCTURE: A company with revenue needs capital, assets, and realistic financial position

FINANCIAL ANALYSIS REQUIREMENTS:
1. POINTS FORTS: Analyze actual numbers and identify specific strengths (e.g., "Chiffre d'affaires de X TND montre une activité commerciale solide", "Marge brute de Y% indique une bonne rentabilité")
2. POINTS FAIBLES: Identify real weaknesses from the data (e.g., "Charges d'exploitation élevées à Z% du CA", "Manque de liquidités avec seulement W TND disponibles")
3. RECOMMANDATIONS: Provide actionable business advice based on the financial situation (e.g., "Réduire les coûts opérationnels", "Améliorer le recouvrement des créances clients")

CRITICAL DATA EXTRACTION INSTRUCTIONS:
⚠️ IGNORE the "extracted_info" field - it contains incorrect amounts (all showing 0)
⚠️ READ THE ACTUAL DOCUMENT TEXT and extract amounts directly from the text content
⚠️ Look for patterns like: "25,000", "Total: 174.00 €", "Balance Due: $51.94", "Débit (DT)", etc.

Instructions:
1. EXTRACT FINANCIAL DATA FROM TEXT: Read the actual document text content and find all monetary amounts
   - Ooredoo invoices: Look for "Total facture du mois", "Montant total", amounts in TND
   - Bank statements: Extract all "Crédit (DT)" and "Débit (DT)" amounts
   - Foreign invoices: Convert EUR/USD to TND (USD≈3.1, EUR≈3.3, TND=1.0)
   - Purchase orders: Extract order amounts and convert currencies
2. CURRENCY CONVERSION: Convert to TND (USD≈3.1, EUR≈3.3, TND=1.0)
3. CALCULATE TOTALS: Sum all revenue and expenses from the ACTUAL TEXT CONTENT
4. ANALYZE PERFORMANCE: Calculate actual ratios and provide specific insights based on real numbers
5. NO PLACEHOLDERS: Never use generic terms like "Strength1" or "Recommendation1" - always provide specific, data-driven analysis
4. CREATE REALISTIC BILAN COMPTABLE:
   
   ACTIF (Assets) - Must reflect business operations:
   - If total revenue > 50,000 TND → Add immobilisations_corporelles (equipment/office)
   - Always include clients_et_comptes_rattaches = 60-80% of total revenue (unpaid invoices)
   - Always include disponibilites = 20-40% of total revenue (cash from paid invoices)
   - Add realistic stocks if applicable
   
   PASSIF (Liabilities + Equity) - Must balance with Actif:
   - capital_social = 60-80% of total actif (owner investment)
   - resultat_net_exercice = net profit from compte de résultat
   - Add fournisseurs_et_comptes_rattaches = 10-20% of expenses (unpaid suppliers)
   - Add dettes_fiscales_et_sociales = 15-19% of revenue (taxes owed)
   
5. ENSURE BALANCE: Total Actif = Total Passif (verify this calculation)
6. COMPTE DE RESULTAT: Calculate realistic P&L with proper categories
7. FINANCIAL RATIOS: Calculate based on the balanced bilan
8. PROFESSIONAL ANALYSIS: Provide meaningful insights
9. JSON ONLY: Return only the JSON structure

EXAMPLE REALISTIC STRUCTURE for 25,735 TND revenue:
```
Actif:
- clients_et_comptes_rattaches: 18,000 TND (70% of revenue - unpaid invoices)
- disponibilites: 8,000 TND (30% of revenue - cash received)
- immobilisations_corporelles: 5,000 TND (office equipment)
Total Actif: 31,000 TND

Passif:
- capital_social: 25,000 TND (owner investment)
- resultat_net_exercice: 161 TND (net profit)
- dettes_fiscales_et_sociales: 3,900 TND (15% tax on revenue)
- autres_dettes_ct: 1,939 TND (other liabilities)
Total Passif: 31,000 TND ✓ (BALANCED)
```

MANDATORY: Verify that Total Actif = Total Passif before returning the JSON.

CRITICAL: For analyse_financiere section, provide SPECIFIC insights based on actual data:
- Replace "Strength1" with actual strengths like "Chiffre d'affaires de 25,735 TND indique une activité commerciale solide"
- Replace "Weakness1" with real issues like "Charges d'exploitation élevées à 85% du CA réduisent la rentabilité"
- Replace "Recommendation1" with actionable advice like "Négocier de meilleurs tarifs fournisseurs pour améliorer la marge"

Never use placeholder text like "Strength1", "Weakness2", "Recommendation1" - always analyze the real financial data.
"""
    
    return prompt

def extract_data_from_groq_response_DISABLED(response: str, documents_data: List[Dict]) -> Dict[str, Any]:
    """DISABLED - This function was overriding Groq's response"""
    try:
        import re
        
        # Extract revenue from documents
        total_revenue = 0
        transactions = []
        
        for i, doc in enumerate(documents_data, 1):
            # Try to extract amounts from document info
            doc_info = str(doc.get('extracted_info', ''))
            doc_text = str(doc.get('extracted_text', ''))
            
            # Look for amounts in various formats
            amounts = re.findall(r'[\d,]+\.?\d*', doc_info + ' ' + doc_text)
            if amounts:
                try:
                    # Take the largest amount as the main amount
                    main_amount = max([float(a.replace(',', '')) for a in amounts if float(a.replace(',', '')) > 10])
                    
                    # Currency conversion (basic detection)
                    currency_multiplier = 1.0
                    if 'EUR' in doc_info.upper() or 'EURO' in doc_info.upper():
                        currency_multiplier = 3.3
                    elif 'USD' in doc_info.upper() or 'DOLLAR' in doc_info.upper():
                        currency_multiplier = 3.1
                    
                    converted_amount = main_amount * currency_multiplier
                    total_revenue += converted_amount
                    
                    transactions.append({
                        "document_id": f"doc{i}",
                        "type": doc.get('document_type', 'invoice'),
                        "montant": round(converted_amount, 2),
                        "compte_comptable": "701",
                        "libelle": f"Revenue from {doc.get('filename', 'document')}"
                    })
                    
                except (ValueError, TypeError):
                    continue
        
        # Generate realistic bilan based on extracted revenue
        if total_revenue > 0:
            # Calculate realistic structure
            equipment = 0  # No equipment purchases found in documents
            receivables = int(total_revenue * 0.65)
            cash = int(total_revenue * 0.25)
            total_actif = equipment + receivables + cash
            
            capital = int(total_actif * 0.75)
            profit = int(total_revenue * 0.08)
            tax_debt = int(total_revenue * 0.15)
            other_debts = total_actif - capital - profit - tax_debt
            
            # Calculate ratios
            margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            roa = (profit / total_actif * 100) if total_actif > 0 else 0
            liquidity = ((receivables + cash) / (tax_debt + other_debts)) if (tax_debt + other_debts) > 0 else 0
            autonomy = ((capital + profit) / total_actif * 100) if total_actif > 0 else 0
            
            return {
                "bilan_comptable": {
                    "actif": {
                        "actif_non_courant": {
                            "immobilisations_corporelles": equipment,
                            "immobilisations_incorporelles": 0,
                            "immobilisations_financieres": 0,
                            "total_actif_non_courant": equipment
                        },
                        "actif_courant": {
                            "stocks_et_en_cours": 0,
                            "clients_et_comptes_rattaches": receivables,
                            "autres_creances": 0,
                            "disponibilites": cash,
                            "total_actif_courant": receivables + cash
                        },
                        "total_actif": total_actif
                    },
                    "passif": {
                        "capitaux_propres": {
                            "capital_social": capital,
                            "reserves": 0,
                            "resultat_net_exercice": profit,
                            "total_capitaux_propres": capital + profit
                        },
                        "passif_non_courant": {
                            "emprunts_dettes_financieres_lt": 0,
                            "provisions_lt": 0,
                            "total_passif_non_courant": 0
                        },
                        "passif_courant": {
                            "fournisseurs_et_comptes_rattaches": max(0, other_debts),
                            "dettes_fiscales_et_sociales": tax_debt,
                            "autres_dettes_ct": 0,
                            "total_passif_courant": tax_debt + max(0, other_debts)
                        },
                        "total_passif": total_actif
                    }
                },
                "compte_de_resultat": {
                    "produits_exploitation": {
                        "chiffre_affaires": int(total_revenue),
                        "production_immobilisee": 0,
                        "subventions_exploitation": 0,
                        "autres_produits_exploitation": 0,
                        "total_produits_exploitation": int(total_revenue)
                    },
                    "charges_exploitation": {
                        "achats_consommes": int(total_revenue * 0.6),
                        "charges_personnel": int(total_revenue * 0.2),
                        "dotations_amortissements": int(total_revenue * 0.05),
                        "autres_charges_exploitation": int(total_revenue * 0.07),
                        "total_charges_exploitation": int(total_revenue * 0.92)
                    },
                    "resultat_exploitation": profit,
                    "resultat_financier": 0,
                    "resultat_exceptionnel": 0,
                    "resultat_avant_impot": profit,
                    "impot_sur_benefices": 0,
                    "resultat_net": profit
                },
                "ratios_financiers": {
                    "marge_brute_percent": round(margin, 1),
                    "marge_nette_percent": round(margin, 1),
                    "rentabilite_actif_percent": round(roa, 1),
                    "liquidite_generale": round(liquidity, 1),
                    "autonomie_financiere_percent": round(autonomy, 1)
                },
                "analyse_financiere": {
                    "points_forts": [
                        f"Chiffre d'affaires de {int(total_revenue):,} TND généré",
                        f"Marge nette positive de {round(margin, 1)}%",
                        f"Bonne autonomie financière ({round(autonomy, 1)}%)"
                    ],
                    "points_faibles": [
                        "Dépendance aux créances clients",
                        "Besoin d'améliorer la trésorerie",
                        "Diversification des revenus à développer"
                    ],
                    "recommandations": [
                        "Accélérer le recouvrement des créances",
                        "Constituer une réserve de trésorerie",
                        "Développer de nouveaux produits/services"
                    ]
                },
                "details_transactions": transactions
            }
        
        # If no revenue found, return minimal structure
        return {
            "error": "No financial data extracted from documents",
            "bilan_comptable": {"actif": {"total_actif": 0}, "passif": {"total_passif": 0}},
            "compte_de_resultat": {"resultat_net": 0},
            "ratios_financiers": {},
            "analyse_financiere": {"points_forts": [], "points_faibles": [], "recommandations": []},
            "details_transactions": []
        }
        
    except Exception as e:
        logger.error(f"Error in manual data extraction: {str(e)}")
        return {
            "error": f"Manual extraction failed: {str(e)}",
            "bilan_comptable": {"actif": {"total_actif": 0}, "passif": {"total_passif": 0}},
            "compte_de_resultat": {"resultat_net": 0},
            "ratios_financiers": {},
            "analyse_financiere": {"points_forts": [], "points_faibles": [], "recommandations": []},
            "details_transactions": []
        }

def validate_and_fix_bilan(bilan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Just validate the bilan - let Groq do all the work"""
    try:
        # Just log what Groq generated - no automatic fixing
        bilan = bilan_data.get("bilan_comptable", {})
        actif = bilan.get("actif", {})
        passif = bilan.get("passif", {})
        
        total_actif = actif.get("total_actif", 0)
        total_passif = passif.get("total_passif", 0)
        
        logger.info(f"Groq generated bilan: Actif={total_actif}, Passif={total_passif}")
        
        # Return exactly what Groq generated - no modifications
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error validating bilan: {str(e)}")
        return bilan_data

async def generate_bilan_from_downloaded_files(documents_data: List[Dict], period_days: int, session_id: str) -> Dict[str, Any]:
    """Send downloaded files information to Groq for complete bilan generation with real amounts"""
    try:
        # Prepare documents summary with file information
        documents_summary = ""
        for i, doc in enumerate(documents_data[:12], 1):  # Limit to 12 docs for more content per doc
            # Use first 1500 characters to capture amount in letters and more context
            content_preview = doc['extracted_text'][:1500] if len(doc['extracted_text']) > 1500 else doc['extracted_text']
            
            documents_summary += f"""
=== DOCUMENT {i} ===
ID: {doc['id']}
TYPE: {doc['document_type']}
FILENAME: {doc['filename']}
DATE: {doc['date']}

FULL CONTENT TO ANALYZE:
{content_preview}

ANALYSIS REQUIRED:
- Find all monetary amounts (numbers)
- Find amount in letters/words (French/Arabic)
- Determine if 25,000 means 25 or 25,000 dinars
- Apply currency conversion if needed
- Verify reasonableness for document type

"""
        
        # Create prompt that simulates sending actual files to Groq
        files_info = ""
        for i, doc in enumerate(documents_data[:15], 1):
            files_info += f"""
FILE {i}: {doc['filename']}
- Document Type: {doc['document_type']}
- File Path: {doc['file_path']}
- File Size: {len(doc['extracted_text'])} characters
- Document ID: {doc['id']}
- Date: {doc['date']}
"""
        
        prompt = f"""
You are an expert financial analyst. I'm sending you {len(documents_data)} business documents. 

Please:
1. Read each document and understand what type it is (invoice, receipt, bank statement, etc.)
2. Extract the main financial amount from each document (the amount the customer actually needs to pay or received)
3. Convert any foreign currencies to Tunisian Dinars (TND): USD×3.1, EUR×3.3
4. Generate a complete Tunisian accounting bilan using the real amounts you found

SESSION: {session_id} | PERIOD: {period_days} days

DOCUMENTS TO ANALYZE:
{files_info}

RETURN ONLY ONE COMPLETE JSON OBJECT:

{{
    "bilan_comptable": {{
        "actif": {{
            "actif_non_courant": {{
                "immobilisations_corporelles": [REALISTIC_AMOUNT],
                "immobilisations_incorporelles": 0,
                "immobilisations_financieres": 0,
                "total_actif_non_courant": [TOTAL]
            }},
            "actif_courant": {{
                "stocks_et_en_cours": 0,
                "clients_et_comptes_rattaches": [TOTAL_REVENUE_FROM_DOCUMENTS],
                "autres_creances": 0,
                "disponibilites": [CASH_AMOUNT],
                "total_actif_courant": [TOTAL]
            }},
            "total_actif": [TOTAL_ASSETS]
        }},
        "passif": {{
            "capitaux_propres": {{
                "capital_social": [REALISTIC_CAPITAL],
                "reserves": 0,
                "resultat_net_exercice": [NET_RESULT],
                "total_capitaux_propres": [TOTAL]
            }},
            "passif_non_courant": {{
                "emprunts_dettes_financieres_lt": 0,
                "provisions_lt": 0,
                "total_passif_non_courant": 0
            }},
            "passif_courant": {{
                "fournisseurs_et_comptes_rattaches": [REALISTIC_AMOUNT],
                "dettes_fiscales_et_sociales": [TAX_AMOUNT],
                "autres_dettes_ct": 0,
                "total_passif_courant": [TOTAL]
            }},
            "total_passif": [MUST_EQUAL_TOTAL_ACTIF]
        }}
    }},
    "compte_de_resultat": {{
        "produits_exploitation": {{
            "chiffre_affaires": [TOTAL_REVENUE_FROM_ALL_DOCUMENTS],
            "production_immobilisee": 0,
            "subventions_exploitation": 0,
            "autres_produits_exploitation": 0,
            "total_produits_exploitation": [TOTAL_REVENUE]
        }},
        "charges_exploitation": {{
            "achats_consommes": 0,
            "charges_personnel": 0,
            "dotations_amortissements": 0,
            "autres_charges_exploitation": 0,
            "total_charges_exploitation": 0
        }},
        "resultat_exploitation": [REVENUE_MINUS_CHARGES],
        "resultat_financier": 0,
        "resultat_exceptionnel": 0,
        "resultat_avant_impot": [OPERATING_RESULT],
        "impot_sur_benefices": 0,
        "resultat_net": [FINAL_RESULT]
    }},
    "ratios_financiers": {{
        "marge_brute_percent": [CALCULATED],
        "marge_nette_percent": [CALCULATED],
        "rentabilite_actif_percent": [CALCULATED],
        "liquidite_generale": [CALCULATED],
        "autonomie_financiere_percent": [CALCULATED]
    }},
    "analyse_financiere": {{
        "points_forts": ["Chiffre d'affaires de [REAL_AMOUNT] TND basé sur documents réels"],
        "points_faibles": ["Point faible basé sur analyse réelle"],
        "recommandations": ["Recommandation basée sur données réelles"]
    }},
    "details_transactions": [
        {{
            "document_id": "[DOC_ID]",
            "type": "[DOCUMENT_TYPE]",
            "montant": [REAL_AMOUNT],
            "compte_comptable": "[ACCOUNT_CODE]",
            "libelle": "[DESCRIPTION]",
            "date": "[DATE]"
        }}
    ]
}}

Just use your intelligence to understand each document and extract the correct amounts. 

Return this JSON structure with the real amounts you found:

[JSON structure remains the same]

Important:
- Use the actual amounts from the documents
- Make sure total_actif equals total_passif (balanced accounting)
- Include one transaction per document in details_transactions
- Return only the JSON, no explanations
"""
        
        print(f"📤 Sending session {session_id} data to Groq for COMPLETE BILAN GENERATION")
        print("🏦 FULL MODE: Generating complete bilan with real amounts")
        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent financial document analyst. Read and understand any business document in any language. Extract the correct financial amounts based on what the document actually says. Use your natural understanding - don't overthink it."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=6000  # Increased for complete JSON response
        )
        
        # Parse Groq's response
        groq_response = response.choices[0].message.content
        print(f"✅ Groq processed session {session_id} and returned {len(groq_response)} characters")
        
        # Parse JSON directly (bypass complex validation that rejects zeros)
        import json
        import re
        
        try:
            # Clean the response
            clean_response = groq_response.strip()
            
            # Remove markdown code blocks if present
            if '```' in clean_response:
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', clean_response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    clean_response = json_match.group(1)
            
            # Handle case where Groq returns multiple JSON objects - take the last/complete one
            if clean_response.count('{') > 1:
                # Find the last complete JSON object (usually the updated one with real data)
                json_objects = []
                brace_count = 0
                start_pos = -1
                
                for i, char in enumerate(clean_response):
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos >= 0:
                            json_objects.append(clean_response[start_pos:i+1])
                
                if json_objects:
                    # Find the JSON object with actual data (non-zero amounts)
                    best_json = None
                    for json_obj in json_objects:
                        try:
                            test_parse = json.loads(json_obj)
                            # Check if this JSON has real data (non-zero chiffre_affaires)
                            ca = test_parse.get('compte_de_resultat', {}).get('chiffre_affaires', 0)
                            total_actif = test_parse.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 0)
                            
                            if ca > 0 or total_actif > 0:
                                best_json = json_obj
                                print(f"🎯 Found JSON with real data: CA={ca}, Actif={total_actif}")
                                break
                        except:
                            continue
                    
                    # Use the JSON with real data, or fall back to the last one
                    clean_response = best_json if best_json else json_objects[-1]
                    print(f"🔧 Selected JSON object from {len(json_objects)} objects for session {session_id}")
            
            # Fix common JSON issues - handle number formatting carefully
            # Only remove commas that are clearly thousands separators (not decimal separators)
            # Pattern: digit(s), comma, exactly 3 digits (thousands separator)
            clean_response = re.sub(r'(\d+),(\d{3})', r'\1\2', clean_response)  # 25,000 → 25000
            # Don't touch other commas (like in arrays or decimal numbers)
            
            # Try to fix incomplete JSON by finding the last complete object
            if not clean_response.endswith('}'):
                # Find the last complete closing brace
                brace_count = 0
                last_valid_pos = -1
                for i, char in enumerate(clean_response):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid_pos = i + 1
                            break
                
                if last_valid_pos > 0:
                    clean_response = clean_response[:last_valid_pos]
                    print(f"🔧 Fixed incomplete JSON for session {session_id}")
            
            bilan_data = json.loads(clean_response)
            print(f"✅ Direct JSON parsing successful for session {session_id}")
            
            # Validate that we have real bilan data (not zeros)
            ca = bilan_data.get('compte_de_resultat', {}).get('produits_exploitation', {}).get('chiffre_affaires', 0)
            total_actif = bilan_data.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 0)
            
            if ca == 0 and total_actif == 0:
                print(f"⚠️ WARNING: Parsed JSON has all zeros for session {session_id}")
                print("This means Groq returned template data instead of real amounts")
                print("Full response for debugging:")
                print(groq_response)
                raise HTTPException(
                    status_code=500, 
                    detail=f"Groq returned template data with zeros instead of real amounts from documents. Session: {session_id}"
                )
            
            # Show what Groq actually returned
            print("=" * 60)
            print(f"GROQ RESPONSE FOR SESSION {session_id}:")
            print("=" * 60)
            print(f"� Totalh Actif: {total_actif} TND")
            print(f"💰 Chiffre d'affaires: {ca} TND")
            
            if 'compte_de_resultat' in bilan_data:
                cr = bilan_data.get('compte_de_resultat', {})
                charges = cr.get('charges_exploitation', 0)
                resultat = cr.get('resultat_net', 0)
                print(f"💸 Charges: {charges} TND")
                print(f"📈 Résultat net: {resultat} TND")
            
            if 'analyse_financiere' in bilan_data:
                analyse = bilan_data.get('analyse_financiere', {})
                points_forts = analyse.get('points_forts', [])
                print(f"✅ Points forts: {points_forts}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Direct JSON parsing failed for session {session_id}: {e}")
            print("Full Groq response:")
            print(groq_response)
            print("-" * 50)
            
            raise HTTPException(status_code=500, detail=f"Failed to parse Groq response for session {session_id}: {str(e)}")
        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error processing downloaded files for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

async def generate_bilan_from_cloudinary_urls(documents: List[Dict], period_days: int) -> Dict[str, Any]:
    """Send Cloudinary URLs directly to Groq - let Groq handle everything"""
    try:
        # Prepare document URLs for Groq
        documents_summary = ""
        for i, doc in enumerate(documents[:20], 1):  # Limit to 20 docs to avoid token limits
            documents_summary += f"""
Document {i}:
- ID: {doc['id']}
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Cloudinary URL: {doc['cloudinaryUrl']}
- Date: {doc['created_at']}
"""
        
        # Create prompt for Groq to handle everything
        prompt = f"""
You are a Tunisian certified accountant. I will provide you with Cloudinary URLs of business documents. 

IMPORTANT: You need to:
1. Access each Cloudinary URL to download and read the document content
2. Extract all financial amounts from each document
3. Generate a complete Tunisian accounting bilan following Plan Comptable Tunisien standards

Documents to process:
{documents_summary}

Period: {period_days} days ({"trimestriel" if period_days == 90 else "annuel"})

For each document URL:
1. Download the document from the Cloudinary URL
2. Extract the text content (OCR if needed)
3. Find all monetary amounts (convert to TND: USD×3.1, EUR×3.3)
4. Classify as revenue or expense

Then generate a complete JSON bilan with:
- bilan_comptable (balanced actif/passif)
- compte_de_resultat (real revenue/expenses)
- ratios_financiers
- analyse_financiere (specific insights based on real data)
- details_transactions

CRITICAL: Use ONLY the actual amounts found in the documents. No artificial or estimated values.

Return complete JSON bilan structure following Plan Comptable Tunisien.
"""
        
        print(f"🚀 Sending {len(documents)} Cloudinary URLs to Groq for complete processing")
        
        # Call Groq API - let Groq handle everything
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Tunisian certified accountant with access to download and process documents from URLs. You can access Cloudinary URLs to download documents and extract financial data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        # Parse Groq's complete response
        groq_response = response.choices[0].message.content
        print(f"✅ Groq processed all documents and returned {len(groq_response)} characters")
        
        # Show what Groq actually returned for debugging
        print("=" * 80)
        print("GROQ RESPONSE DEBUG:")
        print("=" * 80)
        print(f"Response length: {len(groq_response)} characters")
        print("First 500 characters:")
        print(groq_response[:500])
        print("=" * 80)
        
        # Parse the JSON response
        bilan_data = parse_groq_bilan_response(groq_response)
        
        if "error" in bilan_data:
            # If parsing fails, try simple JSON parsing
            import json
            import re
            
            try:
                if '```' in groq_response:
                    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', groq_response, re.DOTALL | re.IGNORECASE)
                    if json_match:
                        groq_response = json_match.group(1)
                
                bilan_data = json.loads(groq_response)
                print("✅ Simple JSON parsing successful")
            except Exception as e:
                print(f"❌ JSON parsing failed: {e}")
                print("Full Groq response:")
                print(groq_response)
                raise HTTPException(status_code=500, detail=f"Failed to parse Groq response: {str(e)}")
        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error in Groq Cloudinary processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing Cloudinary documents with Groq: {str(e)}")

async def extract_financial_data_batch(documents_data: List[Dict]) -> List[Dict]:
    """Extract financial data from a batch of documents using Groq"""
    try:
        # Create prompt for financial data extraction
        documents_summary = ""
        for i, doc in enumerate(documents_data, 1):
            # Use first 1000 characters to capture amounts while staying under token limit
            content_preview = doc['extracted_text'][:1000] if len(doc['extracted_text']) > 1000 else doc['extracted_text']
            documents_summary += f"""
Document {i}:
- ID: {doc['id']}
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Content: {content_preview}
"""
        
        prompt = f"""
Extract ONLY the financial amounts from these {len(documents_data)} documents. Return a JSON array with the financial data found.

Documents:
{documents_summary}

For each document, extract:
- Document ID
- Document type  
- All monetary amounts found (convert to TND: USD×3.1, EUR×3.3)
- Transaction type (revenue/expense)
- Description

Return JSON format:
[
  {{
    "document_id": "doc_id",
    "document_type": "invoice",
    "amounts": [
      {{"amount": 25000, "currency": "TND", "type": "revenue", "description": "Facture Ooredoo"}}
    ]
  }}
]

CRITICAL: Extract ALL amounts you see in the text. Look for patterns like "25,000", "$51.94", "174.00 €", etc.
"""
        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data extraction expert. Extract only the monetary amounts explicitly stated in documents."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        # Parse response
        groq_response = response.choices[0].message.content
        
        # Extract JSON from response
        import json
        import re
        
        if '```' in groq_response:
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', groq_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                groq_response = json_match.group(1)
        
        # Parse JSON
        financial_data = json.loads(groq_response)
        
        print(f"✅ Extracted financial data from {len(documents_data)} documents")
        return financial_data
        
    except Exception as e:
        logger.error(f"Error extracting financial data from batch: {str(e)}")
        return []

async def generate_bilan_from_financial_data(financial_data: List[Dict], period_days: int) -> Dict[str, Any]:
    """Generate final bilan from extracted financial data"""
    try:
        # Summarize all financial data
        total_revenue = 0
        total_expenses = 0
        transactions_summary = []
        
        for doc_data in financial_data:
            for amount_data in doc_data.get('amounts', []):
                amount = amount_data.get('amount', 0)
                if amount_data.get('type') == 'revenue':
                    total_revenue += amount
                else:
                    total_expenses += amount
                
                transactions_summary.append({
                    "document_id": doc_data.get('document_id'),
                    "type": doc_data.get('document_type'),
                    "amount": amount,
                    "description": amount_data.get('description', '')
                })
        
        # Create summary for Groq
        summary = f"""
FINANCIAL SUMMARY FROM {len(financial_data)} DOCUMENTS:
- Total Revenue Found: {total_revenue} TND
- Total Expenses Found: {total_expenses} TND
- Net Result: {total_revenue - total_expenses} TND

TRANSACTIONS:
{chr(10).join([f"- {t['type']}: {t['amount']} TND ({t['description']})" for t in transactions_summary[:10]])}
"""
        
        # Generate bilan using Groq
        prompt = f"""
Based on this REAL financial data extracted from documents, generate a complete Tunisian accounting bilan:

{summary}

Generate a balanced bilan comptable following Plan Comptable Tunisien standards. Use the ACTUAL amounts provided above.

Return JSON format with complete bilan structure including:
- bilan_comptable (actif/passif balanced)
- compte_de_resultat (using real revenue/expenses)
- ratios_financiers
- analyse_financiere (based on real data)
- details_transactions

CRITICAL: Use the REAL amounts provided. Total revenue = {total_revenue} TND, Total expenses = {total_expenses} TND.
"""
        
        # Call Groq for final bilan
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Tunisian certified accountant. Generate professional bilans using real financial data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=3000
        )
        
        # Parse final bilan
        groq_response = response.choices[0].message.content
        bilan_data = parse_groq_bilan_response(groq_response)
        
        if "error" in bilan_data:
            error_msg = f"Failed to generate bilan from financial data: {bilan_data.get('error', 'Unknown error')}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        return bilan_data
        
    except Exception as e:
        logger.error(f"Error generating final bilan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating final bilan: {str(e)}")



def clean_extracted_text(extracted_text: str) -> str:
    """Clean extracted_text that might contain nested JSON"""
    try:
        # Check if it's JSON containing a "text" field
        if extracted_text.strip().startswith('{') and '"text"' in extracted_text:
            import json
            parsed = json.loads(extracted_text)
            if isinstance(parsed, dict) and "text" in parsed:
                return parsed["text"]
        
        # Return as-is if not nested JSON
        return extracted_text
        
    except (json.JSONDecodeError, KeyError):
        # If parsing fails, return original text
        return extracted_text

def extract_json_manually(json_str: str) -> Dict[str, Any]:
    """Manually extract data when JSON parsing completely fails"""
    import re
    
    # Try to extract key numbers from the response
    ca = 0  # No default - extract from actual documents
    ca_match = re.search(r'"chiffre_affaires":\s*(\d+(?:[\.,]\d+)?)', json_str)
    if ca_match:
        ca = float(ca_match.group(1).replace(',', ''))
    
    # Create complete structure like your example
    result = {
        "bilan_comptable": {
            "actif": {
                "actif_non_courant": {
                    "immobilisations_corporelles": 0,
                    "immobilisations_incorporelles": 0,
                    "immobilisations_financieres": 0,
                    "total_actif_non_courant": 0
                },
                "actif_courant": {
                    "stocks_et_en_cours": 0,
                    "clients_et_comptes_rattaches": int(ca * 0.7),  # 70% of revenue
                    "autres_creances": 0,
                    "disponibilites": int(ca * 0.3),  # 30% of revenue
                    "total_actif_courant": int(ca * 1.0)
                },
                "total_actif": int(ca * 1.2)
            },
            "passif": {
                "capitaux_propres": {
                    "capital_social": int(ca * 0.9),
                    "reserves": 0,
                    "resultat_net_exercice": int(ca * 0.08),
                    "total_capitaux_propres": int(ca * 0.98)
                },
                "passif_non_courant": {
                    "emprunts_dettes_financieres_lt": 0,
                    "provisions_lt": 0,
                    "total_passif_non_courant": 0
                },
                "passif_courant": {
                    "fournisseurs_et_comptes_rattaches": int(ca * 0.1),
                    "dettes_fiscales_et_sociales": int(ca * 0.15),
                    "autres_dettes_ct": 0,
                    "total_passif_courant": int(ca * 0.25)
                },
                "total_passif": int(ca * 1.2)
            }
        },
        "compte_de_resultat": {
            "produits_exploitation": {
                "chiffre_affaires": int(ca),
                "production_immobilisee": 0,
                "subventions_exploitation": 0,
                "autres_produits_exploitation": 0,
                "total_produits_exploitation": int(ca)
            },
            "charges_exploitation": {
                "achats_consommes": int(ca * 0.6),
                "charges_personnel": int(ca * 0.2),
                "dotations_amortissements": int(ca * 0.05),
                "autres_charges_exploitation": int(ca * 0.07),
                "total_charges_exploitation": int(ca * 0.92)
            },
            "resultat_exploitation": int(ca * 0.08),
            "resultat_financier": 0,
            "resultat_exceptionnel": 0,
            "resultat_avant_impot": int(ca * 0.08),
            "impot_sur_benefices": 0,
            "resultat_net": int(ca * 0.08)
        },
        "ratios_financiers": {
            "marge_brute_percent": 40.0,
            "marge_nette_percent": 8.0,
            "rentabilite_actif_percent": 6.7,
            "liquidite_generale": 4.0,
            "autonomie_financiere_percent": 81.7
        },
        "analyse_financiere": {
            "points_forts": ["Bonne rentabilité avec marge nette de 8%", "Liquidité élevée", "Forte autonomie financière"],
            "points_faibles": ["Dépendance aux créances clients", "Pas de diversification"],
            "recommandations": ["Améliorer le recouvrement des créances", "Diversifier les sources de revenus", "Optimiser la gestion de trésorerie"]
        },
        "details_transactions": []
    }
    
    return result

def fix_json_format(json_str: str) -> str:
    """Fix common JSON formatting issues from Groq"""
    import re
    
    # Fix the main issue: commas in numbers (18,000 -> 18000)
    # Handle various number formats with commas
    json_str = re.sub(r':\s*(\d{1,3}),(\d{3})\b', r': \1\2', json_str)
    json_str = re.sub(r':\s*(\d{1,3}),(\d{3}),(\d{3})\b', r': \1\2\3', json_str)
    
    # Fix decimal numbers with commas (25,735.50 -> 25735.50)
    json_str = re.sub(r':\s*(\d{1,3}),(\d{3})\.(\d+)', r': \1\2.\3', json_str)
    
    # Fix the specific issue: missing space after comma in JSON structure
    # "18000,"immobilisations" -> "18000", "immobilisations"
    json_str = re.sub(r'(\d+),"([a-zA-Z_])', r'\1, "\2', json_str)
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes
    json_str = json_str.replace("'", '"')
    
    return json_str

def parse_groq_bilan_response(response: str) -> Dict[str, Any]:
    """Parse Groq's bilan JSON response"""
    try:
        import re
        import json
        
        # Clean the response
        cleaned_response = response.strip()
        
        # Handle multiple JSON blocks in markdown - reconstruct complete structure
        if '```' in cleaned_response:
            # Initialize complete bilan structure
            complete_bilan = {
                "bilan_comptable": {"actif": {"total_actif": 0}, "passif": {"total_passif": 0}},
                "compte_de_resultat": {"resultat_net": 0},
                "ratios_financiers": {},
                "analyse_financiere": {"points_forts": [], "points_faibles": [], "recommandations": []},
                "details_transactions": []
            }
            
            # Extract all code blocks (both JSON objects and arrays)
            # Use a more robust regex that captures complete JSON structures
            code_blocks = []
            
            # Find all ``` blocks
            block_pattern = r'```[^`]*?([\s\S]*?)```'
            matches = re.findall(block_pattern, cleaned_response)
            logger.debug(f"Regex found {len(matches)} total markdown blocks")
            
            for i, match in enumerate(matches, 1):
                # Clean the match and check if it's JSON
                cleaned_match = match.strip()
                logger.debug(f"Block {i}: starts with '{cleaned_match[:50]}...', is JSON: {cleaned_match.startswith('{') or cleaned_match.startswith('[')}")
                if cleaned_match.startswith('{') or cleaned_match.startswith('['):
                    code_blocks.append(cleaned_match)
            
            logger.info(f"Found {len(code_blocks)} JSON code blocks out of {len(matches)} total blocks")
            
            # Parse each JSON block and merge into complete structure
            for i, block in enumerate(code_blocks):
                try:
                    # Try multiple approaches to parse the JSON block
                    try:
                        # First try: direct parsing
                        logger.debug(f"Trying direct parsing for block {i+1}")
                        parsed = json.loads(block)
                        logger.info(f"Direct parsing succeeded for block {i+1}")
                    except json.JSONDecodeError as e1:
                        logger.debug(f"Direct parsing failed: {e1}")
                        try:
                            # Second try: aggressive number fixing first (most common issue)
                            import re
                            logger.debug(f"Trying aggressive number fixing for block {i+1}")
                            aggressive_fix = re.sub(r'(\d+),(\d+)', r'\1\2', block)
                            parsed = json.loads(aggressive_fix)
                            logger.info(f"Aggressive number fixing succeeded for block {i+1}")
                        except json.JSONDecodeError as e2:
                            logger.debug(f"Aggressive number fixing failed: {e2}")
                            try:
                                # Third try: comprehensive formatting fixes
                                logger.debug(f"Trying comprehensive fixes for block {i+1}")
                                fixed_block = fix_json_format(block)
                                # Also remove commas from numbers in the fixed version
                                fixed_block = re.sub(r'(\d+),(\d+)', r'\1\2', fixed_block)
                                parsed = json.loads(fixed_block)
                                logger.info(f"Comprehensive fixes succeeded for block {i+1}")
                            except json.JSONDecodeError as e3:
                                logger.debug(f"Comprehensive fixes failed: {e3}")
                                # Fourth try: remove all commas from numbers (very aggressive)
                                logger.debug(f"Trying super aggressive fixes for block {i+1}")
                                super_aggressive = re.sub(r'(\d+),(\d+),(\d+)', r'\1\2\3', block)
                                super_aggressive = re.sub(r'(\d+),(\d+)', r'\1\2', super_aggressive)
                                parsed = json.loads(super_aggressive)
                                logger.info(f"Super aggressive fixes succeeded for block {i+1}")
                except json.JSONDecodeError:
                    # If still failing, try more aggressive fixing
                    try:
                        # Use Python's ast.literal_eval as a fallback for simple cases
                        import ast
                        # Replace 'true'/'false' with Python equivalents
                        python_block = block.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                        parsed = ast.literal_eval(python_block)
                    except:
                        # Skip this block if all parsing attempts fail
                        logger.warning(f"All parsing attempts failed for block {i+1}, skipping")
                        continue
                    
                    if isinstance(parsed, dict):
                        logger.info(f"Parsed JSON keys: {list(parsed.keys())}")
                        logger.info(f"Parsed JSON has {len(parsed.keys())} keys")
                        
                        # IMMEDIATELY return if this looks like a complete bilan
                        if "bilan_comptable" in parsed:
                            logger.info("FOUND COMPLETE BILAN - RETURNING DIRECTLY!")
                            return parsed
                        
                        if "bilan_comptable" in parsed:
                            logger.debug(f"Found bilan_comptable with actif total_actif: {parsed.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 'NOT_FOUND')}")
                        if "compte_de_resultat" in parsed:
                            logger.debug(f"Found compte_de_resultat with chiffre_affaires: {parsed.get('compte_de_resultat', {}).get('produits_exploitation', {}).get('chiffre_affaires', 'NOT_FOUND')}")
                        
                        # Merge dictionary into complete structure
                        for key, value in parsed.items():
                            if key in complete_bilan:
                                if isinstance(value, dict) and isinstance(complete_bilan[key], dict):
                                    # Deep merge
                                    def deep_merge(target, source):
                                        for k, v in source.items():
                                            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                                                deep_merge(target[k], v)
                                            else:
                                                target[k] = v
                                    deep_merge(complete_bilan[key], value)
                                    logger.debug(f"Deep merged {key}")
                                else:
                                    complete_bilan[key] = value
                                    logger.debug(f"Direct assigned {key}")
                            else:
                                complete_bilan[key] = value
                                logger.debug(f"Added new key {key}")
                        
                        logger.debug(f"After merge, complete_bilan actif total: {complete_bilan.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 'NOT_FOUND')}")
                                
                    elif isinstance(parsed, list):
                        # Handle array (details_transactions)
                        complete_bilan["details_transactions"] = parsed
                        
                except json.JSONDecodeError as e:
                    logger.error(f"All parsing attempts failed for JSON block {i+1}: {e}")
                    logger.error(f"Original block (first 300 chars): {block[:300]}...")
                    logger.error(f"Error position: line {e.lineno}, column {e.colno}")
                    continue
            
            # Check if any real data was extracted (more flexible check)
            logger.debug(f"Validating complete_bilan structure: {list(complete_bilan.keys())}")
            
            has_actif_data = complete_bilan.get("bilan_comptable", {}).get("actif", {}).get("total_actif", 0) > 0
            has_revenue_data = complete_bilan.get("compte_de_resultat", {}).get("chiffre_affaires", 0) > 0 or complete_bilan.get("compte_de_resultat", {}).get("produits_exploitation", {}).get("chiffre_affaires", 0) > 0
            has_transaction_data = len(complete_bilan.get("details_transactions", [])) > 0
            
            logger.info(f"Validation details: actif_data={has_actif_data} (value: {complete_bilan.get('bilan_comptable', {}).get('actif', {}).get('total_actif', 0)})")
            logger.info(f"Validation details: revenue_data={has_revenue_data} (values: {complete_bilan.get('compte_de_resultat', {}).get('chiffre_affaires', 0)}, {complete_bilan.get('compte_de_resultat', {}).get('produits_exploitation', {}).get('chiffre_affaires', 0)})")
            logger.info(f"Validation details: transaction_data={has_transaction_data} (count: {len(complete_bilan.get('details_transactions', []))})")
            
            if not (has_actif_data or has_revenue_data or has_transaction_data):
                logger.error("No meaningful data was successfully extracted from JSON blocks")
                logger.error(f"Complete bilan keys: {list(complete_bilan.keys())}")
                logger.error(f"Bilan comptable keys: {list(complete_bilan.get('bilan_comptable', {}).keys())}")
                return {"error": "Failed to parse any JSON blocks", "raw_response": response[:1000]}
            else:
                logger.info(f"Data validation passed: actif={has_actif_data}, revenue={has_revenue_data}, transactions={has_transaction_data}")
            
            logger.info("Successfully merged multiple JSON blocks into complete bilan structure")
            return complete_bilan
        
        # If no markdown blocks, try to parse as single JSON
        logger.info("No markdown blocks found, trying single JSON parsing")
        logger.debug(f"Response length: {len(cleaned_response)}, starts with: {cleaned_response[:100]}...")
        
        # Remove any remaining text before JSON starts
        if '{' in cleaned_response:
            start_idx = cleaned_response.find('{')
            cleaned_response = cleaned_response[start_idx:]
            logger.debug(f"Found JSON start at position {start_idx}")
        
        # Remove any text after JSON ends
        if '}' in cleaned_response:
            end_idx = cleaned_response.rfind('}') + 1
            cleaned_response = cleaned_response[:end_idx]
            logger.debug(f"JSON ends at position {end_idx}")
        
        cleaned_response = cleaned_response.strip()
        logger.debug(f"Cleaned JSON length: {len(cleaned_response)}")
        
        # Try multiple approaches to parse the JSON
        try:
            # First try: direct parsing
            logger.debug("Trying direct JSON parsing")
            bilan_data = json.loads(cleaned_response)
            logger.info("Successfully parsed Groq JSON directly")
            return bilan_data
        except json.JSONDecodeError as e:
            logger.error(f"First JSON parse failed: {e}")
            # Show the problematic area
            error_pos = getattr(e, 'pos', 416)
            start = max(0, error_pos - 50)
            end = min(len(cleaned_response), error_pos + 50)
            logger.error(f"Problematic area: '{cleaned_response[start:end]}'")
            
            try:
                # Second try: fix formatting and parse
                fixed_response = fix_json_format(cleaned_response)
                logger.info(f"After fixing: '{fixed_response[start:end]}'")
                bilan_data = json.loads(fixed_response)
                logger.info("Successfully parsed Groq JSON after formatting fixes")
                return bilan_data
            except json.JSONDecodeError as e2:
                logger.error(f"Second JSON parse failed: {e2}")
                try:
                    # Third try: more aggressive number fixing
                    import re
                    aggressive_fix = re.sub(r'(\d+),(\d+)', r'\1\2', cleaned_response)
                    logger.info(f"Aggressive fix: '{aggressive_fix[start:end]}'")
                    bilan_data = json.loads(aggressive_fix)
                    logger.info("Successfully parsed Groq JSON with aggressive number fixing")
                    return bilan_data
                except json.JSONDecodeError as e3:
                    logger.error(f"All JSON parsing attempts failed: {e3}")
                    # Continue to the outer exception handler
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Groq bilan JSON: {e}")
        logger.error(f"Raw response: {response}")
        
        # Return fallback structure
        return {
            "error": "Failed to parse Groq response",
            "raw_response": response,
            "bilan_comptable": {
                "actif": {"total_actif": 0},
                "passif": {"total_passif": 0}
            },
            "compte_de_resultat": {
                "resultat_net": 0
            }
        }

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
        logger.info(f"Rule-based classification result: {doc_type}")
        
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
        return ClassificationResponse(
            document_id=cloudinary_response["public_id"],
            model_prediction=model_result["prediction"],
            model_confidence=model_result["confidence"],
            rule_based_prediction=doc_type,
            final_prediction=doc_type if doc_type != "❓ Unknown Document Type" else model_result["prediction"],
            confidence_flag="high" if model_result["confidence"] >= 0.7 else "low",
            confidence_scores=model_result["confidence_scores"],
            text_excerpt=text,
            processing_time_ms=processing_time,
            cloudinary_url=cloudinary_response["secure_url"],
            public_id=cloudinary_response["public_id"],
            extracted_info=extracted_info,
            document_embedding=document_embedding,
            embedding_model=embedding_model_name
        )
        
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