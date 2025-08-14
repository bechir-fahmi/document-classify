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

@app.post("/bilan/")
async def generate_tunisian_bilan(
    request: Dict[str, Any]
):
    """
    Generate Tunisian accounting bilan (Bilan Comptable + Compte de Résultat) using Groq AI
    
    Request Body:
    {
        "documents": [
            {
                "id": "14edca9d-4f10-4afb-9e9d-7e11ce6d9544",
                "filename": "FACTURE_01-03-2025_NÂ°2220000032299099.pdf",
                "document_type": "invoice",
                "confidence": 0.959465650679308,
                "extracted_text": "Full OCR text...",
                "extracted_info": "{\"date\": \"2025-03-31\", \"client_name\": \"M. BACHIR FAHMI\"}",
                "created_at": "2025-07-27 03:22:38.85642"
            }
        ],
        "period_days": 30
    }
    
    Returns: Complete Tunisian accounting bilan following Plan Comptable Tunisien
    """
    start_time = time.time()
    
    try:
        # Extract parameters
        documents = request.get("documents", [])
        period_days = request.get("period_days", 365)  # Default to annual bilan
        document_only = request.get("document_only", False)  # New parameter for document-only mode
        
        # Validate period_days - only quarterly or annual allowed
        if period_days not in [90, 365]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid period_days. Only quarterly (90 days) or annual (365 days) bilans are supported."
            )
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        mode = "document-only" if document_only else "full accounting"
        logger.info(f"Generating Tunisian bilan ({mode}) from {len(documents)} documents for {period_days} days")
        
        # Prepare all document texts for Groq analysis
        documents_data = []
        for doc in documents:
            # Clean extracted_text - handle nested JSON
            extracted_text = doc.get("extracted_text", "")
            cleaned_text = clean_extracted_text(extracted_text)
            
            documents_data.append({
                "id": doc["id"],
                "filename": doc.get("filename", "unknown"),
                "document_type": doc.get("document_type", "unknown"),
                "extracted_text": cleaned_text,
                "date": doc.get("created_at", ""),
                "extracted_info": doc.get("extracted_info", "{}")
            })
        
        # Use Groq to generate bilan (document-only or full accounting mode)
        if document_only:
            bilan_result = await generate_document_only_bilan(documents_data, period_days)
        else:
            bilan_result = await generate_tunisian_bilan_with_groq(documents_data, period_days)
        
        # Add metadata
        bilan_result["metadata"] = {
            "documents_processed": len(documents),
            "period_days": period_days,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "generated_at": datetime.now().isoformat(),
            "standard": "Plan Comptable Tunisien"
        }
        
        logger.info(f"Generated Tunisian bilan in {(time.time() - start_time)*1000:.1f}ms")
        
        return bilan_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating Tunisian bilan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating bilan: {str(e)}")

async def generate_document_only_bilan(documents_data: List[Dict], period_days: int) -> Dict[str, Any]:
    """Generate bilan using ONLY data found in documents - no artificial additions"""
    
    try:
        # Create document-only prompt
        prompt = create_document_only_bilan_prompt(documents_data, period_days)
        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
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

async def generate_tunisian_bilan_with_groq(documents_data: List[Dict], period_days: int) -> Dict[str, Any]:
    """Generate Tunisian accounting bilan using Groq AI"""
    
    try:
        # Create comprehensive prompt for Tunisian bilan generation
        prompt = create_tunisian_bilan_prompt(documents_data, period_days)
        
        # Call Groq API
        from utils.groq_utils import client as groq_client
        
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
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
        
        # Parse JSON response - handle multiple JSON blocks from Groq
        logger.info("Starting JSON parsing...")
        bilan_data = parse_groq_bilan_response(groq_response)
        logger.info(f"JSON parsing result: {type(bilan_data)}, keys: {list(bilan_data.keys()) if isinstance(bilan_data, dict) else 'Not a dict'}")
        
        # Check if parsing succeeded
        if "error" in bilan_data:
            logger.error("JSON parsing failed - Groq response could not be parsed")
            logger.error(f"Error details: {bilan_data.get('error', 'Unknown error')}")
            # Instead of returning error, try to extract at least some data
            logger.info("Attempting to extract basic data from Groq response")
            bilan_data = extract_basic_data_from_response(groq_response)
        else:
            logger.info("Successfully parsed Groq's raw response - returning as-is")
        
        # Return Groq's data exactly as generated - no validation or fixing
        logger.info("Returning Groq's raw data without modifications")
        
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
    for i, doc in enumerate(documents_data[:10], 1):
        documents_summary += f"""
Document {i}:
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Date: {doc['date']}
- Content: {doc['extracted_text'][:800]}...
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
    "document_transactions": [
        {{
            "document_id": "doc_id",
            "document_type": "type",
            "filename": "filename",
            "date": "date_if_found",
            "amounts": {{
                "total_amount": <amount_if_found>,
                "tax_amount": <tax_if_found>,
                "net_amount": <net_if_found>,
                "currency": "currency_if_found"
            }},
            "description": "description_from_document",
            "parties": {{
                "company": "company_name_if_found",
                "client_customer": "client_name_if_found"
            }}
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
    for i, doc in enumerate(documents_data[:10], 1):  # Limit to first 10 for prompt size
        documents_summary += f"""
Document {i}:
- Type: {doc['document_type']}
- Filename: {doc['filename']}
- Date: {doc['date']}
- Content: {doc['extracted_text'][:500]}...
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
        "points_forts": ["strength1", "strength2"],
        "points_faibles": ["weakness1", "weakness2"],
        "recommandations": ["recommendation1", "recommendation2"]
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

Instructions:
1. EXTRACT FINANCIAL DATA: Process mixed languages/currencies and extract meaningful transactions
2. CURRENCY CONVERSION: Convert to TND (USD≈3.1, EUR≈3.3, TND=1.0)
3. CALCULATE TOTALS: Sum all revenue and expenses from the documents
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
                                else:
                                    complete_bilan[key] = value
                            else:
                                complete_bilan[key] = value
                                
                    elif isinstance(parsed, list):
                        # Handle array (details_transactions)
                        complete_bilan["details_transactions"] = parsed
                        
                except json.JSONDecodeError as e:
                    logger.error(f"All parsing attempts failed for JSON block {i+1}: {e}")
                    logger.error(f"Original block (first 300 chars): {block[:300]}...")
                    logger.error(f"Error position: line {e.lineno}, column {e.colno}")
                    continue
            
            # Check if any real data was extracted (more flexible check)
            has_actif_data = complete_bilan.get("bilan_comptable", {}).get("actif", {}).get("total_actif", 0) > 0
            has_revenue_data = complete_bilan.get("compte_de_resultat", {}).get("chiffre_affaires", 0) > 0 or complete_bilan.get("compte_de_resultat", {}).get("produits_exploitation", {}).get("chiffre_affaires", 0) > 0
            has_transaction_data = len(complete_bilan.get("details_transactions", [])) > 0
            
            if not (has_actif_data or has_revenue_data or has_transaction_data):
                logger.error("No meaningful data was successfully extracted from JSON blocks")
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