"""
API Response Models - Single Responsibility Principle
Defines all API response data models
"""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ClassificationResponse(BaseModel):
    """Response model for document classification"""
    model_config = {"protected_namespaces": ()}
    
    document_id: str
    model_prediction: str
    model_confidence: float
    rule_based_prediction: str
    final_prediction: str
    confidence_flag: str
    confidence_scores: Dict[str, float]
    text_excerpt: str
    processing_time_ms: float
    cloudinary_url: Optional[str] = None
    public_id: Optional[str] = None
    extracted_info: Optional[Dict[str, Any]] = None
    document_embedding: List[float]
    embedding_model: str


class CommercialDocumentResponse(BaseModel):
    """Response model for commercial document analysis"""
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
    """Response model for errors"""
    error: str
    detail: Optional[str] = None


class FinancialTransactionResponse(BaseModel):
    """Response model for financial transaction analysis"""
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
    """Response model for Groq-powered financial analysis"""
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
    """Response model for financial bilan"""
    period: Dict[str, Any]
    summary: Dict[str, float]
    currency_breakdown: Dict[str, Dict[str, float]]
    category_breakdown: Optional[Dict[str, Dict[str, Any]]] = {}
    transaction_count: int
    recommendations: List[str]
    generated_at: str
    processing_time_ms: Optional[float] = None
    processed_documents: Optional[int] = None


class DocumentFinancialAnalysisResponse(BaseModel):
    """Response model for combined document classification and financial analysis"""
    document_id: str
    financial_transaction: FinancialTransactionResponse
    document_classification: Dict[str, Any]
    document_embedding: List[float]
    embedding_model: str
    processing_time_ms: float