"""
Service Factory - Dependency Injection Container
Creates and manages service dependencies following SOLID principles
"""
import logging
from typing import Optional

from api.interfaces.document_classifier import (
    IDocumentClassifier, IDocumentAnalyzer, ITextExtractor,
    IEmbeddingService, ICloudStorageService, IFinancialAnalyzer
)
from api.adapters.ml_classifier_adapter import MLClassifierAdapter
from api.adapters.document_analyzer_adapter import DocumentAnalyzerAdapter
from api.adapters.text_extractor_adapter import TextExtractorAdapter
from api.adapters.embedding_service_adapter import EmbeddingServiceAdapter
from api.adapters.cloudinary_adapter import CloudinaryAdapter
from api.adapters.financial_analyzer_adapter import (
    FinancialAnalyzerAdapter, GroqFinancialAnalyzerAdapter
)
from api.services.document_classification_service import DocumentClassificationService
from api.services.financial_analysis_service import FinancialAnalysisService
from api.controllers.classification_controller import ClassificationController
from api.controllers.financial_controller import FinancialController

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory for creating and managing service dependencies"""
    
    def __init__(self):
        self._instances = {}
    
    def get_document_classifier(self, model_type: Optional[str] = None) -> IDocumentClassifier:
        """Get document classifier instance"""
        key = f"classifier_{model_type}"
        if key not in self._instances:
            self._instances[key] = MLClassifierAdapter(model_type)
        return self._instances[key]
    
    def get_document_analyzer(self) -> IDocumentAnalyzer:
        """Get document analyzer instance"""
        if "analyzer" not in self._instances:
            self._instances["analyzer"] = DocumentAnalyzerAdapter()
        return self._instances["analyzer"]
    
    def get_text_extractor(self) -> ITextExtractor:
        """Get text extractor instance"""
        if "text_extractor" not in self._instances:
            self._instances["text_extractor"] = TextExtractorAdapter()
        return self._instances["text_extractor"]
    
    def get_embedding_service(self) -> IEmbeddingService:
        """Get embedding service instance"""
        if "embedding_service" not in self._instances:
            self._instances["embedding_service"] = EmbeddingServiceAdapter()
        return self._instances["embedding_service"]
    
    def get_cloud_storage_service(self) -> ICloudStorageService:
        """Get cloud storage service instance"""
        if "cloud_storage" not in self._instances:
            self._instances["cloud_storage"] = CloudinaryAdapter()
        return self._instances["cloud_storage"]
    
    def get_financial_analyzer(self, use_groq: bool = True) -> IFinancialAnalyzer:
        """Get financial analyzer instance"""
        key = f"financial_analyzer_{'groq' if use_groq else 'standard'}"
        if key not in self._instances:
            if use_groq:
                self._instances[key] = GroqFinancialAnalyzerAdapter()
            else:
                self._instances[key] = FinancialAnalyzerAdapter()
        return self._instances[key]
    
    def get_classification_service(self, model_type: Optional[str] = None) -> DocumentClassificationService:
        """Get document classification service instance"""
        key = f"classification_service_{model_type}"
        if key not in self._instances:
            self._instances[key] = DocumentClassificationService(
                classifier=self.get_document_classifier(model_type),
                analyzer=self.get_document_analyzer(),
                text_extractor=self.get_text_extractor(),
                embedding_service=self.get_embedding_service(),
                cloud_storage=self.get_cloud_storage_service()
            )
        return self._instances[key]
    
    def get_financial_service(self, use_groq: bool = True, model_type: Optional[str] = None) -> FinancialAnalysisService:
        """Get financial analysis service instance"""
        key = f"financial_service_{'groq' if use_groq else 'standard'}_{model_type}"
        if key not in self._instances:
            self._instances[key] = FinancialAnalysisService(
                financial_analyzer=self.get_financial_analyzer(use_groq),
                text_extractor=self.get_text_extractor(),
                classification_service=self.get_classification_service(model_type)
            )
        return self._instances[key]
    
    def get_classification_controller(self, model_type: Optional[str] = None) -> ClassificationController:
        """Get classification controller instance"""
        key = f"classification_controller_{model_type}"
        if key not in self._instances:
            self._instances[key] = ClassificationController(
                classification_service=self.get_classification_service(model_type)
            )
        return self._instances[key]
    
    def get_financial_controller(self, use_groq: bool = True, model_type: Optional[str] = None) -> FinancialController:
        """Get financial controller instance"""
        key = f"financial_controller_{'groq' if use_groq else 'standard'}_{model_type}"
        if key not in self._instances:
            self._instances[key] = FinancialController(
                financial_service=self.get_financial_service(use_groq, model_type)
            )
        return self._instances[key]


# Global factory instance
service_factory = ServiceFactory()