"""
Document Classifier Interface - Dependency Inversion Principle
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class IDocumentClassifier(ABC):
    """Interface for document classification services"""
    
    @abstractmethod
    def classify(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Classify a document based on its text content"""
        pass
    
    @abstractmethod
    def get_confidence_scores(self, text: str) -> Dict[str, float]:
        """Get confidence scores for all document classes"""
        pass


class IDocumentAnalyzer(ABC):
    """Interface for document analysis services"""
    
    @abstractmethod
    def analyze(self, file_path: str) -> str:
        """Analyze document and return document type"""
        pass
    
    @abstractmethod
    def extract_metadata(self, text: str, doc_type: str = None) -> Dict[str, Any]:
        """Extract metadata from document text"""
        pass


class ITextExtractor(ABC):
    """Interface for text extraction services"""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text from document file"""
        pass


class IFinancialAnalyzer(ABC):
    """Interface for financial analysis services"""
    
    @abstractmethod
    def analyze_financial_data(self, text: str, document_type: str, document_id: str) -> Dict[str, Any]:
        """Analyze financial information from document text"""
        pass


class IEmbeddingService(ABC):
    """Interface for document embedding services"""
    
    @abstractmethod
    def embed_document(self, text: str) -> List[float]:
        """Generate document embedding"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class ICloudStorageService(ABC):
    """Interface for cloud storage services"""
    
    @abstractmethod
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """Upload document to cloud storage"""
        pass