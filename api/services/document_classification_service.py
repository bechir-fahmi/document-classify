"""
Document Classification Service - Single Responsibility Principle
Handles document classification logic
"""
import logging
import time
import uuid
from typing import Dict, Any, Optional

from api.interfaces.document_classifier import (
    IDocumentClassifier, IDocumentAnalyzer, ITextExtractor, 
    IEmbeddingService, ICloudStorageService
)

logger = logging.getLogger(__name__)


class DocumentClassificationService:
    """Service for document classification - follows SRP"""
    
    def __init__(
        self,
        classifier: IDocumentClassifier,
        analyzer: IDocumentAnalyzer,
        text_extractor: ITextExtractor,
        embedding_service: IEmbeddingService,
        cloud_storage: ICloudStorageService
    ):
        self._classifier = classifier
        self._analyzer = analyzer
        self._text_extractor = text_extractor
        self._embedding_service = embedding_service
        self._cloud_storage = cloud_storage
    
    def classify_document(self, file_path: str, upload_to_cloud: bool = True) -> Dict[str, Any]:
        """
        Classify a document file
        
        Args:
            file_path: Path to the document file
            upload_to_cloud: Whether to upload to cloud storage
            
        Returns:
            Classification result with metadata
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            # Extract text
            text = self._text_extractor.extract_text(file_path)
            
            # Rule-based analysis
            rule_based_prediction = self._analyzer.analyze(file_path)
            
            # ML-based classification
            classification_result = self._classifier.classify(text, document_id)
            
            # Determine final prediction
            final_prediction = self._determine_final_prediction(
                rule_based_prediction, 
                classification_result
            )
            
            # Generate embedding
            document_embedding = self._embedding_service.embed_document(text)
            
            # Upload to cloud if requested
            cloud_result = None
            if upload_to_cloud:
                cloud_result = self._cloud_storage.upload_document(file_path)
            
            # Extract metadata with document type
            metadata = self._analyzer.extract_metadata(text, final_prediction)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "document_id": document_id,
                "model_prediction": classification_result.get("prediction"),
                "model_confidence": classification_result.get("confidence"),
                "rule_based_prediction": rule_based_prediction,
                "final_prediction": final_prediction,
                "confidence_flag": self._get_confidence_flag(classification_result.get("confidence", 0)),
                "confidence_scores": classification_result.get("confidence_scores", {}),
                "text_excerpt": text[:500],
                "processing_time_ms": processing_time,
                "cloudinary_url": cloud_result.get("url") if cloud_result else None,
                "public_id": cloud_result.get("public_id") if cloud_result else None,
                "extracted_info": metadata,
                "document_embedding": document_embedding,
                "embedding_model": "all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            raise
    
    def _determine_final_prediction(self, rule_based: str, ml_result: Dict[str, Any]) -> str:
        """Determine final prediction using hybrid approach"""
        # Clean rule-based prediction
        final_prediction = rule_based.replace("❓ Unknown Document Type", "unknown")
        
        if final_prediction == "unknown":
            # Use ML prediction if confidence is reasonable
            if ml_result.get("confidence", 0) > 0.2:
                final_prediction = ml_result.get("prediction", "unknown")
            else:
                # Find highest confidence prediction
                confidence_scores = ml_result.get("confidence_scores", {})
                if confidence_scores:
                    best_prediction = max(confidence_scores.items(), key=lambda x: x[1])
                    if best_prediction[1] > 0.25:
                        final_prediction = best_prediction[0]
        
        return final_prediction
    
    def _get_confidence_flag(self, confidence: float) -> str:
        """Get confidence flag based on confidence score"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"