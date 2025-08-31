"""
ML Classifier Adapter - Adapter Pattern
Adapts existing ML models to the IDocumentClassifier interface
"""
import logging
from typing import Dict, Any

from api.interfaces.document_classifier import IDocumentClassifier
from models import get_model

logger = logging.getLogger(__name__)


class MLClassifierAdapter(IDocumentClassifier):
    """Adapter for existing ML classification models"""
    
    def __init__(self, model_type: str = None):
        self._model_type = model_type
        self._model = None
        self._load_model()
    
    def _load_model(self):
        """Load the ML model"""
        try:
            self._model = get_model(self._model_type)
            logger.info(f"Loaded ML model: {self._model_type}")
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            raise
    
    def classify(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Classify document text using ML model"""
        try:
            if hasattr(self._model, 'hybrid_predict'):
                # Use hybrid prediction if available
                result = self._model.hybrid_predict(
                    text, 
                    document_id=document_id, 
                    text_excerpt=text[:500]
                )
                return {
                    "prediction": result.get("final_prediction"),
                    "confidence": result.get("model_confidence", 0),
                    "confidence_scores": result.get("confidence_scores", {})
                }
            else:
                # Use standard prediction
                result = self._model.predict(text)
                return {
                    "prediction": result.get("prediction"),
                    "confidence": result.get("confidence", 0),
                    "confidence_scores": result.get("confidence_scores", {})
                }
        except Exception as e:
            logger.error(f"Error in ML classification: {str(e)}")
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "confidence_scores": {}
            }
    
    def get_confidence_scores(self, text: str) -> Dict[str, float]:
        """Get confidence scores for all document classes"""
        try:
            result = self.classify(text)
            return result.get("confidence_scores", {})
        except Exception as e:
            logger.error(f"Error getting confidence scores: {str(e)}")
            return {}