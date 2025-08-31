"""
Embedding Service Adapter - Adapter Pattern
Adapts existing embedding utilities to the interface
"""
import logging
from typing import List

from api.interfaces.document_classifier import IEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingServiceAdapter(IEmbeddingService):
    """Adapter for existing embedding utilities"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
    
    def embed_document(self, text: str) -> List[float]:
        """Generate document embedding"""
        try:
            from utils.embedding_utils import embed_document
            return embed_document(text, self._model_name)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        try:
            from utils.embedding_utils import get_embedder
            embedder = get_embedder()
            return embedder.get_embedding_dimension()
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            return 384  # Default dimension for all-MiniLM-L6-v2