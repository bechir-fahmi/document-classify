"""
Document embedding utilities using sentence-transformers
"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import config

logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """Document embedding class using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document embedder
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            numpy array containing the embedding
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array containing the embeddings
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Limit text length to avoid memory issues (sentence-transformers has token limits)
        max_length = 512  # tokens, roughly 2000-3000 characters
        if len(text) > max_length * 6:  # rough estimate
            text = text[:max_length * 6]
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
        
        Returns:
            Embedding dimension
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        return self.model.get_sentence_embedding_dimension()

# Global embedder instance
_embedder = None

def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> DocumentEmbedder:
    """
    Get a global embedder instance (singleton pattern)
    
    Args:
        model_name: Name of the sentence-transformer model
        
    Returns:
        DocumentEmbedder instance
    """
    global _embedder
    
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = DocumentEmbedder(model_name)
    
    return _embedder

def embed_document(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """
    Convenience function to embed a document
    
    Args:
        text: Document text to embed
        model_name: Name of the sentence-transformer model
        
    Returns:
        List of floats representing the embedding
    """
    embedder = get_embedder(model_name)
    embedding = embedder.embed_text(text)
    return embedding.tolist()

def embed_documents(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Convenience function to embed multiple documents
    
    Args:
        texts: List of document texts to embed
        model_name: Name of the sentence-transformer model
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    embedder = get_embedder(model_name)
    embeddings = embedder.embed_texts(texts)
    return embeddings.tolist()