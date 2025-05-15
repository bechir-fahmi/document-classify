import logging
import config
from .sklearn_model import SklearnClassifier
from .bert_model import BertClassifier
from .layoutlm_model import LayoutLMClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model(model_type=None):
    """
    Factory function to get the appropriate model based on model_type
    
    Args:
        model_type: Type of model to use. If None, uses the default model from config
        
    Returns:
        Model instance
    """
    if model_type is None:
        model_type = config.DEFAULT_MODEL
    
    logger.info(f"Creating model of type: {model_type}")
    
    if model_type == "sklearn_tfidf_svm":
        return SklearnClassifier()
    elif model_type == "bert":
        return BertClassifier()
    elif model_type == "layoutlm":
        return LayoutLMClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 