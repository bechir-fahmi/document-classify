import os
import logging
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BertClassifier:
    """
    Document classifier using BERT
    """
    
    def __init__(self):
        self.model_dir = os.path.join(config.MODEL_DIR, "bert")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = config.DOCUMENT_CLASSES
        
        self.load_model()
    
    def load_model(self):
        """Load the model from disk if it exists"""
        model_path = os.path.join(self.model_dir, "model")
        tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            logger.info(f"Loading existing BERT model from {model_path}")
            try:
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
                self.model = BertForSequenceClassification.from_pretrained(model_path)
                self.model.to(self.device)
                return True
            except Exception as e:
                logger.error(f"Error loading BERT model: {str(e)}")
                self.model = None
                self.tokenizer = None
                return False
        else:
            logger.info("No existing BERT model found, initializing new model")
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.classes),
                    output_attentions=False,
                    output_hidden_states=False
                )
                self.model.to(self.device)
                return False
            except Exception as e:
                logger.error(f"Error initializing BERT model: {str(e)}")
                return False
    
    def save_model(self):
        """Save the model to disk"""
        if self.model is not None and self.tokenizer is not None:
            model_path = os.path.join(self.model_dir, "model")
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            
            logger.info(f"Saving BERT model to {model_path}")
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(tokenizer_path)
            return True
        else:
            logger.warning("No BERT model to save")
            return False
    
    def train(self, X, y, test_size=0.2, epochs=3, batch_size=16):
        """Train the BERT model on the given data"""
        logger.info(f"Training BERT model on {len(X)} documents")
        
        # Ensure model and tokenizer are initialized
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Convert class labels to indices
        label_map = {label: i for i, label in enumerate(self.classes)}
        y_indices = [label_map[label] for label in y]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_indices, test_size=test_size, random_state=42, stratify=y_indices
        )
        
        # Tokenize the input text
        train_encodings = self.tokenizer(
            X_train, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        
        # Create tensor datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'], 
            train_encodings['attention_mask'],
            torch.tensor(y_train)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            for batch in train_loader:
                # Get batch data
                batch_input_ids = batch[0].to(self.device)
                batch_attention_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
        
        # Save the model
        self.save_model()
        
        return {'status': 'success'}
    
    def predict(self, text):
        """Predict the class of a document using BERT"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("BERT model not trained yet")
        
        # Tokenize the input text
        encoding = self.tokenizer(
            text, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Get predicted class and confidence scores
        predicted_idx = np.argmax(probabilities)
        prediction = self.classes[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Create confidence dict
        confidence_scores = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        
        # Check if confidence is high enough
        if confidence < config.CONFIDENCE_THRESHOLD:
            prediction = "unknown"
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'confidence_scores': confidence_scores
        }