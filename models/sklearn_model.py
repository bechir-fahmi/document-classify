import os
import pickle
import numpy as np
import pandas as pd
import logging
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import config
from utils.document_analyzer import DOCUMENT_PATTERNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(text):
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    return text.lower().strip()

def rule_based_classification(text):
    norm_text = normalize_text(text)
    best_type = None
    best_score = 0
    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        score = 0
        for kw in patterns.get('keywords', []):
            if kw in norm_text:
                score += 1
        for pat in patterns.get('strong_patterns', []):
            if re.search(pat, norm_text, re.IGNORECASE):
                score += 2
        if score > best_score:
            best_score = score
            best_type = doc_type
    if best_score >= 2:
        return best_type
    return None

class SklearnClassifier:
    """
    Document classifier using sklearn's TF-IDF and SVM
    """
    
    def __init__(self):
        # Try to load the excellence model first, fallback to standard model
        self.excellence_model_path = os.path.join(config.MODEL_DIR, "commercial_doc_classifier_excellence.pkl")
        self.model_path = os.path.join(config.MODEL_DIR, "sklearn_tfidf_svm.pkl")
        self.pipeline = None
        self.classes = config.DOCUMENT_CLASSES
        self.load_model()
    
    def load_model(self):
        """
        Load the model from disk if it exists - try excellence model first
        """
        # Try to load the excellence model first
        if os.path.exists(self.excellence_model_path):
            logger.info(f"Loading Excellence model from {self.excellence_model_path}")
            try:
                with open(self.excellence_model_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                logger.info("Excellence model loaded successfully! 🎉")
                return True
            except Exception as e:
                logger.error(f"Error loading excellence model: {str(e)}")
                logger.info("Falling back to standard model...")
        
        # Fallback to standard model
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            try:
                with open(self.model_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                logger.info("Standard model loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.pipeline = None
                return False
        else:
            logger.info("No existing model found, creating new pipeline")
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    use_idf=True,
                    ngram_range=(1, 2)
                )),
                ('classifier', SVC(
                    kernel='linear',
                    probability=True,
                    C=1.5,
                    class_weight='balanced'
                ))
            ])
            return False
    
    def save_model(self):
        """
        Save the model to disk
        """
        if self.pipeline is not None:
            logger.info(f"Saving model to {self.model_path}")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
            logger.info("Model saved successfully")
            return True
        else:
            logger.warning("No model to save")
            return False
    
    def train(self, X, y, test_size=0.2, optimize=False):
        """
        Train the model on the given data
        
        Args:
            X: List of document texts
            y: List of document class labels
            test_size: Proportion of data to use for testing
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training model on {len(X)} documents")
        
        # Count samples per class to check if stratification is possible
        class_counts = {}
        for label in y:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Check if all classes have enough samples for stratification
        min_samples_per_class = min(class_counts.values()) if class_counts else 0
        logger.info(f"Minimum samples per class: {min_samples_per_class}")
        
        # If we have too few samples in some classes, don't use stratification
        if min_samples_per_class < 2:
            logger.warning("Some classes have too few samples for stratification, using random split instead")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            # Use stratified split if we have enough samples
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        if optimize:
            logger.info("Performing hyperparameter optimization")
            
            param_grid = {
                'tfidf__max_features': [5000, 10000, 15000],
                'tfidf__min_df': [3, 5, 7],
                'tfidf__max_df': [0.7, 0.8, 0.9],
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['linear', 'rbf']
            }
            
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            self.pipeline = grid_search.best_estimator_
        else:
            # Train the model with default parameters
            self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Determine the actual classes in the test set (for accurate reporting)
        unique_classes = sorted(set(y_test).union(set(y_pred)))
        
        # Generate the classification report with only classes that appear in the data
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info(f"Model training complete. Test accuracy: {accuracy:.4f}")
        logger.info(f"Unique classes in test set: {unique_classes}")
        
        # Save the model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def predict(self, text):
        """
        Predict the class of a document
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with predicted class and confidence scores
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet")
        
        # Special case for invoice detection - check for key invoice terms
        # --- Multilingual and expanded indicators for all document types ---
        invoice_indicators = [
            # French
            'facture', 'montant', 'tva', 'ttc', 'échéance', 'client', 'date', 'montant ht', 'montant ttc', 'total tva', 'facture du mois', 'facture proforma', 'n° facture', 'numéro de facture',
            # English
            'invoice', 'payment', 'due', 'total', 'tax', 'subtotal', 'customer', 'amount', 'paid', 'invoice number', 'proforma invoice', 'vat', 'balance due',
            # Arabic
            'فاتورة', 'تاريخ', 'مبلغ', 'ضريبة', 'إجمالي', 'دفع', 'مستحق', 'عميل', 'فاتورة رقم', 'فاتورة الشهر', 'المبلغ الإجمالي', 'ضريبة القيمة المضافة',
            # Mixed/ocr
            'factura', 'factur', 'inv', 'فاتوره', 'فواتير', 'فكتورة', 'فاتورة#', 'invoice#', 'facture#', 'رقم الفاتورة', 'رقم الفاتوره',
        ]
        strong_invoice_patterns = [
            r'facture\s*[#:n°]', r'invoice\s*[#:n°]', r'فاتورة\s*رقم', r'رقم\s*الفاتورة', r'facture\s*du\s*mois', r'facture\s*n.*\s*\d+', r'invoice\s*n.*\s*\d+', r'فاتورة\s*الشهر', r'مبلغ\s*إجمالي', r'المبلغ\s*الإجمالي', r'ضريبة\s*القيمة\s*المضافة', r'montant\s*total\s*hors', r'total\s*tva', r'facture.*\d{4,}', r'invoice number', r'n[°o]?\s*facture', r'facture n[°o]?', r'فاتورة\s*الشهر', r'facture.*\d{4,}',
            r'proforma invoice', r'paid invoice', r'فاتورة مدفوعة', r'فاتورة ضريبية', r'فاتورة مبيعات', r'فاتورة مشتريات',
        ]
        definitive_invoice_markers = [
            'facture n', 'invoice #', 'فاتورة رقم', 'facture du mois', 'فاتورة الشهر', 'montant ttc', 'montant ht', 'facture proforma', 'ضريبة القيمة المضافة', 'n° facture', 'invoice number', 'facture n°', 'paid invoice', 'فاتورة مدفوعة',
        ]
        purchase_order_indicators = [
            # French
            'bon de commande', 'commande', 'fournisseur', 'acheteur', 'référence', 'modalité de paiement', 'délai de livraison', 'articles', 'conditions', 'numéro de commande',
            # English
            'purchase order', 'order', 'p.o.', 'po', 'supplier', 'buyer', 'reference', 'order number', 'payment terms', 'delivery date', 'item', 'items', 'quantity',
            # Arabic
            'أمر شراء', 'طلب شراء', 'رقم أمر الشراء', 'المورد', 'المشتري', 'رقم الطلب', 'شروط الدفع', 'تاريخ التسليم', 'كمية', 'مواد',
            # Mixed/ocr
            'b.c.', 'bc', 'p.o', 'p o', 'بون دي كوموند', 'امر شراء', 'امر توريد', 'رقم امر الشراء',
        ]
        strong_po_patterns = [
            r'bon\s+de\s+commande', r'purchase\s+order', r'p\.?o\.?\s*[#:n°]', r'b\.?c\.?\s*[#:n°]', r'commande\s*[#:n°]', r'order\s*[#:n°]', r'bon\s+de\s+commande\s+n[o°]', r'numéro\s*:\s*bc-', r'numéro\s*de\s*commande', r'pour\s+le\s+fournisseur.*pour\s+le\s+client', r'fournisseur\s*:.*client\s*:',
            r'أمر\s*شراء', r'طلب\s*شراء', r'رقم\s*أمر\s*الشراء',
        ]
        po_exclusion_patterns = [
            r'facture\s+acquitt[ée]', r'paid\s+invoice', r'reçu\s+de\s+paiement', r'payment\s+receipt', r'فاتورة', r'invoice',
        ]
        quote_indicators = [
            'devis', 'quotation', 'quote', 'offre de prix', 'proposition commerciale', 'عرض سعر', 'تسعيرة', 'عرض أسعار', 'quote #', 'devis n°', 'quotation number', 'numéro de devis', 'devis رقم',
        ]
        strong_quote_patterns = [
            r'devis\s*n[°o]?', r'quotation\s*[#:n°]?', r'quote\s*[#:n°]?', r'devis numéro', r'numéro de devis', r'offre de prix', r'proposition commerciale', r'عرض\s*سعر',
        ]
        delivery_note_indicators = [
            'bon de livraison', 'delivery note', 'b.l.', 'bl', 'numéro de bon de livraison', 'delivery note number', 'note de livraison', 'إشعار تسليم', 'إيصال تسليم', 'رقم إشعار التسليم', 'رقم إيصال التسليم',
        ]
        strong_delivery_patterns = [
            r'bon de livraison', r'delivery note', r'b\.l\.\s*n[°o]?', r'bl\s*n[°o]?', r'bon de livraison n[°o]?', r'delivery note n[°o]?', r'numéro de bon de livraison', r'إشعار\s*تسليم',
        ]
        receipt_indicators = [
            'reçu', 'receipt', 'reçu de paiement', 'payment receipt', 'numéro de reçu', 'reçu de caisse', 'إيصال', 'إيصال دفع', 'إيصال استلام', 'رقم الإيصال', 'سند قبض', 'سند صرف',
        ]
        strong_receipt_patterns = [
            r'reçu\s*n[°o]?', r'receipt\s*[#:n°]?', r'reçu de paiement', r'payment receipt', r'numéro de reçu', r'reçu de caisse', r'إيصال\s*دفع', r'إيصال\s*استلام',
        ]
        bank_statement_indicators = [
            'relevé bancaire', 'bank statement', 'relevé de compte', 'statement', 'relevé', 'bank', 'compte', 'رقم الحساب', 'كشف حساب', 'كشف بنكي', 'بيان حساب',
        ]
        strong_bank_patterns = [
            r'relevé bancaire', r'bank statement', r'relevé de compte', r'statement\s*[#:n°]?', r'relevé\s*n[°o]?', r'bank statement n[°o]?', r'كشف\s*حساب',
        ]
        expense_report_indicators = [
            'note de frais', 'expense report', 'remboursement', 'frais', 'rapport de dépenses', 'تقرير مصاريف', 'مصاريف', 'نفقات', 'expense', 'expenses', 'reimbursement',
        ]
        strong_expense_patterns = [
            r'note de frais', r'expense report', r'note de frais\s*n[°o]?', r'expense report\s*[#:n°]?', r'expense report n[°o]?', r'تقرير\s*مصاريف',
        ]
        payslip_indicators = [
            'bulletin de paie', 'payslip', 'fiche de paie', 'bulletin de salaire', 'fiche de paie n°', 'payslip number', 'كشف رواتب', 'قسيمة راتب', 'بيان رواتب', 'رقم قسيمة الراتب',
        ]
        strong_payslip_patterns = [
            r'bulletin de paie', r'payslip', r'fiche de paie', r'payslip\s*[#:n°]?', r'bulletin de salaire', r'fiche de paie n[°o]?', r'payslip n[°o]?', r'كشف\s*رواتب',
        ]
        
        text_lower = text.lower()
        
        # SPEED OPTIMIZATION: Get ML prediction first and skip expensive checks if confident
        prediction = self.pipeline.predict([text])[0]
        try:
            probabilities = self.pipeline.predict_proba([text])[0]
            confidence_scores = dict(zip(self.pipeline.classes_, probabilities))
        except (AttributeError, NotImplementedError):
            decision_values = self.pipeline.decision_function([text])[0]
            e_x = np.exp(decision_values - np.max(decision_values))
            probabilities = e_x / e_x.sum()
            confidence_scores = dict(zip(self.pipeline.classes_, probabilities))
        
        confidence = confidence_scores.get(prediction, 0.0)
        
        # If ML model is very confident (≥80%), skip expensive rule checks
        if confidence >= 0.8:
            logger.info(f"High confidence prediction ({confidence:.3f}), skipping rule checks")
            return {
                "prediction": prediction,
                "confidence": confidence,
                "confidence_scores": confidence_scores
            }
        
        # First check for definitive invoice markers - these take precedence over everything
        definitive_invoice_match = False
        for marker in definitive_invoice_markers:
            if marker in text_lower:
                definitive_invoice_match = True
                break
        
        # If these terms appear in the text, increase likelihood of invoice classification
        invoice_match = False
        for term in invoice_indicators:
            if term in text_lower:
                invoice_match = True
                break
        
        # Check for strong invoice patterns
        strong_invoice_match = False
        for pattern in strong_invoice_patterns:
            if re.search(pattern, text_lower):
                strong_invoice_match = True
                break
        
        # Check for purchase order match
        po_match = False
        for term in purchase_order_indicators:
            if term in text_lower:
                po_match = True
                break
        
        # Check for strong purchase order patterns
        strong_po_match = False
        for pattern in strong_po_patterns:
            if re.search(pattern, text_lower):
                strong_po_match = True
                break
                
        # Check for purchase order exclusion patterns
        po_exclusion = False
        for pattern in po_exclusion_patterns:
            if re.search(pattern, text_lower):
                po_exclusion = True
                break
        
        # Get prediction
        prediction = self.pipeline.predict([text])[0]
        
        # Try to get probabilities - handle different classifier types
        try:
            # Standard case - classifier supports predict_proba
            probabilities = self.pipeline.predict_proba([text])[0]
            confidence_scores = dict(zip(self.pipeline.classes_, probabilities))
        except (AttributeError, NotImplementedError):
            # Handle OneVsRestClassifier with LinearSVC which doesn't support predict_proba
            logger.info("Classifier doesn't support predict_proba, using decision function")
            try:
                # Create a simple confidence score based on the decision function
                decision_values = self.pipeline.decision_function([text])[0]
                
                # Convert decision values to pseudo-probabilities using softmax
                def softmax(x):
                    """Convert values to pseudo-probabilities"""
                    e_x = np.exp(x - np.max(x))
                    return e_x / e_x.sum()
                
                # If decision_values is 1D for binary classification, convert to 2D
                if len(decision_values.shape) == 1 and len(self.pipeline.classes_) > 2:
                    pseudo_probs = softmax(decision_values)
                elif len(decision_values.shape) == 1:
                    # Binary case
                    pos_prob = 1 / (1 + np.exp(-decision_values))
                    pseudo_probs = np.array([1 - pos_prob, pos_prob])
                else:
                    # Multi-class case
                    pseudo_probs = softmax(decision_values)
                
                confidence_scores = dict(zip(self.pipeline.classes_, pseudo_probs))
            except (AttributeError, NotImplementedError):
                # If all else fails, create a simple confidence for the predicted class
                logger.warning("Classifier doesn't support decision_function, using simple confidence")
                confidence_scores = {cls: 0.1 for cls in self.classes}
                confidence_scores[prediction] = 0.9
        
        # Check for a definitive invoice match first (this takes priority)
        if definitive_invoice_match and 'invoice' in confidence_scores:
            # This is definitely an invoice - strongly boost its confidence
            confidence_scores['invoice'] = max(confidence_scores.get('invoice', 0) * 4.0, 0.6)
            
            # Strongly reduce purchase order confidence
            if 'purchase_order' in confidence_scores:
                confidence_scores['purchase_order'] = confidence_scores.get('purchase_order', 0) * 0.3
                
            logger.info("Definitive invoice marker detected - strongly boosting invoice confidence")
        
        # Apply purchase order boosting only if we don't have a definitive invoice match
        elif po_match and 'purchase_order' in confidence_scores and not po_exclusion and not definitive_invoice_match:
            # Stronger boost factors for purchase orders to prioritize over invoices
            boost_factor = 2.0  # Higher initial boost than invoice
            if strong_po_match:
                boost_factor = 3.5  # Much stronger boost for definite PO patterns
                
            # Boost purchase order confidence with higher minimum
            confidence_scores['purchase_order'] = max(confidence_scores.get('purchase_order', 0) * boost_factor, 0.35)
            
            # If we have a strong purchase order pattern, reduce invoice confidence
            if strong_po_match and 'invoice' in confidence_scores:
                confidence_scores['invoice'] = confidence_scores.get('invoice', 0) * 0.7
        
        # Apply invoice boosting if no definitive match was already applied
        elif invoice_match and 'invoice' in confidence_scores and not strong_po_match and not definitive_invoice_match:
            boost_factor = 1.8  # Increased from 1.5
            if strong_invoice_match:
                boost_factor = 3.0  # Increased from 2.5
                
            # Boost invoice confidence but ensure it's at least 0.35 (increased from 0.25)
            confidence_scores['invoice'] = max(confidence_scores.get('invoice', 0) * boost_factor, 0.35)
            
            # If we have a strong invoice match, reduce purchase order confidence
            if strong_invoice_match and 'purchase_order' in confidence_scores:
                confidence_scores['purchase_order'] = confidence_scores.get('purchase_order', 0) * 0.5  # Increased reduction
        
        # Special handling for exact match of "Bon de commande" at start
        if text_lower.startswith("bon de commande") and 'purchase_order' in confidence_scores:
            confidence_scores['purchase_order'] = max(confidence_scores.get('purchase_order', 0), 0.50)
            if 'invoice' in confidence_scores:
                confidence_scores['invoice'] = confidence_scores.get('invoice', 0) * 0.6
        
        # Normalize all confidence scores to ensure they sum to 1
        total = sum(confidence_scores.values())
        for key in confidence_scores:
            confidence_scores[key] = confidence_scores[key] / total
        
        # Sort by confidence (highest first)
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Check if confidence is high enough
        if sorted_scores[0][1] < config.CONFIDENCE_THRESHOLD:
            prediction = "unknown"
        else:
            # Use the class with highest confidence after our adjustments
            prediction = sorted_scores[0][0]
        
        # Only keep supported document types in confidence_scores
        SUPPORTED_TYPES = [
            "invoice", "quote", "purchase_order", "delivery_note", "receipt", "bank_statement", "expense_report", "payslip"
        ]
        confidence_scores = {k: v for k, v in confidence_scores.items() if k in SUPPORTED_TYPES}
        # Re-normalize if needed
        total = sum(confidence_scores.values())
        if total > 0:
            for k in confidence_scores:
                confidence_scores[k] = confidence_scores[k] / total
        else:
            confidence_scores = {k: 1.0 / len(SUPPORTED_TYPES) for k in SUPPORTED_TYPES}
        # Update prediction if not in supported types
        if prediction not in SUPPORTED_TYPES:
            prediction = max(confidence_scores, key=confidence_scores.get)

        # --- Strong type overrides: force prediction if definitive marker is present for any supported type (robust matching) ---
        definitive_type_markers = {
            'invoice': [
                r'facture\s*[#:n°]', r'invoice\s*[#:n°]', r'فاتورة\s*رقم', r'facture du mois', r'فاتورة الشهر',
                r'montant\s*ttc', r'montant\s*ht', r'facture proforma', r'ضريبة القيمة المضافة',
                r'n[°o]?\s*facture', r'invoice number', r'facture n[°o]?', r'فاتورة\s*الشهر', r'facture.*\d{4,}'
            ],
            'purchase_order': [
                r'bon\s+de\s+commande\s*[#:n°]?', r'purchase order\s*[#:n°]?', r'\bp\.?o\.?\s*[#:n°]', r'bc\s*n[°o]?', r'b\.c\.\s*n[°o]?',
                r'numéro\s*de\s*commande', r'order number', r'bon de commande', r'purchase order',
                r'bon de commande n[°o]?', r'purchase order n[°o]?'
            ],
            'quote': [
                r'devis\s*n[°o]?', r'quotation\s*[#:n°]?', r'quote\s*[#:n°]?', r'devis numéro', r'numéro de devis',
                r'offre de prix', r'proposition commerciale'
            ],
            'delivery_note': [
                r'bon de livraison', r'delivery note', r'b\.l\.\s*n[°o]?', r'bl\s*n[°o]?', r'bon de livraison n[°o]?',
                r'delivery note n[°o]?', r'numéro de bon de livraison'
            ],
            'receipt': [
                r'reçu\s*n[°o]?', r'receipt\s*[#:n°]?', r'reçu de paiement', r'payment receipt',
                r'numéro de reçu', r'reçu de caisse'
            ],
            'bank_statement': [
                r'relevé bancaire', r'bank statement', r'relevé de compte bancaire', r'releve de compte bancaire',
                r'statement\s*[#:n°]?', r'relevé\s*n[°o]?', r'bank statement n[°o]?', r'attijariwafa bank',
                r'releve d.identite bancaire', r'relevé d.identité bancaire', r'total mouvements',
                r'solde depart', r'solde final', r'crediteur', r'devise\s*:\s*dirham'
            ],
            'expense_report': [
                r'note de frais', r'expense report', r'note de frais\s*n[°o]?', r'expense report\s*[#:n°]?',
                r'expense report n[°o]?'
            ],
            'payslip': [
                r'bulletin de paie', r'payslip', r'fiche de paie', r'payslip\s*[#:n°]?', r'bulletin de salaire',
                r'fiche de paie n[°o]?', r'payslip n[°o]?'
            ]
        }
        for doc_type, patterns in definitive_type_markers.items():
            if doc_type in confidence_scores:
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        logger.info(f"Strong {doc_type} override: definitive marker pattern '{pattern}' present, forcing {doc_type} prediction.")
                        prediction = doc_type
                        confidence = confidence_scores[doc_type]
                        break
                if prediction == doc_type:
                    break

        # --- Ambiguity handling: if both invoice and purchase order are strong and close, flag as ambiguous ---
        if (
            'invoice' in confidence_scores and 'purchase_order' in confidence_scores and
            abs(confidence_scores['invoice'] - confidence_scores['purchase_order']) < 0.1 and
            (invoice_match or strong_invoice_match) and (po_match or strong_po_match)
        ):
            logger.info("Ambiguous document: both invoice and purchase order indicators present and close confidence.")
            prediction = 'unknown'
            # Optionally, set both confidences to 0.5 or keep as is
            confidence_scores['invoice'] = confidence_scores['purchase_order'] = 0.5
            # Re-normalize other classes
            total = sum(confidence_scores.values())
            for k in confidence_scores:
                confidence_scores[k] = confidence_scores[k] / total

        # Always set confidence to match the predicted class's score
        confidence = confidence_scores.get(prediction, max(confidence_scores.values()))

        return {
            'prediction': prediction,
            'confidence': confidence,
            'confidence_scores': confidence_scores
        }

    def hybrid_predict(self, text, document_id=None, text_excerpt=None):
        model_result = self.predict(text)
        model_prediction = model_result['prediction']
        model_confidence = model_result['confidence']
        confidence_scores = model_result['confidence_scores']
        rule_based_prediction = rule_based_classification(text)
        if model_confidence >= 0.7 and model_prediction != 'unknown':
            final_prediction = model_prediction
            confidence_flag = 'ok'
        elif rule_based_prediction:
            final_prediction = rule_based_prediction
            confidence_flag = 'needs_review'
        else:
            final_prediction = model_prediction
            confidence_flag = 'needs_review'
        return {
            'document_id': document_id or '',
            'model_prediction': model_prediction,
            'model_confidence': model_confidence,
            'rule_based_prediction': rule_based_prediction,
            'final_prediction': final_prediction,
            'confidence_flag': confidence_flag,
            'confidence_scores': confidence_scores,
            'text_excerpt': text_excerpt or (text[:500] if text else '')
        }