import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create model directory if it doesn't exist
os.makedirs('data/models', exist_ok=True)

# Configuration
MODEL_FILE = 'data/models/commercial_doc_classifier_enhanced.pkl'
TRAINING_DATA = 'data/samples/complete_training_data.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def preprocess_text(text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def train_models():
    """Train the document classification models"""
    print("Loading training data...")
    df = pd.read_csv(TRAINING_DATA)
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count} samples")
    
    # Apply preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Create and train multiple models
    models = {
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=2
            )),
            ('classifier', OneVsRestClassifier(LinearSVC(random_state=RANDOM_STATE)))
        ]),
        
        'RandomForest': Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=2
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=200, 
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        
        'LogisticRegression': Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                min_df=2
            )),
            ('classifier', LogisticRegression(
                C=10,
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Generate report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred,
            'true_values': y_test
        }
        
        # Print purchase order metrics
        if 'purchase_order' in report:
            print(f"Purchase Order - Precision: {report['purchase_order']['precision']:.4f}, "
                  f"Recall: {report['purchase_order']['recall']:.4f}, "
                  f"F1: {report['purchase_order']['f1-score']:.4f}")
    
    # Find the best model based on accuracy
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    # Save the best model
    print(f"Saving {best_model_name} model to {MODEL_FILE}...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Plot confusion matrix for the best model
    plot_confusion_matrix(
        results[best_model_name]['true_values'],
        results[best_model_name]['predictions'],
        best_model_name
    )
    
    return best_model, best_model_name, results

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix for model evaluation"""
    plt.figure(figsize=(12, 10))
    
    # Get unique labels
    labels = sorted(set(list(y_true) + list(y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'data/models/confusion_matrix_{model_name}.png')
    print(f"Saved confusion matrix to data/models/confusion_matrix_{model_name}.png")

def test_purchase_order_examples():
    """Test the model specifically on purchase order examples"""
    # Load the model
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    
    # Create some test examples
    purchase_order_examples = [
        "BON DE COMMANDE\nNuméro: PO-5432\nDate: 15/04/2023\nFournisseur: Martin SARL\nArticles: 5 x Laptop Dell\nTotal: 4500.00€",
        "Bon de commande\nCommande N°: 123\nDate: 03/04/2020\nClient: Nom\nTotal: 550€",
        "PURCHASE ORDER\nPO Number: PO-9876\nDate: 05/20/2023\nVendor: Office Supplies Inc.\nItems: 20 x Office chairs\nTotal: $2,400.00",
        "BON DE COMMANDE\nRéférence: BC-7890\nDate: 10/06/2023\nFournisseur: Tech Solutions\nArticles: 5 x Ordinateurs portables\nPrix unitaire: 800.00€\nTotal: 4000.00€",
    ]
    
    # Preprocess and predict
    processed_examples = [preprocess_text(example) for example in purchase_order_examples]
    predictions = model.predict(processed_examples)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(processed_examples)
        classes = model.classes_
    else:
        probas = None
        classes = None
    
    # Print results
    print("\nPurchase Order Test Results:")
    for i, (example, prediction) in enumerate(zip(purchase_order_examples, predictions)):
        print(f"\nExample {i+1}:")
        print("-" * 40)
        print(example[:100] + "..." if len(example) > 100 else example)
        print("-" * 40)
        print(f"Prediction: {prediction}")
        
        if probas is not None:
            top_indices = np.argsort(probas[i])[::-1][:3]  # Top 3 predictions
            print("Top 3 predictions:")
            for idx in top_indices:
                print(f"  {classes[idx]}: {probas[i][idx]:.4f}")

def main():
    """Main function to train and evaluate models"""
    print("=== Training Enhanced Document Classification Model ===\n")
    
    # Train models
    best_model, best_model_name, results = train_models()
    
    # Test purchase order examples
    test_purchase_order_examples()
    
    print("\n=== Model Training Complete ===")

if __name__ == "__main__":
    main() 