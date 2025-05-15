# Document Classification Service - Usage Guide

This guide explains how to use the document classification service.

## Prerequisites

1. Python 3.7+ installed
2. Required Python packages installed via `pip install -r requirements.txt`
3. Tesseract OCR installed on your system (see README.md for installation instructions)

## Training a Model

Before you can classify documents, you need to train a model. The service includes sample data generation to get you started:

```bash
# Train the default model (sklearn_tfidf_svm)
python app.py train

# Train with a specific model type
python app.py train --model bert

# Train with hyperparameter optimization (takes longer)
python app.py train --model sklearn_tfidf_svm --optimize
```

## Classifying a Document

You can classify a single document using the CLI:

```bash
# Classify a document using the default model
python app.py classify path/to/your/document.pdf

# Classify with a specific model
python app.py classify path/to/your/document.pdf --model bert
```

## Running the API Server

For production use, you can start the API server:

```bash
# Start the API server
python app.py api
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Basic health check
- `GET /health`: Service health check
- `GET /models`: List available models and document classes
- `POST /classify`: Classify a document

### Using the Classification API

To classify a document via the API, send a POST request to `/classify`:

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf" \
  -F "model_type=sklearn_tfidf_svm"
```

Response format:

```json
{
  "document_id": "uuid-string",
  "prediction": "invoice",
  "confidence": 0.92,
  "confidence_scores": {
    "invoice": 0.92,
    "contract": 0.05,
    "id_card": 0.01,
    ...
  },
  "text_excerpt": "INVOICE #: INV-1234\nDate: 7/15/2023...",
  "processing_time_ms": 1234.56
}
```

## Using Your Own Training Data

To use your own training data:

1. Create a CSV file with the following columns:
   - `text`: The document text
   - `label`: The document classification label

2. Pass the path to your data file:

```python
from utils import load_sample_data
from models import get_model

# Load your own data
X, y = load_sample_data("path/to/your/data.csv")

# Get model and train
model = get_model("sklearn_tfidf_svm")
model.train(X, y)
```

## Working with LayoutLM

For document classification that incorporates layout information (LayoutLM):

1. You need to extract both text and bounding box information from documents
2. Use the LayoutLM model for training and prediction
3. Provide document data with appropriate format (see LayoutLM documentation)

## Customization

To customize the service:

1. Edit `config.py` to change configurations
2. Modify the `DOCUMENT_CLASSES` list to match your classification needs
3. Add new model types by extending the model factories
4. Enhance preprocessing for your specific document types 