# 🚀 Document Classification API

A production-ready document classification system using machine learning to automatically classify business documents with **95.5% accuracy**. Supports multiple languages and includes financial analysis capabilities powered by Groq AI.

---

## ⚡️ Quick Start Guide

### Prerequisites

- Python 3.10 or higher
- Tesseract OCR (for text extraction from images/PDFs)
- Git

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd document-classification-api
```

### 2. System Dependencies

#### Windows

```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH or set TESSERACT_PATH in config.py
```

#### Linux/Ubuntu

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-ara
sudo apt-get install libsm6 libxext6 libxrender-dev libglib2.0-0
```

#### macOS

```bash
brew install tesseract
```

### 3. Python Environment Setup

```bash
# Create virtual environment
python -m venv ai_model_env

# Activate virtual environment
# Windows:
ai_model_env\Scripts\activate
# Linux/Mac:
source ai_model_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Required for document upload and storage
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Required for AI-powered document analysis
GROQ_API_KEY=your_groq_api_key
```

**Get your API keys:**

- **Cloudinary**: Sign up at [cloudinary.com](https://cloudinary.com) for free document storage
- **Groq**: Get API key at [console.groq.com](https://console.groq.com) for AI analysis

### 5. Initial Setup & Model Training

```bash
# Train the classification model (first time setup)
python app.py train --enhanced

# Or train with optimization (takes longer but better accuracy)
python app.py train --enhanced --optimize
```

### 6. Start the API Server

```bash
python app.py api
```

The API will be available at: [http://localhost:8000](http://localhost:8000)

---

## 🚀 Docker Deployment (Recommended)

### Quick Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t doc-classifier .
docker run -p 8000:8000 --env-file .env doc-classifier
```

The containerized API includes all system dependencies and is ready to use.

---

## 📋 API Usage Examples

### 1. Document Classification

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

**Response:**

```json
{
  "document_id": "unique-document-id",
  "model_prediction": "invoice",
  "model_confidence": 0.85,
  "final_prediction": "invoice",
  "confidence_flag": "high",
  "confidence_scores": {
    "invoice": 0.85,
    "receipt": 0.1,
    "purchase_order": 0.05
  },
  "text_excerpt": "INVOICE #001...",
  "processing_time_ms": 1250,
  "cloudinary_url": "https://res.cloudinary.com/...",
  "extracted_info": {
    "date": "2025-01-15",
    "client_name": "John Doe",
    "amount": "$150.00"
  }
}
```

### 2. Financial Analysis (Groq AI)

```bash
curl -X POST "http://localhost:8000/analyze-financial" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@invoice.pdf"
```

### 3. Generate Financial Report

```bash
curl -X POST "http://localhost:8000/generate-bilan" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@invoice1.pdf" \
  -F "files=@receipt1.pdf" \
  -F "files=@bank_statement.pdf"
```

### 4. Health Check

```bash
curl http://localhost:8000/health
```

### 5. Available Models

```bash
curl http://localhost:8000/models
```

---

## 🔧 Configuration & Customization

### Key Configuration (config.py)

```python
# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Model Settings
DEFAULT_MODEL = "sklearn_tfidf_svm"
CONFIDENCE_THRESHOLD = 0.10

# OCR Settings
TESSERACT_PATH = "tesseract"  # System PATH
```

### Command Line Usage

```bash
# Train different models
python app.py train --model sklearn_tfidf_svm
python app.py train --model bert
python app.py train --enhanced --optimize

# Classify single document
python app.py classify document.pdf --model sklearn_tfidf_svm

# Start API server
python app.py api
```

### Available Scripts

- `auto_dataset_processor.py` - Process and augment training data
- `clean_training_data.py` - Clean and validate training dataset
- `check_data_quality.py` - Analyze dataset quality and distribution
- `visualize_dataset_confidence.py` - Generate confidence analysis charts

---

## 📈 Model Performance & Features

### Architecture

- **Hybrid Classification**: Rule-based + ML model ensemble
- **Advanced Features**: 20,000 TF-IDF features with n-grams
- **Multi-language Support**: English, French, Arabic
- **Confidence Calibration**: Realistic confidence scoring
- **Financial Analysis**: Groq AI-powered extraction

### Supported Document Types

| Document Type       | Description                | Languages |
| ------------------- | -------------------------- | --------- |
| **Invoice**         | Bills and invoices         | 🇬🇧 🇫🇷 🇸🇦  |
| **Receipt**         | Payment receipts           | 🇬🇧 🇫🇷 🇸🇦  |
| **Quote**           | Price quotes and estimates | 🇬🇧 🇫🇷     |
| **Purchase Order**  | Procurement documents      | 🇬🇧 🇫🇷     |
| **Delivery Note**   | Shipping documents         | 🇬🇧 🇫🇷     |
| **Bank Statement**  | Account statements         | 🇬🇧 🇫🇷     |
| **Expense Report**  | Reimbursement forms        | 🇬🇧 🇫🇷     |
| **Payslip**         | Salary statements          | 🇬🇧 🇫🇷     |
| **Credit Note**     | Credit memos               | 🇬🇧 🇫🇷     |
| **Tax Declaration** | Tax documents              | 🇬🇧 🇫🇷     |

### Performance Metrics

- **Accuracy**: 95.5% on test dataset
- **Processing Speed**: ~1.2s per document
- **Confidence Calibration**: Realistic probability scores
- **Multi-format Support**: PDF, PNG, JPG, TIFF

---

## 🏗️ Project Structure

```
document-classification-api/
├── api/                    # FastAPI application
│   ├── app.py             # Main API endpoints
│   ├── config.py          # API configuration
│   └── integrations/      # External service integrations
├── models/                # ML model implementations
├── preprocessor/          # Text preprocessing utilities
├── utils/                 # Utility functions
│   ├── cloudinary_utils.py    # Document upload
│   ├── text_extraction.py    # OCR and text extraction
│   ├── groq_utils.py         # AI-powered analysis
│   └── financial_analyzer.py # Financial document analysis
├── data/
│   ├── models/           # Trained model files
│   ├── samples/          # Training datasets
│   └── temp/            # Temporary processing files
├── app.py               # CLI interface
├── config.py           # Global configuration
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
└── docker-compose.yml # Multi-service deployment
```

---

## 🚀 Production Deployment

### Docker Production (Recommended)

```bash
# Production docker-compose
version: '3'
services:
  doc-classifier:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CLOUDINARY_CLOUD_NAME=${CLOUDINARY_CLOUD_NAME}
      - CLOUDINARY_API_KEY=${CLOUDINARY_API_KEY}
      - CLOUDINARY_API_SECRET=${CLOUDINARY_API_SECRET}
      - GROQ_API_KEY=${GROQ_API_KEY}
    restart: unless-stopped
    volumes:
      - ./data:/app/data
```

### Manual Production Deployment

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:8000

# Or with custom configuration
gunicorn -c gunicorn.conf.py api.app:app
```

### Environment Variables (Production)

```bash
export CLOUDINARY_CLOUD_NAME="your_cloud_name"
export CLOUDINARY_API_KEY="your_api_key"
export CLOUDINARY_API_SECRET="your_api_secret"
export GROQ_API_KEY="your_groq_api_key"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

---

## 🔧 Development & Training

### Dataset Management

```bash
# Download additional training data
python download_cuad_dataset.py
python download_company_documents.py
python download_sroie_kaggle.py

# Process and clean training data
python auto_dataset_processor.py
python clean_training_data.py

# Analyze data quality
python check_data_quality.py
python visualize_dataset_confidence.py
```

### Model Training Options

```bash
# Basic training
python app.py train

# Enhanced training with expanded dataset
python app.py train --enhanced

# Hyperparameter optimization (slower but better)
python app.py train --enhanced --optimize

# Train specific model type
python app.py train --model bert
python app.py train --model layoutlm
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Tesseract not found**

```bash
# Windows: Download and install from GitHub
# Linux: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

**2. Model files missing**

```bash
python app.py train --enhanced
```

**3. API key errors**

```bash
# Check .env file exists and has correct keys
cat .env
```

**4. Memory issues during training**

```bash
# Use smaller dataset or reduce batch size
python app.py train  # without --enhanced flag
```

### Health Checks

```bash
# Test API health
curl http://localhost:8000/health

# Test model loading
curl http://localhost:8000/models

# Test classification
curl -X POST "http://localhost:8000/classify" -F "file=@test.pdf"
```

---

## 📞 Support & Contributing

### Getting Help

- Check the logs for detailed error information
- Ensure all environment variables are set correctly
- Verify model files are present in `data/models/`
- Test with the `/health` endpoint first

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Performance Monitoring

- Monitor `/health` endpoint for uptime
- Check processing times in API responses
- Review confidence scores for model accuracy

---

## 📄 License

MIT License - see LICENSE file for details
