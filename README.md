# 🚀 Document Classification API

A production-ready document classification system using machine learning to automatically classify business documents with **95.5% accuracy**.

## 📋 Features

- **High Accuracy**: 95.5% overall accuracy with Excellence Model
- **9 Document Types**: Invoice, Receipt, Contract, Purchase Order, Quote, Bank Statement, Expense Report, Payslip, Delivery Note
- **Real-time API**: Fast classification with confidence scores
- **Cloud Storage**: Automatic document upload to Cloudinary
- **AI Information Extraction**: Powered by Groq AI
- **Production Ready**: Docker support, comprehensive error handling

## 🎯 Performance

- **Excellent (100% accuracy)**: 8 document types
- **Good (50-99% accuracy)**: 1 document type  
- **Average Confidence**: 66.2%
- **Processing Time**: ~2-3 seconds per document

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd document-classification-api
```

2. **Create virtual environment**
```bash
python -m venv ai_model_env
ai_model_env\Scripts\activate  # Windows
# or
source ai_model_env/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file:
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
GROQ_API_KEY=your_groq_api_key
```

5. **Start the API**
```bash
python app.py api
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### 1. Classify Document
```bash
POST /classify

curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### 2. Classify Commercial Document
```bash
POST /classify-commercial

curl -X POST "http://localhost:8000/classify-commercial" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### 3. Health Check
```bash
GET /health
```

### 4. List Models
```bash
GET /models
```

## 📊 Response Format

```json
{
  "document_id": "unique-document-id",
  "model_prediction": "invoice",
  "model_confidence": 0.85,
  "final_prediction": "invoice",
  "confidence_flag": "high",
  "confidence_scores": {
    "invoice": 0.85,
    "receipt": 0.10,
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

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Production Deployment
```bash
docker-compose up -d
```

## 📁 Project Structure

```
document-classification-api/
├── app.py                 # Main application entry point
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── api/                 # FastAPI application
│   └── app.py          # API routes and logic
├── models/              # ML models
│   └── sklearn_model.py # Main classification model
├── utils/               # Utility functions
│   ├── text_extraction.py
│   ├── cloudinary_utils.py
│   └── document_analyzer.py
├── preprocessor/        # Text preprocessing
│   └── text_processor.py
└── data/
    └── models/         # Trained model files
        └── commercial_doc_classifier_excellence.pkl
```

## 🔧 Configuration

Key configuration options in `config.py`:

- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions
- `DOCUMENT_CLASSES`: Supported document types

## 📈 Model Performance

The system uses an **Excellence Model** with:
- **Ensemble Architecture**: SVM + Random Forest + Naive Bayes
- **Advanced Features**: 20,000 TF-IDF features with 4-grams
- **Rule-based Fallbacks**: Keyword detection for edge cases
- **Confidence Calibration**: Realistic confidence scores

### Supported Document Types
1. **Invoice** - Bills and invoices
2. **Receipt** - Payment receipts and tickets  
3. **Contract** - Employment and service contracts
4. **Purchase Order** - Purchase orders and procurement
5. **Quote** - Price quotes and estimates
6. **Bank Statement** - Bank account statements
7. **Expense Report** - Expense and reimbursement reports
8. **Payslip** - Salary and wage statements
9. **Delivery Note** - Shipping and delivery documents

## 🚀 Production Deployment

### Environment Variables
Set these in your production environment:
- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY` 
- `CLOUDINARY_API_SECRET`
- `GROQ_API_KEY`

### Example Production Command
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:8000
```

## 📞 Support

For issues and questions:
- Check the logs for detailed error information
- Ensure all environment variables are set
- Verify model files are present in `data/models/`
- Test with the `/health` endpoint

## 📄 License

MIT License