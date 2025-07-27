# 🚀 Document Classification API

A production-ready document classification system using machine learning to automatically classify business documents with **95.5% accuracy**.

---

## ⚡️ Step-by-Step Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd document-classification-api
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv ai_model_env
ai_model_env\Scripts\activate  # Windows
# or
source ai_model_env/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following content:
```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
GROQ_API_KEY=your_groq_api_key
```

- These are required for document upload and AI extraction.
- You can also set these variables directly in your shell or deployment environment.

### 5. Run the FastAPI Server
```bash
python app.py api
```
- The API will be available at: [http://localhost:8000](http://localhost:8000)

---

## 🏃‍♂️ FastAPI Example Usage

### Classify a Document (using `curl`)
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

#### Example Response
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

---

## 🔧 Configuration

Key configuration options in `config.py`:

- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions
- `DOCUMENT_CLASSES`: Supported document types

---

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