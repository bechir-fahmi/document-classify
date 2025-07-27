# 📊 Financial Document Analysis & Reporting API

## Overview
This API provides AI-powered financial document analysis and reporting capabilities using advanced machine learning models and Groq AI for accurate data extraction.

## 🚀 Key Features
- **Document Classification**: Automatically identifies document types (invoices, quotes, purchase orders, etc.)
- **Financial Data Extraction**: Extracts amounts, currencies, dates, and metadata using Groq AI
- **Document Embeddings**: Generates semantic embeddings for document similarity and search
- **Financial Reporting**: Creates comprehensive financial reports (bilan) from multiple documents
- **Multi-language Support**: Handles English, French, and Arabic documents
- **High Accuracy**: Uses hybrid AI approach for maximum precision

---

## 📋 API Endpoints

### 1. Document Classification
**Endpoint**: `POST /classify`
**Description**: Classifies a document and extracts basic information

```javascript
// Frontend Usage (Next.js)
const classifyDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/classify', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

**Response Example**:
```json
{
  "document_id": "documents/abc123.pdf",
  "model_prediction": "invoice",
  "model_confidence": 0.94,
  "rule_based_prediction": "invoice",
  "final_prediction": "invoice",
  "confidence_flag": "high",
  "confidence_scores": {
    "invoice": 0.94,
    "quote": 0.03,
    "purchase_order": 0.02
  },
  "text_excerpt": "INVOICE #INV-001...",
  "processing_time_ms": 1250.5,
  "cloudinary_url": "https://res.cloudinary.com/...",
  "public_id": "documents/abc123",
  "extracted_info": {...}
}
```

### 2. Groq-Powered Financial Analysis
**Endpoint**: `POST /analyze-financial-groq`
**Description**: Advanced financial analysis using Groq AI for maximum accuracy

```javascript
// Frontend Usage (Next.js)
const analyzeFinancialDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/analyze-financial-groq', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

**Response Example**:
```json
{
  "document_id": "abc123-def456",
  "groq_financial_analysis": {
    "document_type": "invoice",
    "amount": 51.94,
    "currency": "USD",
    "date": "2024-01-15T00:00:00",
    "description": "Consulting Services and Development Work",
    "category": "income",
    "subcategory": "sales_revenue",
    "confidence": 0.95,
    "raw_groq_response": {
      "line_items": [
        {"item": "Consulting", "amount": 25.00},
        {"item": "Development", "amount": 20.00}
      ],
      "tax_amount": 4.50,
      "subtotal": 45.00,
      "payment_terms": "Net 30 days",
      "vendor_customer": "ABC Company"
    }
  },
  "document_classification": {
    "rule_based_prediction": "invoice",
    "model_prediction": "invoice",
    "model_confidence": 0.94,
    "final_prediction": "invoice"
  },
  "document_embedding": [0.123, -0.456, ...], // 384-dimensional vector
  "embedding_model": "all-MiniLM-L6-v2",
  "processing_time_ms": 2150.3,
  "extraction_method": "groq_ai"
}
```

### 3. Financial Bilan Generation
**Endpoint**: `POST /generate-bilan-groq`
**Description**: Generate comprehensive financial reports from multiple documents

```javascript
// Frontend Usage (Next.js)
const generateFinancialBilan = async (files, periodDays = 30) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  const response = await fetch(`/api/generate-bilan-groq?period_days=${periodDays}`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

**Response Example**:
```json
{
  "period": {
    "days": 30,
    "start_date": "2024-06-25T00:00:00",
    "end_date": "2024-07-25T00:00:00"
  },
  "summary": {
    "total_income": 5240.50,
    "total_expenses": 2180.75,
    "potential_income": 1500.00,
    "net_result": 3059.75,
    "profit_margin_percent": 58.4
  },
  "currency_breakdown": {
    "USD": {
      "income": 3240.50,
      "expense": 1180.75,
      "potential": 1000.00,
      "net": 2059.75
    },
    "EUR": {
      "income": 2000.00,
      "expense": 1000.00,
      "potential": 500.00,
      "net": 1000.00
    }
  },
  "document_analysis": {
    "invoice": {
      "count": 8,
      "total_amount": 5240.50,
      "average_amount": 655.06
    },
    "purchase_order": {
      "count": 5,
      "total_amount": 2180.75,
      "average_amount": 436.15
    },
    "quote": {
      "count": 3,
      "total_amount": 1500.00,
      "average_amount": 500.00
    }
  },
  "transaction_count": 16,
  "recommendations": [
    "💰 Strong cash flow. Consider investment opportunities.",
    "📈 High potential income from quotes. Focus on conversion.",
    "💱 Multiple currencies detected. Consider currency risk management."
  ],
  "generated_at": "2024-07-25T17:30:00"
}
```

### 4. API Information Endpoints

**Financial Summary**: `GET /financial-summary`
```json
{
  "supported_document_types": ["invoice", "quote", "purchase_order", "receipt", "bank_statement", "expense_report", "payslip", "delivery_note"],
  "supported_currencies": ["EUR", "USD", "TND", "GBP"],
  "analysis_features": ["Amount extraction", "Currency detection", "Date extraction", "Transaction categorization"],
  "bilan_metrics": ["Total income", "Total expenses", "Net result", "Profit margin", "Currency breakdown"]
}
```

**Embedding Info**: `GET /embedding-info`
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "description": "Document embeddings using sentence-transformers"
}
```

---

## 🎨 Frontend Integration Examples

### React/Next.js Component Example

```jsx
import { useState } from 'react';

const FinancialAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    
    if (selectedFile) {
      setLoading(true);
      try {
        const result = await analyzeFinancialDocument(selectedFile);
        setAnalysis(result);
      } catch (error) {
        console.error('Analysis failed:', error);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="financial-analyzer">
      <input 
        type="file" 
        accept=".pdf,.png,.jpg,.jpeg,.txt"
        onChange={handleFileUpload}
      />
      
      {loading && <div>Analyzing document...</div>}
      
      {analysis && (
        <div className="analysis-results">
          <h3>Financial Analysis Results</h3>
          <div className="summary">
            <p><strong>Document Type:</strong> {analysis.groq_financial_analysis.document_type}</p>
            <p><strong>Amount:</strong> {analysis.groq_financial_analysis.amount} {analysis.groq_financial_analysis.currency}</p>
            <p><strong>Category:</strong> {analysis.groq_financial_analysis.category}</p>
            <p><strong>Confidence:</strong> {(analysis.groq_financial_analysis.confidence * 100).toFixed(1)}%</p>
          </div>
          
          {analysis.groq_financial_analysis.raw_groq_response.line_items && (
            <div className="line-items">
              <h4>Line Items:</h4>
              {analysis.groq_financial_analysis.raw_groq_response.line_items.map((item, index) => (
                <div key={index}>
                  {item.item}: ${item.amount}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
```

### Bilan Dashboard Component

```jsx
const BilanDashboard = () => {
  const [files, setFiles] = useState([]);
  const [bilan, setBilan] = useState(null);
  const [period, setPeriod] = useState(30);

  const generateBilan = async () => {
    if (files.length === 0) return;
    
    const result = await generateFinancialBilan(files, period);
    setBilan(result);
  };

  return (
    <div className="bilan-dashboard">
      <div className="controls">
        <input 
          type="file" 
          multiple 
          onChange={(e) => setFiles(Array.from(e.target.files))}
        />
        <select value={period} onChange={(e) => setPeriod(e.target.value)}>
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
        <button onClick={generateBilan}>Generate Bilan</button>
      </div>

      {bilan && (
        <div className="bilan-results">
          <div className="summary-cards">
            <div className="card income">
              <h3>Total Income</h3>
              <p>${bilan.summary.total_income}</p>
            </div>
            <div className="card expenses">
              <h3>Total Expenses</h3>
              <p>${bilan.summary.total_expenses}</p>
            </div>
            <div className="card profit">
              <h3>Net Result</h3>
              <p>${bilan.summary.net_result}</p>
            </div>
            <div className="card margin">
              <h3>Profit Margin</h3>
              <p>{bilan.summary.profit_margin_percent}%</p>
            </div>
          </div>

          <div className="recommendations">
            <h3>Recommendations</h3>
            {bilan.recommendations.map((rec, index) => (
              <div key={index} className="recommendation">{rec}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
```

---

## 🔧 Backend Integration (Next.js API Routes)

### API Route Example: `/pages/api/analyze-financial.js`

```javascript
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const formData = new FormData();
    // Handle file upload from frontend
    formData.append('file', req.body.file);

    const response = await fetch('http://localhost:8000/analyze-financial-groq', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    
    if (response.ok) {
      res.status(200).json(data);
    } else {
      res.status(response.status).json(data);
    }
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
}
```

---

## 📊 Data Models

### Financial Transaction
```typescript
interface FinancialTransaction {
  document_type: string;
  amount: number;
  currency: string;
  date: string | null;
  description: string;
  category: 'income' | 'expense' | 'potential_income' | 'mixed';
  subcategory: string;
  document_id: string;
  confidence: number;
}
```

### Financial Bilan
```typescript
interface FinancialBilan {
  period: {
    days: number;
    start_date: string;
    end_date: string;
  };
  summary: {
    total_income: number;
    total_expenses: number;
    potential_income: number;
    net_result: number;
    profit_margin_percent: number;
  };
  currency_breakdown: Record<string, {
    income: number;
    expense: number;
    potential: number;
    net: number;
  }>;
  document_analysis: Record<string, {
    count: number;
    total_amount: number;
    average_amount: number;
  }>;
  transaction_count: number;
  recommendations: string[];
  generated_at: string;
}
```

---

## 🎯 Use Cases

### 1. **Small Business Accounting**
- Upload invoices, receipts, and purchase orders
- Generate monthly/quarterly financial reports
- Track income vs expenses automatically
- Get AI-powered financial recommendations

### 2. **Expense Management**
- Scan receipts and expense reports
- Categorize expenses automatically
- Multi-currency expense tracking
- Generate expense summaries

### 3. **Invoice Processing**
- Automatic invoice data extraction
- Payment tracking and due dates
- Customer/vendor information extraction
- Invoice validation and verification

### 4. **Financial Analytics**
- Document similarity search using embeddings
- Trend analysis across time periods
- Currency risk assessment
- Profit margin optimization

---

## 🚀 Performance & Scalability

- **Response Time**: 1-3 seconds per document
- **Throughput**: ~0.4 requests/second (can be scaled)
- **Accuracy**: 95%+ with Groq AI extraction
- **File Formats**: PDF, PNG, JPG, JPEG, TXT
- **Languages**: English, French, Arabic
- **Embedding Dimension**: 384 (semantic search ready)

---

## 🔒 Security & Best Practices

- **File Validation**: Validate file types and sizes
- **Rate Limiting**: Implement rate limiting for API calls
- **Error Handling**: Comprehensive error handling and logging
- **Data Privacy**: Temporary file cleanup after processing
- **API Keys**: Secure Groq API key management

---

## 📈 Future Enhancements

- **Real-time Processing**: WebSocket support for live updates
- **Batch Processing**: Handle multiple documents simultaneously
- **Custom Models**: Train domain-specific classification models
- **Advanced Analytics**: Predictive financial modeling
- **Integration APIs**: Connect with accounting software (QuickBooks, Xero)

This documentation provides everything you need to integrate the financial reporting API into your Next.js application! 🎉