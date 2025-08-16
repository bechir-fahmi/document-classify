"""
Groq-based financial data extraction for accurate financial analysis
"""
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from utils.groq_utils import client as groq_client

logger = logging.getLogger(__name__)

@dataclass
class GroqFinancialTransaction:
    """Financial transaction extracted using Groq AI"""
    document_type: str
    amount: float
    currency: str
    date: Optional[datetime]
    description: str
    category: str
    subcategory: str
    document_id: str
    confidence: float
    raw_groq_response: Dict[str, Any]

class GroqFinancialAnalyzer:
    """Uses Groq AI to extract financial information from documents"""
    
    def __init__(self):
        self.client = groq_client
    
    def extract_financial_data(self, text: str, document_type: str, document_id: str) -> GroqFinancialTransaction:
        """Extract financial data using Groq AI"""
        
        try:
            # Create a comprehensive prompt for financial data extraction
            prompt = self._create_financial_extraction_prompt(text, document_type)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial document analysis expert. Extract financial information accurately and return it in the specified JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=1000
            )
            
            # Parse the response
            groq_response = response.choices[0].message.content
            logger.info(f"Groq financial extraction response: {groq_response}")
            
            # Parse JSON response
            financial_data = self._parse_groq_response(groq_response)
            
            # Create transaction object
            return GroqFinancialTransaction(
                document_type=financial_data.get('document_type', document_type),
                amount=financial_data.get('amount', 0.0),
                currency=financial_data.get('currency', 'USD'),
                date=self._parse_date(financial_data.get('date')),
                description=financial_data.get('description', f"{document_type} transaction"),
                category=financial_data.get('category', 'mixed'),
                subcategory=financial_data.get('subcategory', 'unclassified'),
                document_id=document_id,
                confidence=financial_data.get('confidence', 0.8),
                raw_groq_response=financial_data
            )
            
        except Exception as e:
            logger.error(f"Error in Groq financial extraction: {str(e)}")
            # Fallback to basic extraction
            return self._create_fallback_transaction(text, document_type, document_id)
    
    def _create_financial_extraction_prompt(self, text: str, document_type: str) -> str:
        """Create a comprehensive prompt for financial data extraction"""
        
        prompt = f"""
Analyze this {document_type} document and extract the financial information. Return the data in JSON format.

Document Text:
{text}

Please extract and return the following information in JSON format:
{{
    "document_type": "invoice|quote|purchase_order|receipt|bank_statement|expense_report|payslip|delivery_note|unknown",
    "amount": <total_amount_as_number>,
    "currency": "USD|EUR|GBP|TND",
    "date": "YYYY-MM-DD",
    "description": "brief description of the transaction",
    "category": "income|expense|potential_income|mixed",
    "subcategory": "sales_revenue|purchases|operational_expense|quoted_sales|salary_expense|bank_transaction|delivered_goods|unclassified",
    "confidence": <confidence_score_0_to_1>,
    "line_items": [
        {{"item": "description", "amount": <amount>}},
        ...
    ],
    "tax_amount": <tax_amount_if_present>,
    "subtotal": <subtotal_if_present>,
    "payment_terms": "payment terms if mentioned",
    "vendor_customer": "vendor or customer name if present"
}}

Instructions:
1. For AMOUNT: Find the TOTAL amount (not subtotal, not individual items). Look for "Total:", "Grand Total:", "Amount Due:", "Final Amount:", etc.
2. For CURRENCY: Identify from symbols ($, €, £) or text (USD, EUR, GBP, TND, DT)
3. For DATE: Extract the document date, invoice date, or transaction date
4. For CATEGORY: 
   - "income" for invoices, delivery notes
   - "expense" for purchase orders, receipts, expense reports, payslips
   - "potential_income" for quotes
   - "mixed" for bank statements or unclear documents
5. For CONFIDENCE: Rate 0.0-1.0 based on how clear the financial information is
6. Return ONLY the JSON, no additional text.
"""
        return prompt
    
    def _parse_groq_response(self, response: str) -> Dict[str, Any]:
        """Parse Groq's JSON response"""
        try:
            # Clean the response - remove any markdown formatting and text before JSON
            cleaned_response = response.strip()
            
            # Remove any text before the JSON starts
            if '{' in cleaned_response:
                start_idx = cleaned_response.find('{')
                cleaned_response = cleaned_response[start_idx:]
            
            # Remove any text after the JSON ends
            if '}' in cleaned_response:
                end_idx = cleaned_response.rfind('}') + 1
                cleaned_response = cleaned_response[:end_idx]
            
            # Remove markdown formatting
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            data = json.loads(cleaned_response)
            logger.info(f"Successfully parsed Groq JSON response")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq JSON response: {e}")
            logger.error(f"Raw response: {response}")
            
            # Try to extract key information using regex as fallback
            return self._extract_fallback_data(response)
    
    def _extract_fallback_data(self, response: str) -> Dict[str, Any]:
        """Extract data from response if JSON parsing fails"""
        import re
        
        # Try to extract amount
        amount_match = re.search(r'"amount":\s*([0-9.]+)', response)
        amount = float(amount_match.group(1)) if amount_match else 0.0
        
        # Try to extract currency
        currency_match = re.search(r'"currency":\s*"([A-Z]{3})"', response)
        currency = currency_match.group(1) if currency_match else "USD"
        
        # Try to extract document type
        doc_type_match = re.search(r'"document_type":\s*"([^"]+)"', response)
        doc_type = doc_type_match.group(1) if doc_type_match else "unknown"
        
        return {
            "document_type": doc_type,
            "amount": amount,
            "currency": currency,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "description": "Extracted from document",
            "category": "mixed",
            "subcategory": "unclassified",
            "confidence": 0.6
        }
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try different date formats
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If no format works, return current date
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def _create_fallback_transaction(self, text: str, document_type: str, document_id: str) -> GroqFinancialTransaction:
        """Create a fallback transaction if Groq extraction fails"""
        return GroqFinancialTransaction(
            document_type=document_type,
            amount=0.0,
            currency="USD",
            date=datetime.now(),
            description=f"Fallback {document_type} transaction",
            category="mixed",
            subcategory="unclassified",
            document_id=document_id,
            confidence=0.3,
            raw_groq_response={"error": "Groq extraction failed"}
        )

def analyze_document_with_groq(text: str, document_type: str, document_id: str) -> GroqFinancialTransaction:
    """Convenience function to analyze a document using Groq"""
    analyzer = GroqFinancialAnalyzer()
    return analyzer.extract_financial_data(text, document_type, document_id)