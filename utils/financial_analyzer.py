"""
Financial analysis utilities for generating financial reports (bilan)
"""
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

@dataclass
class FinancialTransaction:
    """Represents a financial transaction extracted from a document"""
    document_type: str
    amount: float
    currency: str
    date: Optional[datetime]
    description: str
    category: str  # income, expense, asset, liability
    subcategory: str  # invoice_income, purchase_expense, etc.
    document_id: str
    confidence: float

class FinancialAnalyzer:
    """Analyzes documents to extract financial information and generate reports"""
    
    def __init__(self):
        self.currency_patterns = {
            'EUR': [r'€', r'EUR', r'euro'],
            'USD': [r'\$', r'USD', r'dollar'],
            'TND': [r'TND', r'DT', r'dinar'],
            'GBP': [r'£', r'GBP', r'pound']
        }
        
        # Priority patterns for totals (most specific first)
        self.total_patterns = [
            r'total[:\s]*\$\s*([\d,]+\.?\d*)',  # "Total: $51.94"
            r'total[:\s]*([\d,]+\.?\d*)\s*\$',  # "Total: 51.94 $"
            r'grand\s+total[:\s]*\$\s*([\d,]+\.?\d*)',  # "Grand Total: $51.94"
            r'amount\s+due[:\s]*\$\s*([\d,]+\.?\d*)',  # "Amount Due: $51.94"
            r'final\s+amount[:\s]*\$\s*([\d,]+\.?\d*)',  # "Final Amount: $51.94"
            r'total[:\s]*€\s*([\d,]+\.?\d*)',  # "Total: €51.94"
            r'total[:\s]*([\d,]+\.?\d*)\s*€',  # "Total: 51.94 €"
            r'montant\s+total[:\s]*€?\s*([\d,]+\.?\d*)',  # "Montant Total: €51.94"
            r'total[:\s]*£\s*([\d,]+\.?\d*)',  # "Total: £51.94"
        ]
        
        # Secondary patterns (less specific)
        self.amount_patterns = [
            r'montant[:\s]*([€$£]?\s*[\d,]+\.?\d*)\s*([€$£TND DT]*)',
            r'amount[:\s]*([€$£]?\s*[\d,]+\.?\d*)\s*([€$£TND DT]*)',
            r'prix[:\s]*([€$£]?\s*[\d,]+\.?\d*)\s*([€$£TND DT]*)',
            r'price[:\s]*([€$£]?\s*[\d,]+\.?\d*)\s*([€$£TND DT]*)',
            r'([€$£])\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*([€$£TND DT]+)'
        ]
        
        self.date_patterns = [
            r'date[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}\s+\w+\s+\d{4})'
        ]
    
    def extract_financial_data(self, text: str, document_type: str, document_id: str) -> FinancialTransaction:
        """Extract financial data from document text"""
        
        # Clean and ensure text is properly encoded
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            text = str(text)
        except:
            text = str(text)
        
        # Extract amount and currency
        amount, currency = self._extract_amount_and_currency(text)
        
        # Extract date
        date = self._extract_date(text)
        
        # Determine category and subcategory based on document type
        category, subcategory = self._categorize_transaction(document_type)
        
        # Calculate confidence based on extraction success
        confidence = self._calculate_confidence(amount, currency, date, document_type)
        
        return FinancialTransaction(
            document_type=document_type,
            amount=amount,
            currency=currency,
            date=date,
            description=self._extract_description(text, document_type),
            category=category,
            subcategory=subcategory,
            document_id=document_id,
            confidence=confidence
        )
    
    def _extract_amount_and_currency(self, text: str) -> tuple[float, str]:
        """Extract monetary amount and currency from text with priority for totals"""
        text_lower = text.lower()
        
        # STEP 1: Try to find amounts specifically labeled as "total" (highest priority)
        for pattern in self.total_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = str(match).replace(',', '').strip()
                    if amount_str:
                        amount = float(amount_str)
                        if amount > 0:
                            # Determine currency from the pattern context
                            currency = self._identify_currency(text_lower)
                            logger.info(f"Found total amount: {amount} {currency} using pattern: {pattern}")
                            return amount, currency
                except (ValueError, InvalidOperation):
                    continue
        
        # STEP 2: Try secondary patterns if no total found
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple) and len(match) >= 2:
                        amount_str = match[0].strip()
                        currency_str = match[1].strip()
                    else:
                        amount_str = str(match).strip()
                        currency_str = ""
                    
                    # Clean amount string
                    amount_str = re.sub(r'[€$£,\s]', '', amount_str)
                    if amount_str:
                        amount = float(amount_str)
                        
                        # Determine currency
                        currency = self._identify_currency(currency_str + " " + text_lower)
                        
                        if amount > 0:
                            logger.info(f"Found amount: {amount} {currency} using secondary pattern: {pattern}")
                            return amount, currency
                            
                except (ValueError, InvalidOperation):
                    continue
        
        logger.warning("No amount found in document")
        return 0.0, "USD"  # Default values
    
    def _identify_currency(self, text: str) -> str:
        """Identify currency from text"""
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return currency
        return "EUR"  # Default currency
    
    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract date from text"""
        # Clean text to handle encoding issues
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            text = str(text)
        except:
            text = str(text)
            
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try different date formats
                    date_str = match.strip()
                    
                    # Try DD/MM/YYYY or DD-MM-YYYY
                    if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
                        parts = re.split(r'[/-]', date_str)
                        if len(parts) == 3:
                            day, month, year = parts
                            if len(year) == 2:
                                year = "20" + year
                            return datetime(int(year), int(month), int(day))
                    
                    # Try YYYY-MM-DD
                    elif re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        return datetime.strptime(date_str, '%Y-%m-%d')
                        
                except (ValueError, IndexError):
                    continue
        
        return datetime.now()  # Default to current date
    
    def _categorize_transaction(self, document_type: str) -> tuple[str, str]:
        """Categorize transaction based on document type"""
        categorization = {
            'invoice': ('income', 'sales_revenue'),
            'quote': ('potential_income', 'quoted_sales'),
            'purchase_order': ('expense', 'purchases'),
            'receipt': ('expense', 'operational_expense'),
            'bank_statement': ('mixed', 'bank_transaction'),
            'expense_report': ('expense', 'operational_expense'),
            'payslip': ('expense', 'salary_expense'),
            'delivery_note': ('income', 'delivered_goods'),
            'unknown': ('mixed', 'unclassified')
        }
        
        return categorization.get(document_type, ('mixed', 'unclassified'))
    
    def _extract_description(self, text: str, document_type: str) -> str:
        """Extract a meaningful description from the document"""
        # Get first meaningful line or summary
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Look for description patterns
        description_patterns = [
            r'description[:\s]*(.+)',
            r'objet[:\s]*(.+)',
            r'subject[:\s]*(.+)',
            r'item[:\s]*(.+)'
        ]
        
        for pattern in description_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:100]
        
        # Fallback to document type and first meaningful content
        if lines:
            return f"{document_type.title()}: {lines[0][:50]}..."
        
        return f"{document_type.title()} transaction"
    
    def _calculate_confidence(self, amount: float, currency: str, date: Optional[datetime], document_type: str) -> float:
        """Calculate confidence score for the extracted financial data"""
        confidence = 0.0
        
        # Amount confidence
        if amount > 0:
            confidence += 0.4
        
        # Currency confidence
        if currency and currency != "EUR":  # Default currency gets lower score
            confidence += 0.2
        elif currency:
            confidence += 0.1
        
        # Date confidence
        if date and date != datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
            confidence += 0.2
        elif date:
            confidence += 0.1
        
        # Document type confidence
        if document_type != 'unknown':
            confidence += 0.2
        
        return min(confidence, 1.0)

class FinancialReportGenerator:
    """Generates financial reports (bilan) from analyzed documents"""
    
    def __init__(self):
        self.analyzer = FinancialAnalyzer()
    
    def generate_bilan(self, transactions: List[FinancialTransaction], period_days: int = 30) -> Dict[str, Any]:
        """Generate a financial bilan (balance sheet) from transactions"""
        
        # Filter transactions by period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_transactions = [
            t for t in transactions 
            if t.date and t.date >= cutoff_date
        ]
        
        # Calculate totals by category
        income_total = sum(t.amount for t in recent_transactions if t.category == 'income')
        expense_total = sum(t.amount for t in recent_transactions if t.category == 'expense')
        potential_income = sum(t.amount for t in recent_transactions if t.category == 'potential_income')
        
        # Group by currency
        currency_breakdown = {}
        for transaction in recent_transactions:
            currency = transaction.currency
            if currency not in currency_breakdown:
                currency_breakdown[currency] = {'income': 0, 'expense': 0, 'potential': 0}
            
            if transaction.category == 'income':
                currency_breakdown[currency]['income'] += transaction.amount
            elif transaction.category == 'expense':
                currency_breakdown[currency]['expense'] += transaction.amount
            elif transaction.category == 'potential_income':
                currency_breakdown[currency]['potential'] += transaction.amount
        
        # Calculate key metrics
        net_result = income_total - expense_total
        profit_margin = (net_result / income_total * 100) if income_total > 0 else 0
        
        # Group by document type
        document_type_summary = {}
        for transaction in recent_transactions:
            doc_type = transaction.document_type
            if doc_type not in document_type_summary:
                document_type_summary[doc_type] = {
                    'count': 0,
                    'total_amount': 0,
                    'average_amount': 0
                }
            
            document_type_summary[doc_type]['count'] += 1
            document_type_summary[doc_type]['total_amount'] += transaction.amount
        
        # Calculate averages
        for doc_type in document_type_summary:
            summary = document_type_summary[doc_type]
            summary['average_amount'] = summary['total_amount'] / summary['count']
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            income_total, expense_total, potential_income, recent_transactions
        )
        
        return {
            'period': {
                'days': period_days,
                'start_date': cutoff_date.isoformat(),
                'end_date': datetime.now().isoformat()
            },
            'summary': {
                'total_income': round(income_total, 2),
                'total_expenses': round(expense_total, 2),
                'potential_income': round(potential_income, 2),
                'net_result': round(net_result, 2),
                'profit_margin_percent': round(profit_margin, 2)
            },
            'currency_breakdown': {
                currency: {
                    'income': round(amounts['income'], 2),
                    'expense': round(amounts['expense'], 2),
                    'potential': round(amounts['potential'], 2),
                    'net': round(amounts['income'] - amounts['expense'], 2)
                }
                for currency, amounts in currency_breakdown.items()
            },
            'document_analysis': {
                doc_type: {
                    'count': summary['count'],
                    'total_amount': round(summary['total_amount'], 2),
                    'average_amount': round(summary['average_amount'], 2)
                }
                for doc_type, summary in document_type_summary.items()
            },
            'transaction_count': len(recent_transactions),
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, income: float, expenses: float, potential: float, 
                                transactions: List[FinancialTransaction]) -> List[str]:
        """Generate financial recommendations based on the analysis"""
        recommendations = []
        
        # Cash flow recommendations
        if expenses > income:
            recommendations.append("⚠️ Expenses exceed income. Consider cost reduction strategies.")
        elif income > expenses * 2:
            recommendations.append("💰 Strong cash flow. Consider investment opportunities.")
        
        # Potential income recommendations
        if potential > income * 0.5:
            recommendations.append("📈 High potential income from quotes. Focus on conversion.")
        
        # Document type recommendations
        doc_counts = {}
        for t in transactions:
            doc_counts[t.document_type] = doc_counts.get(t.document_type, 0) + 1
        
        if doc_counts.get('quote', 0) > doc_counts.get('invoice', 0):
            recommendations.append("📋 More quotes than invoices. Improve quote-to-invoice conversion.")
        
        # Currency recommendations
        currencies = set(t.currency for t in transactions)
        if len(currencies) > 2:
            recommendations.append("💱 Multiple currencies detected. Consider currency risk management.")
        
        # General recommendations
        if len(transactions) < 5:
            recommendations.append("📊 Limited transaction data. More documents needed for better analysis.")
        
        return recommendations[:5]  # Limit to 5 recommendations

def analyze_document_for_bilan(text: str, document_type: str, document_id: str) -> FinancialTransaction:
    """Convenience function to analyze a single document for financial reporting"""
    analyzer = FinancialAnalyzer()
    return analyzer.extract_financial_data(text, document_type, document_id)