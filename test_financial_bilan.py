"""
Test script for the financial bilan generation functionality
"""
import requests
import tempfile
import os
import json
from typing import List

def create_test_documents() -> List[tuple]:
    """Create test financial documents"""
    documents = [
        ("invoice_1.txt", """
INVOICE #INV-2024-001
Date: January 15, 2024
Bill To: ABC Company
Items:
- Consulting Services: €1,500.00
- Software License: €500.00
Subtotal: €2,000.00
Tax (20%): €400.00
Total: €2,400.00
"""),
        ("invoice_2.txt", """
FACTURE #FAC-2024-002
Date: January 20, 2024
Client: XYZ Corp
Montant HT: €3,200.00
TVA (20%): €640.00
Montant TTC: €3,840.00
"""),
        ("purchase_order.txt", """
PURCHASE ORDER #PO-2024-001
Date: January 18, 2024
Vendor: Office Supplies Inc
Items:
- Office Equipment: $800.00
- Stationery: $200.00
Total: $1,000.00
"""),
        ("receipt.txt", """
RECEIPT #REC-2024-001
Date: January 22, 2024
Store: Tech Store
Payment Method: Credit Card
Items:
- Laptop: €1,200.00
- Mouse: €25.00
Total: €1,225.00
"""),
        ("quote.txt", """
QUOTATION #QUO-2024-001
Date: January 25, 2024
Quote For: Future Client
Valid Until: February 25, 2024
Services:
- Web Development: €5,000.00
- Maintenance (1 year): €1,200.00
Total: €6,200.00
""")
    ]
    
    return documents

def test_single_document_analysis():
    """Test analyzing a single document for financial information"""
    print("🔍 Testing Single Document Financial Analysis")
    print("-" * 50)
    
    # Create a test invoice
    test_content = """
INVOICE #INV-2024-TEST
Date: January 15, 2024
Bill To: Test Company
Services: Consulting
Amount: €1,500.00
Tax: €300.00
Total: €1,800.00
"""
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/analyze-financial", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Single document analysis successful!")
            print(f"   Document Type: {data['financial_transaction']['document_type']}")
            print(f"   Amount: {data['financial_transaction']['amount']} {data['financial_transaction']['currency']}")
            print(f"   Category: {data['financial_transaction']['category']}")
            print(f"   Description: {data['financial_transaction']['description']}")
            print(f"   Confidence: {data['financial_transaction']['confidence']:.2f}")
            print(f"   Processing Time: {data['processing_time_ms']:.1f}ms")
            return True
        else:
            print(f"❌ Single document analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    finally:
        os.unlink(temp_file.name)

def test_bilan_generation():
    """Test generating a financial bilan from multiple documents"""
    print("\n📊 Testing Financial Bilan Generation")
    print("-" * 50)
    
    documents = create_test_documents()
    temp_files = []
    
    try:
        # Create temporary files
        for filename, content in documents:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        # Prepare files for upload
        files = []
        for temp_path in temp_files:
            files.append(('files', open(temp_path, 'rb')))
        
        try:
            # Send request
            response = requests.post(
                "http://localhost:8000/generate-bilan?period_days=30", 
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Bilan generation successful!")
                
                # Display summary
                summary = data['summary']
                print(f"\n📈 Financial Summary:")
                print(f"   Total Income: €{summary['total_income']:.2f}")
                print(f"   Total Expenses: €{summary['total_expenses']:.2f}")
                print(f"   Potential Income: €{summary['potential_income']:.2f}")
                print(f"   Net Result: €{summary['net_result']:.2f}")
                print(f"   Profit Margin: {summary['profit_margin_percent']:.1f}%")
                
                # Display currency breakdown
                print(f"\n💱 Currency Breakdown:")
                for currency, amounts in data['currency_breakdown'].items():
                    print(f"   {currency}:")
                    print(f"     Income: {amounts['income']:.2f}")
                    print(f"     Expenses: {amounts['expense']:.2f}")
                    print(f"     Net: {amounts['net']:.2f}")
                
                # Display document analysis
                print(f"\n📄 Document Analysis:")
                for doc_type, analysis in data['document_analysis'].items():
                    print(f"   {doc_type.title()}:")
                    print(f"     Count: {analysis['count']}")
                    print(f"     Total: {analysis['total_amount']:.2f}")
                    print(f"     Average: {analysis['average_amount']:.2f}")
                
                # Display recommendations
                print(f"\n💡 Recommendations:")
                for i, recommendation in enumerate(data['recommendations'], 1):
                    print(f"   {i}. {recommendation}")
                
                print(f"\n📊 Analysis Details:")
                print(f"   Period: {data['period']['days']} days")
                print(f"   Transactions Processed: {data['transaction_count']}")
                print(f"   Generated At: {data['generated_at']}")
                
                return True
                
            else:
                print(f"❌ Bilan generation failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        finally:
            # Close file handles
            for file_tuple in files:
                file_tuple[1].close()
            
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def test_financial_summary():
    """Test the financial summary endpoint"""
    print("\n📋 Testing Financial Summary Endpoint")
    print("-" * 50)
    
    try:
        response = requests.get("http://localhost:8000/financial-summary")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Financial summary retrieved successfully!")
            print(f"   Supported Document Types: {', '.join(data['supported_document_types'])}")
            print(f"   Supported Currencies: {', '.join(data['supported_currencies'])}")
            print(f"   Analysis Features: {len(data['analysis_features'])} features")
            print(f"   Bilan Metrics: {len(data['bilan_metrics'])} metrics")
            return True
        else:
            print(f"❌ Financial summary failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run all financial analysis tests"""
    print("🧪 Financial Bilan API Test Suite")
    print("=" * 60)
    print("Make sure your API is running on http://localhost:8000")
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy\n")
        else:
            print("❌ API health check failed")
            return
    except:
        print("❌ Cannot connect to API")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_financial_summary():
        tests_passed += 1
    
    if test_single_document_analysis():
        tests_passed += 1
    
    if test_bilan_generation():
        tests_passed += 1
    
    # Summary
    print(f"\n🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your financial bilan API is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()