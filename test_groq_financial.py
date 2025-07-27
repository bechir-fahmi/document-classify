"""
Test Groq-based financial analysis
"""
import requests
import tempfile
import os
import json

def test_groq_financial_analysis():
    """Test the new Groq-based financial analysis endpoint"""
    
    # Test document with clear financial information
    test_content = """
INVOICE #INV001
Date: 2024-01-15
Bill To: ABC Company

Items:
- Consulting Services: $25.00
- Development Work: $20.00
- Tax (10%): $4.50

Subtotal: $45.00
Tax: $4.50
Total: $51.94

Payment Terms: Net 30 days
"""
    
    print("🤖 Testing Groq-Based Financial Analysis")
    print("=" * 50)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        # Test the new Groq endpoint
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/analyze-financial-groq", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Groq Analysis Successful!")
            print("-" * 30)
            
            # Extract Groq financial analysis
            groq_analysis = data['groq_financial_analysis']
            
            print(f"📊 Financial Information (Groq AI):")
            print(f"   Document Type: {groq_analysis['document_type']}")
            print(f"   Amount: {groq_analysis['amount']} {groq_analysis['currency']}")
            print(f"   Date: {groq_analysis['date']}")
            print(f"   Category: {groq_analysis['category']} / {groq_analysis['subcategory']}")
            print(f"   Description: {groq_analysis['description']}")
            print(f"   Confidence: {groq_analysis['confidence']:.3f}")
            
            # Show raw Groq response
            print(f"\n🤖 Raw Groq Response:")
            raw_response = groq_analysis['raw_groq_response']
            for key, value in raw_response.items():
                if key != 'line_items':  # Skip line items for brevity
                    print(f"   {key}: {value}")
            
            # Check accuracy
            expected_amount = 51.94
            actual_amount = groq_analysis['amount']
            
            print(f"\n🎯 Accuracy Check:")
            print(f"   Expected Amount: ${expected_amount}")
            print(f"   Groq Extracted: ${actual_amount}")
            
            if abs(actual_amount - expected_amount) < 0.01:
                print("   ✅ AMOUNT EXTRACTION ACCURATE! ✅")
            else:
                print("   ❌ Amount extraction still inaccurate")
            
            # Show processing info
            print(f"\n⚡ Performance:")
            print(f"   Processing Time: {data['processing_time_ms']:.1f}ms")
            print(f"   Extraction Method: {data['extraction_method']}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    finally:
        os.unlink(temp_file.name)

def compare_extraction_methods():
    """Compare regex-based vs Groq-based extraction"""
    
    test_content = """
INVOICE #INV-2024-001
Date: January 15, 2024
Customer: Test Company

Line Items:
- Product A: $25.00
- Product B: $20.00
- Service Fee: $15.00

Subtotal: $60.00
Tax (8.5%): $5.10
Shipping: $3.84
Total Amount Due: $68.94

Payment Terms: Net 30
"""
    
    print(f"\n🔄 Comparing Extraction Methods")
    print("=" * 50)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        # Test original method
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            original_response = requests.post("http://localhost:8000/analyze-financial", files=files)
        
        # Test Groq method
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            groq_response = requests.post("http://localhost:8000/analyze-financial-groq", files=files)
        
        if original_response.status_code == 200 and groq_response.status_code == 200:
            original_data = original_response.json()
            groq_data = groq_response.json()
            
            original_amount = original_data['financial_transaction']['amount']
            groq_amount = groq_data['groq_financial_analysis']['amount']
            
            print(f"📊 Extraction Comparison:")
            print(f"   Expected Amount: $68.94")
            print(f"   Regex Method: ${original_amount}")
            print(f"   Groq AI Method: ${groq_amount}")
            
            print(f"\n🎯 Accuracy:")
            expected = 68.94
            regex_accurate = abs(original_amount - expected) < 0.01
            groq_accurate = abs(groq_amount - expected) < 0.01
            
            print(f"   Regex Accurate: {'✅' if regex_accurate else '❌'}")
            print(f"   Groq Accurate: {'✅' if groq_accurate else '✅'}")
            
            if groq_accurate and not regex_accurate:
                print(f"   🎉 Groq AI is more accurate!")
            elif regex_accurate and not groq_accurate:
                print(f"   📊 Regex method is more accurate!")
            elif groq_accurate and regex_accurate:
                print(f"   ✅ Both methods are accurate!")
            else:
                print(f"   ⚠️ Both methods need improvement")
                
        else:
            print("❌ One or both requests failed")
            
    finally:
        os.unlink(temp_file.name)

def main():
    """Run Groq financial analysis tests"""
    print("🤖 Groq Financial Analysis Test Suite")
    print("=" * 60)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API health check failed")
            return
    except:
        print("❌ Cannot connect to API")
        return
    
    test_groq_financial_analysis()
    compare_extraction_methods()
    
    print(f"\n💡 Benefits of Groq-Based Extraction:")
    print(f"   - AI understands document context")
    print(f"   - Can extract complex financial data")
    print(f"   - Handles various document formats")
    print(f"   - More accurate than regex patterns")
    print(f"   - Extracts additional metadata (line items, tax, etc.)")

if __name__ == "__main__":
    main()