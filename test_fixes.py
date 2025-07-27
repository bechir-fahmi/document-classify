"""
Test the fixes for document classification and amount extraction
"""
import requests
import tempfile
import os

def test_amount_extraction():
    """Test if amount extraction now works correctly"""
    
    # Test document with clear total
    test_content = """
INVOICE #INV001
Date: 2024-01-15
Company: Test Corp

Items:
- Service A: $25.00
- Service B: $20.00
- Tax: $6.94

Total: $51.94
"""
    
    print("🧪 Testing Amount Extraction Fix")
    print("-" * 40)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/analyze-financial", files=files)
        
        if response.status_code == 200:
            data = response.json()
            financial = data['financial_transaction']
            
            print(f"✅ Request successful!")
            print(f"   Expected amount: $51.94")
            print(f"   Extracted amount: {financial['amount']} {financial['currency']}")
            print(f"   Document type: {financial['document_type']}")
            print(f"   Classification confidence: {financial['confidence']:.3f}")
            
            # Check if amount is correct
            if abs(financial['amount'] - 51.94) < 0.01:
                print("✅ Amount extraction FIXED! ✅")
            else:
                print("❌ Amount extraction still wrong")
                
            # Check if document type is correct
            if financial['document_type'] == 'invoice':
                print("✅ Document classification FIXED! ✅")
            else:
                print(f"❌ Document classification still wrong: {financial['document_type']}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            
    finally:
        os.unlink(temp_file.name)

def test_classification_consistency():
    """Test if financial endpoint now matches /classify endpoint"""
    
    test_content = """
INVOICE #INV001
Date: 2024-01-15
Company: Test Corp
Service: Development
Total: $51.94
"""
    
    print(f"\n🔄 Testing Classification Consistency")
    print("-" * 40)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        # Test /classify endpoint
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            classify_response = requests.post("http://localhost:8000/classify", files=files)
        
        # Test /analyze-financial endpoint
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            financial_response = requests.post("http://localhost:8000/analyze-financial", files=files)
        
        if classify_response.status_code == 200 and financial_response.status_code == 200:
            classify_data = classify_response.json()
            financial_data = financial_response.json()
            
            classify_prediction = classify_data['final_prediction']
            financial_prediction = financial_data['financial_transaction']['document_type']
            
            print(f"   /classify prediction: {classify_prediction}")
            print(f"   /analyze-financial prediction: {financial_prediction}")
            
            if classify_prediction == financial_prediction:
                print("✅ Classification consistency FIXED! ✅")
            else:
                print("❌ Classification still inconsistent")
                
        else:
            print("❌ One or both requests failed")
            
    finally:
        os.unlink(temp_file.name)

def main():
    """Run the fix tests"""
    print("🔧 Testing Financial Analysis Fixes")
    print("=" * 50)
    
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
    
    test_amount_extraction()
    test_classification_consistency()
    
    print(f"\n🎯 Summary:")
    print(f"   - Amount extraction should now find 'Total: $51.94' correctly")
    print(f"   - Document classification should now match /classify endpoint")
    print(f"   - Both endpoints should use the same hybrid prediction logic")

if __name__ == "__main__":
    main()