"""
Simple test for the commercial endpoint to debug the issue
"""
import requests
import tempfile
import os

def test_commercial_endpoint():
    # Create a test file
    test_content = "INVOICE #12345\nDate: 2024-01-15\nBill To: ABC Company\nItems:\n- Product A: $100.00\n- Tax: $10.00\nTotal: $110.00"
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(test_content)
    temp_file.close()
    
    try:
        # Test the commercial endpoint
        with open(temp_file.name, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/classify-commercial", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
    finally:
        os.unlink(temp_file.name)

if __name__ == "__main__":
    test_commercial_endpoint()