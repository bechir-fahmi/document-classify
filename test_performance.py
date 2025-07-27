"""
Performance testing script for the Document Classification API with embeddings
"""
import time
import requests
import json
import statistics
import os
from typing import List, Dict, Any
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd

class APIPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def create_test_document(self, content: str, filename: str = "test_doc.txt") -> str:
        """Create a temporary test document"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def test_single_request(self, file_path: str, endpoint: str = "/classify") -> Dict[str, Any]:
        """Test a single API request and measure performance"""
        start_time = time.time()
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}{endpoint}", files=files)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response_time': (end_time - start_time) * 1000,  # ms
                    'server_processing_time': data.get('processing_time_ms', 0),
                    'embedding_length': len(data.get('document_embedding', [])),
                    'prediction': data.get('final_prediction', data.get('document_type', 'unknown')),
                    'confidence': data.get('model_confidence', data.get('confidence', 0)),
                    'response_size': len(response.content)
                }
            else:
                return {
                    'success': False,
                    'response_time': (end_time - start_time) * 1000,
                    'error': response.text,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'response_time': (end_time - start_time) * 1000,
                'error': str(e)
            }
    
    def test_concurrent_requests(self, file_path: str, num_requests: int = 10, 
                               max_workers: int = 5, endpoint: str = "/classify") -> List[Dict[str, Any]]:
        """Test concurrent API requests"""
        print(f"Testing {num_requests} concurrent requests with {max_workers} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.test_single_request, file_path, endpoint) 
                      for _ in range(num_requests)]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        return results
    
    def test_different_document_sizes(self, endpoint: str = "/classify") -> List[Dict[str, Any]]:
        """Test performance with different document sizes"""
        print("Testing different document sizes...")
        
        test_cases = [
            ("Small (100 chars)", "This is a small test invoice document. Invoice #12345. Total: $100.00"),
            ("Medium (500 chars)", "This is a medium-sized test invoice document. " * 10 + 
             "Invoice #12345. Date: 2024-01-01. Customer: Test Company. " +
             "Items: Product A $50, Product B $30, Tax $20. Total: $100.00"),
            ("Large (2000 chars)", "This is a large test invoice document. " * 50 + 
             "Invoice #12345. Date: 2024-01-01. Customer: Test Company. " +
             "Items: Product A $50, Product B $30, Tax $20. Total: $100.00"),
            ("Very Large (5000 chars)", "This is a very large test invoice document. " * 125 + 
             "Invoice #12345. Date: 2024-01-01. Customer: Test Company. " +
             "Items: Product A $50, Product B $30, Tax $20. Total: $100.00")
        ]
        
        results = []
        for size_name, content in test_cases:
            file_path = self.create_test_document(content)
            try:
                result = self.test_single_request(file_path, endpoint)
                result['document_size'] = size_name
                result['content_length'] = len(content)
                results.append(result)
                print(f"  {size_name}: {result['response_time']:.2f}ms")
            finally:
                os.unlink(file_path)
        
        return results
    
    def test_different_document_types(self, endpoint: str = "/classify") -> List[Dict[str, Any]]:
        """Test performance with different document types"""
        print("Testing different document types...")
        
        test_documents = {
            "invoice": "INVOICE #INV-2024-001\nDate: January 15, 2024\nBill To: ABC Company\nItems:\n- Product A: $100.00\n- Product B: $50.00\nSubtotal: $150.00\nTax: $15.00\nTotal: $165.00",
            "quote": "QUOTATION #QUO-2024-001\nDate: January 15, 2024\nQuote For: XYZ Corp\nValid Until: February 15, 2024\nItems:\n- Service A: $200.00\n- Service B: $100.00\nTotal: $300.00",
            "purchase_order": "PURCHASE ORDER #PO-2024-001\nDate: January 15, 2024\nVendor: Supplier Inc\nDelivery Date: January 30, 2024\nItems:\n- Item A: 10 units @ $20.00\n- Item B: 5 units @ $30.00\nTotal: $350.00",
            "receipt": "RECEIPT #REC-2024-001\nDate: January 15, 2024\nStore: Local Shop\nPayment Method: Credit Card\nItems:\n- Coffee: $4.50\n- Sandwich: $8.00\nTotal: $12.50"
        }
        
        results = []
        for doc_type, content in test_documents.items():
            file_path = self.create_test_document(content)
            try:
                result = self.test_single_request(file_path, endpoint)
                result['expected_type'] = doc_type
                results.append(result)
                print(f"  {doc_type}: {result['response_time']:.2f}ms - Predicted: {result.get('prediction', 'N/A')}")
            finally:
                os.unlink(file_path)
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run a comprehensive performance test"""
        print("🚀 Starting comprehensive performance test...")
        print("=" * 60)
        
        # Test API availability
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                print("❌ API is not available!")
                return {}
        except:
            print("❌ Cannot connect to API!")
            return {}
        
        print("✅ API is available")
        
        # Create a test document
        test_content = "INVOICE #12345\nDate: 2024-01-01\nCustomer: Test Company\nAmount: $100.00"
        test_file = self.create_test_document(test_content)
        
        try:
            results = {}
            
            # 1. Single request test
            print("\n1. Single Request Test")
            print("-" * 30)
            single_result = self.test_single_request(test_file)
            if single_result['success']:
                print(f"✅ Response Time: {single_result['response_time']:.2f}ms")
                print(f"✅ Server Processing: {single_result['server_processing_time']:.2f}ms")
                print(f"✅ Embedding Length: {single_result['embedding_length']}")
                print(f"✅ Response Size: {single_result['response_size']} bytes")
            else:
                print(f"❌ Failed: {single_result.get('error', 'Unknown error')}")
            
            results['single_request'] = single_result
            
            # 2. Concurrent requests test
            print("\n2. Concurrent Requests Test")
            print("-" * 30)
            concurrent_results = self.test_concurrent_requests(test_file, num_requests=10, max_workers=3)
            successful_requests = [r for r in concurrent_results if r['success']]
            
            if successful_requests:
                response_times = [r['response_time'] for r in successful_requests]
                print(f"✅ Successful Requests: {len(successful_requests)}/10")
                print(f"✅ Average Response Time: {statistics.mean(response_times):.2f}ms")
                print(f"✅ Min Response Time: {min(response_times):.2f}ms")
                print(f"✅ Max Response Time: {max(response_times):.2f}ms")
                print(f"✅ Std Deviation: {statistics.stdev(response_times) if len(response_times) > 1 else 0:.2f}ms")
            
            results['concurrent_requests'] = concurrent_results
            
            # 3. Different document sizes
            print("\n3. Document Size Performance")
            print("-" * 30)
            size_results = self.test_different_document_sizes()
            results['document_sizes'] = size_results
            
            # 4. Different document types
            print("\n4. Document Type Performance")
            print("-" * 30)
            type_results = self.test_different_document_types()
            results['document_types'] = type_results
            
            # 5. Test commercial endpoint
            print("\n5. Commercial Endpoint Test")
            print("-" * 30)
            commercial_result = self.test_single_request(test_file, "/classify-commercial")
            if commercial_result['success']:
                print(f"✅ Commercial Response Time: {commercial_result['response_time']:.2f}ms")
                print(f"✅ Commercial Embedding Length: {commercial_result['embedding_length']}")
            else:
                print(f"❌ Commercial endpoint failed: {commercial_result.get('error', 'Unknown error')}")
            
            results['commercial_endpoint'] = commercial_result
            
            return results
            
        finally:
            os.unlink(test_file)
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed performance report"""
        report = []
        report.append("📊 PERFORMANCE TEST REPORT")
        report.append("=" * 50)
        
        if 'single_request' in results and results['single_request']['success']:
            sr = results['single_request']
            report.append(f"\n🔍 Single Request Performance:")
            report.append(f"   Response Time: {sr['response_time']:.2f}ms")
            report.append(f"   Server Processing: {sr['server_processing_time']:.2f}ms")
            report.append(f"   Network Overhead: {sr['response_time'] - sr['server_processing_time']:.2f}ms")
            report.append(f"   Embedding Dimension: {sr['embedding_length']}")
            report.append(f"   Response Size: {sr['response_size']} bytes")
        
        if 'concurrent_requests' in results:
            successful = [r for r in results['concurrent_requests'] if r['success']]
            if successful:
                times = [r['response_time'] for r in successful]
                report.append(f"\n⚡ Concurrent Requests (10 requests, 3 workers):")
                report.append(f"   Success Rate: {len(successful)}/10 ({len(successful)*10}%)")
                report.append(f"   Average Time: {statistics.mean(times):.2f}ms")
                report.append(f"   Min/Max Time: {min(times):.2f}ms / {max(times):.2f}ms")
                report.append(f"   Throughput: ~{1000/statistics.mean(times):.1f} requests/second")
        
        if 'document_sizes' in results:
            report.append(f"\n📄 Document Size Impact:")
            for result in results['document_sizes']:
                if result['success']:
                    report.append(f"   {result['document_size']}: {result['response_time']:.2f}ms")
        
        if 'document_types' in results:
            report.append(f"\n📋 Document Type Accuracy:")
            for result in results['document_types']:
                if result['success']:
                    predicted = result.get('prediction', 'unknown')
                    expected = result.get('expected_type', 'unknown')
                    accuracy = "✅" if predicted == expected else "❌"
                    report.append(f"   {expected}: {predicted} {accuracy} ({result['response_time']:.2f}ms)")
        
        return "\n".join(report)

def main():
    """Run the performance test"""
    tester = APIPerformanceTester()
    
    print("🧪 Document Classification API Performance Test")
    print("Make sure your API is running on http://localhost:8000")
    input("Press Enter to start the test...")
    
    results = tester.run_comprehensive_test()
    
    if results:
        print("\n" + "=" * 60)
        report = tester.generate_performance_report(results)
        print(report)
        
        # Save results to file
        with open('performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to 'performance_results.json'")
    
    print("\n🎉 Performance test completed!")

if __name__ == "__main__":
    main()