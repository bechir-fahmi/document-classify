#!/usr/bin/env python3
"""
Run All Tests

This script runs all the tests for the Commercial Document Classification System.
"""

import os
import sys
import unittest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_tests():
    """Run all tests in the tests directory"""
    # Add the parent directory to the path so we can import modules properly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Get all test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    # Print test summary
    print("=== Commercial Document Classification System Tests ===")
    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    print()
    
    # Discover and run all tests
    print("Running all tests...")
    suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Print test summary
    print("\n=== Test Results Summary ===")
    print(f"Ran {result.testsRun} tests")
    print(f"Successes: {result.testsRun - (len(result.errors) + len(result.failures))}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_all_tests()) 