#!/usr/bin/env python3
"""
Test script for the Supply Chain Intelligence Demo
"""

import pandas as pd
import sys
import os

def test_data_loading():
    """Test that data loading works correctly"""
    sys.path.append('.')
    from app import load_supply_chain_data
    
    try:
        data = load_supply_chain_data()
        
        # Check all required dataframes exist
        required_keys = ['items', 'warehouses', 'inventory', 'deliveries']
        for key in required_keys:
            assert key in data, f"Missing {key} data"
            assert not data[key].empty, f"{key} data is empty"
        
        print("✅ Data loading test passed")
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_query_processing():
    """Test query processing functions"""
    sys.path.append('.')
    from app import extract_relevant_data, load_supply_chain_data
    
    try:
        data = load_supply_chain_data()
        
        # Test inventory query
        relevant = extract_relevant_data("Which items need restocking?", data)
        assert 'inventory' in relevant, "Inventory data not extracted for stock query"
        assert 'items' in relevant, "Items data not extracted for stock query"
        
        # Test warehouse query
        relevant = extract_relevant_data("Show warehouse utilization", data)
        assert 'warehouses' in relevant, "Warehouse data not extracted for warehouse query"
        
        print("✅ Query processing test passed")
        return True
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        return False

def test_fallback_response():
    """Test fallback response generation"""
    sys.path.append('.')
    from app import generate_fallback_response, load_supply_chain_data
    
    try:
        data = load_supply_chain_data()
        
        # Test stock query
        response = generate_fallback_response("Which items need restocking?", data)
        assert len(response) > 0, "Empty response generated"
        assert "stock" in response.lower() or "restock" in response.lower(), "Response doesn't contain stock information"
        
        print("✅ Fallback response test passed")
        return True
    except Exception as e:
        print(f"❌ Fallback response test failed: {e}")
        return False

def main():
    print("🧪 Running Supply Chain Demo Tests")
    print("=" * 40)
    
    tests = [
        test_data_loading,
        test_query_processing,
        test_fallback_response
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The app should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 