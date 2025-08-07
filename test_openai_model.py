#!/usr/bin/env python3
"""
Simple test to check if OpenAI GPT-OSS-20B model is available and working
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def test_openai_model():
    """Test if OpenAI GPT-OSS-20B is available"""
    print("🧪 Testing OpenAI GPT-OSS-20B Availability")
    print("=" * 50)
    
    model_name = "openai/gpt-oss-20b"
    
    try:
        print(f"📥 Checking if {model_name} is available...")
        
        # Try to load just the tokenizer first (faster test)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Test a simple encoding
        test_text = "What is supply chain management?"
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenization test passed ({len(tokens)} tokens)")
        
        print("\n🎉 OpenAI GPT-OSS-20B model is available!")
        print("💡 You can now load the full model in the Streamlit app")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        if "404" in str(e) or "not found" in str(e).lower():
            print("\n💡 Model not found - this could mean:")
            print("   - The model hasn't been released yet")
            print("   - The model name has changed")
            print("   - You need authentication")
        elif "connection" in str(e).lower():
            print("\n💡 Connection issue - check your internet connection")
        else:
            print(f"\n💡 Unexpected error: {e}")
        
        print("\n🔄 The app will use rule-based analysis instead")
        return False

if __name__ == "__main__":
    success = test_openai_model()
    
    if not success:
        print("\n" + "="*50)
        print("🔄 FALLBACK MODE")
        print("The demo will work with rule-based analysis")
        print("AI features will be available when the model is released")
    
    sys.exit(0 if success else 1) 