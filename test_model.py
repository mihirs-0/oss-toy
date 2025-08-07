#!/usr/bin/env python3
"""
Test script to verify OpenAI GPT-OSS-20B model loading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys

def test_model_loading():
    """Test if the 20B model can be loaded"""
    print("🧪 Testing OpenAI GPT-OSS-20B Model Loading")
    print("=" * 50)
    
    try:
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU (very slow)")
        
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model_name = "openai/gpt-oss-20b"
        
        print(f"📥 Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        
        print(f"📥 Loading model from {model_name}...")
        print("   This may take several minutes and ~10GB VRAM...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        # Test generation
        print("\n🔬 Testing text generation...")
        test_prompt = "What is supply chain management?"
        
        inputs = tokenizer.encode(test_prompt, return_tensors='pt', max_length=200, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                min_new_tokens=10,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=2,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print("✅ Generation test successful")
        print(f"📄 Sample response: {response[:200]}...")
        
        print("\n🎉 All tests passed! OpenAI GPT-OSS-20B model is ready for use.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   - Ensure you have sufficient VRAM (10GB+ recommended)")
        print("   - Check internet connection for model download")
        print("   - Try restarting if you get CUDA memory errors")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 