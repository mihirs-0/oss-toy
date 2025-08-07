#!/usr/bin/env python3
"""
Test the improved model loading approach for OpenAI GPT-OSS-20B
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys

def test_model_loading_approaches():
    """Test different model loading approaches"""
    print("🧪 Testing OpenAI GPT-OSS-20B Loading Approaches")
    print("=" * 60)
    
    model_name = "openai/gpt-oss-20b"
    
    try:
        print(f"📥 Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        
        # Test different loading approaches
        model = None
        
        # Approach 1: Auto device mapping (recommended)
        try:
            print("\n🔄 Approach 1: Auto device mapping...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ Success with device mapping!")
            
        except Exception as e1:
            print(f"❌ Device mapping failed: {str(e1)[:100]}...")
            
            # Approach 2: Try quantization
            try:
                print("\n🔄 Approach 2: Quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print("✅ Success with quantization!")
                
            except Exception as e2:
                print(f"❌ Quantization failed: {str(e2)[:100]}...")
                
                # Approach 3: Simple loading
                try:
                    print("\n🔄 Approach 3: Standard loading...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    print("✅ Success with standard loading!")
                    
                except Exception as e3:
                    print(f"❌ Standard loading failed: {str(e3)[:100]}...")
                    print("❌ All loading approaches failed")
                    return False
        
        if model is not None:
            # Test a simple generation
            print("\n🔬 Testing text generation...")
            test_prompt = "Supply chain management involves"
            
            inputs = tokenizer.encode(test_prompt, return_tensors='pt', max_length=50)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(f"✅ Generation successful: '{response.strip()}'")
            
            print(f"\n🎉 OpenAI GPT-OSS-20B is working correctly!")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading_approaches()
    
    if success:
        print("\n" + "="*60)
        print("✅ SUCCESS: Model loading works!")
        print("🚀 You can now use the AI features in the Streamlit app")
    else:
        print("\n" + "="*60)  
        print("❌ FAILED: Model loading issues detected")
        print("🔄 The app will use rule-based analysis")
    
    sys.exit(0 if success else 1) 