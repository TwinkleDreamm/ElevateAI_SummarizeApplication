#!/usr/bin/env python3
"""
Test script for LangchainLLMClient with custom base URL support.
Run this to verify the client configuration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_langchain_client():
    """Test LangchainLLMClient configuration."""
    try:
        from src.ai.langchain.llm_client import LangchainLLMClient
        from config.settings import settings
        
        print("=== LangchainLLMClient Test ===")
        print(f"OpenAI API Key: {'Configured' if settings.openai_api_key else 'Not configured'}")
        print(f"OpenAI Base URL: {settings.openai_base_url}")
        print(f"OpenAI Chat Model: {settings.openai_chat_model}")
        
        # Test client initialization
        client = LangchainLLMClient()
        
        # Get client info
        info = client.get_client_info()
        print("\n=== Client Configuration ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test connection
        print("\n=== Testing Connection ===")
        connection_test = client.test_connection()
        if connection_test["success"]:
            print(f"✅ Connection successful: {connection_test['response']}")
            print(f"Model: {connection_test['model']}")
        else:
            print(f"❌ Connection failed: {connection_test['error']}")
        
        # Test settings update mechanism
        print("\n=== Testing Settings Update ===")
        print("Current settings:")
        settings_check = client.check_settings_applied()
        print(f"Settings applied: {settings_check['settings_applied']}")
        if settings_check['mismatches']:
            print("Mismatches found:")
            for key, mismatch in settings_check['mismatches'].items():
                print(f"  {key}: current={mismatch['current']}, expected={mismatch['expected']}")
        
        # Debug configuration state
        print("\n=== Debug Configuration State ===")
        debug_info = client.debug_config()
        print("Self config:", debug_info['self_config'])
        print("Settings values:", debug_info['settings_values'])
        print("Default config:", debug_info['default_config'])
        print("LLM config:", debug_info['llm_config'])
        
        # Test updating settings
        print("\nUpdating temperature to 0.9...")
        client.update_config({"temperature": 0.9})
        
        print("Settings after update:")
        settings_check_after = client.check_settings_applied()
        print(f"Settings applied: {settings_check_after['settings_applied']}")
        if settings_check_after['mismatches']:
            print("Mismatches found:")
            for key, mismatch in settings_check_after['mismatches'].items():
                print(f"  {key}: current={mismatch['current']}, expected={mismatch['expected']}")
        
        # Debug configuration state after update
        print("\n=== Debug Configuration State After Update ===")
        debug_info_after = client.debug_config()
        print("Self config after update:", debug_info_after['self_config'])
        print("LLM config after update:", debug_info_after['llm_config'])
        
        # Test message conversion
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("\n=== Testing Message Conversion ===")
        converted = client._to_langchain_messages(test_messages)
        print(f"Converted {len(converted)} messages")
        
        # Test response generation (if API key is configured)
        if info["api_key_configured"] and info["llm_initialized"]:
            print("\n=== Testing Response Generation ===")
            try:
                response = client.generate_response(test_messages)
                print(f"Response: {response['content'][:100]}...")
                print(f"Model: {response['model']}")
                print(f"Finish Reason: {response['finish_reason']}")
            except Exception as e:
                print(f"Response generation failed: {e}")
        else:
            print("\n=== Skipping Response Generation ===")
            if not info["api_key_configured"]:
                print("Reason: No API key configured")
            if not info["llm_initialized"]:
                print("Reason: LLM not initialized")
        
        print("\n=== Test Completed Successfully ===")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install langchain langchain-openai")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_langchain_client()
