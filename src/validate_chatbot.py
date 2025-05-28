"""
Test script for validating the customer support chatbot.
This script tests the chatbot's responsiveness and accuracy with various queries.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the simulated model
from src.optimize_model import SimulatedFineTunedModel

# Define test queries covering different intents
TEST_QUERIES = [
    # Order related
    "How do I cancel my order?",
    "Where is my package?",
    "I want to track my order #12345",
    "Can I change my order after placing it?",
    
    # Account related
    "How do I reset my password?",
    "I need to update my account information",
    "How can I delete my account?",
    "I'm having trouble logging in",
    
    # Payment related
    "My payment was declined",
    "What payment methods do you accept?",
    "I was charged twice for my order",
    "When will I receive my refund?",
    
    # Contact related
    "How can I speak to a human agent?",
    "What's your customer service phone number?",
    "I need to talk to someone about my order",
    
    # Miscellaneous
    "I received the wrong item",
    "The product I received is damaged",
    "How do I return an item?",
    "What is your shipping policy?"
]

def test_chatbot_responses():
    """Test the chatbot's responses to various queries."""
    print("=== Testing Customer Support Chatbot ===")
    
    # Initialize the model
    print("Initializing model...")
    model = SimulatedFineTunedModel()
    
    # Create results directory
    results_dir = os.path.join(project_root, "validation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Test with template-based responses
    print("\n=== Testing Template-Based Responses ===")
    template_results = []
    template_times = []
    
    for query in TEST_QUERIES:
        print(f"\nQuery: {query}")
        
        # Measure response time
        start_time = time.time()
        response = model.generate_response(query, use_templates=True)
        response_time = time.time() - start_time
        
        print(f"Response: {response}")
        print(f"Response time: {response_time:.4f} seconds")
        
        template_results.append({
            "query": query,
            "response": response,
            "response_time": response_time
        })
        
        template_times.append(response_time)
    
    # Test with model-based responses
    print("\n=== Testing Model-Based Responses ===")
    model_results = []
    model_times = []
    
    for query in TEST_QUERIES[:5]:  # Test fewer queries for model-based responses
        print(f"\nQuery: {query}")
        
        # Measure response time
        start_time = time.time()
        response = model.generate_response(query, use_templates=False)
        response_time = time.time() - start_time
        
        print(f"Response: {response}")
        print(f"Response time: {response_time:.4f} seconds")
        
        model_results.append({
            "query": query,
            "response": response,
            "response_time": response_time
        })
        
        model_times.append(response_time)
    
    # Calculate average response times
    avg_template_time = sum(template_times) / len(template_times)
    avg_model_time = sum(model_times) / len(model_times)
    
    print("\n=== Summary ===")
    print(f"Average template-based response time: {avg_template_time:.4f} seconds")
    print(f"Average model-based response time: {avg_model_time:.4f} seconds")
    print(f"Speedup factor: {avg_model_time / avg_template_time:.2f}x")
    
    # Save results
    results = {
        "template_results": template_results,
        "model_results": model_results,
        "avg_template_time": avg_template_time,
        "avg_model_time": avg_model_time,
        "speedup_factor": avg_model_time / avg_template_time
    }
    
    with open(os.path.join(results_dir, "validation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {os.path.join(results_dir, 'validation_results.json')}")

if __name__ == "__main__":
    test_chatbot_responses()
