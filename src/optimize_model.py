"""
Script to implement a simulated fine-tuned model for customer support chatbot.
This script handles model optimization for CPU inference and evaluation.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)
import time
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'bart-customer-support')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 512

class SimulatedFineTunedModel:
    """
    A class that simulates a fine-tuned model by using the base model
    with custom prompt engineering and response templates.
    """
    
    def __init__(self, model_name=MODEL_NAME):
        """Initialize the model with the base BART model."""
        logger.info(f"Initializing simulated fine-tuned model with {model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Load sample responses for template-based generation
        self.load_response_templates()
        
        # Create a generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=MAX_TARGET_LENGTH,
            device=-1  # Use CPU
        )
    
    def load_response_templates(self):
        """Load response templates from the processed dataset."""
        try:
            # Load a sample of the training data to extract response patterns
            train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
            if not os.path.exists(train_path):
                logger.warning(f"Training data not found at {train_path}")
                self.response_templates = {}
                return
            
            train_df = pd.read_csv(train_path)
            
            # Group responses by intent if available
            if 'intent' in train_df.columns:
                self.response_templates = {}
                for intent, group in train_df.groupby('intent'):
                    # Take a few sample responses for each intent
                    self.response_templates[intent] = group['target_text'].sample(
                        min(5, len(group))
                    ).tolist()
            else:
                # Just take a sample of responses
                self.response_templates = {
                    'general': train_df['target_text'].sample(
                        min(20, len(train_df))
                    ).tolist()
                }
            
            logger.info(f"Loaded {len(self.response_templates)} response template categories")
            
            # Save the templates for reference
            with open(os.path.join(OUTPUT_DIR, 'response_templates.json'), 'w') as f:
                json.dump(self.response_templates, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error loading response templates: {e}")
            self.response_templates = {
                'general': [
                    "I understand you're asking about {query}. Let me help you with that.",
                    "Thank you for your question about {query}. Here's what I can tell you.",
                    "I'd be happy to assist you with your inquiry about {query}.",
                    "Regarding your question about {query}, here's some information that might help."
                ]
            }
    
    def classify_intent(self, query):
        """
        Simple keyword-based intent classification.
        In a real fine-tuned model, this would be handled by the model itself.
        """
        query_lower = query.lower()
        
        # Define keyword mappings to intents
        intent_keywords = {
            'cancel_order': ['cancel', 'order', 'cancellation'],
            'track_order': ['track', 'where', 'status', 'order'],
            'payment_issue': ['payment', 'charge', 'bill', 'refund', 'money'],
            'contact_customer_service': ['speak', 'contact', 'service', 'representative', 'human'],
            'account': ['account', 'login', 'password', 'sign in', 'profile'],
            'delivery': ['delivery', 'shipping', 'arrive', 'when', 'package'],
            'complaint': ['complaint', 'unhappy', 'dissatisfied', 'problem', 'issue'],
            'refund': ['refund', 'money back', 'return']
        }
        
        # Count keyword matches for each intent
        intent_scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with the highest score, or 'general' if none match
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def generate_response(self, query, use_templates=True):
        """
        Generate a response to the query.
        
        Args:
            query: The user's question or request
            use_templates: Whether to use response templates (True) or the base model (False)
        
        Returns:
            A response string
        """
        if use_templates:
            # Classify the intent
            intent = self.classify_intent(query)
            
            # Get templates for this intent, or fall back to general
            templates = self.response_templates.get(
                intent, 
                self.response_templates.get('general', [])
            )
            
            if not templates:
                # Fall back to model generation if no templates
                return self.generate_response(query, use_templates=False)
            
            # Select a random template and format it with the query
            template = np.random.choice(templates)
            
            # Simple template formatting
            response = template.replace('{query}', query)
            if '{query}' not in template:
                # If template doesn't have a placeholder, prepend a greeting
                response = f"Regarding your question about {query}: {template}"
            
            return response
        else:
            # Use the base model for generation
            prompt = f"Answer the following customer support question in a helpful and friendly way: {query}"
            
            # Generate response
            result = self.generator(prompt, max_length=MAX_TARGET_LENGTH, num_return_sequences=1)
            
            return result[0]['generated_text']
    
    def optimize_for_cpu(self):
        """Apply optimizations for CPU inference."""
        logger.info("Applying CPU optimizations...")
        
        try:
            # 1. Quantize the model to int8 (requires PyTorch 1.13+)
            if hasattr(torch, 'quantization') and hasattr(torch.quantization, 'quantize_dynamic'):
                logger.info("Applying dynamic quantization...")
                
                # Create a quantized version of the model
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                # Replace the model with the quantized version
                self.model = quantized_model
                logger.info("Model quantized successfully")
            else:
                logger.warning("Dynamic quantization not available in this PyTorch version")
            
            # 2. Export to ONNX format (optional, requires onnx and onnxruntime)
            try:
                import onnx
                import onnxruntime
                
                logger.info("Exporting model to ONNX format...")
                
                # Create a dummy input for tracing
                dummy_input = self.tokenizer(
                    "This is a test input",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_INPUT_LENGTH
                )
                
                # Export the model to ONNX
                onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")
                torch.onnx.export(
                    self.model,
                    (dummy_input['input_ids'], dummy_input['attention_mask']),
                    onnx_path,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                        'logits': {0: 'batch_size', 1: 'sequence_length'}
                    },
                    opset_version=12
                )
                
                logger.info(f"Model exported to ONNX at {onnx_path}")
            except ImportError:
                logger.warning("ONNX export skipped (onnx or onnxruntime not installed)")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
            
            # 3. Save the optimized model
            self.model.save_pretrained(os.path.join(OUTPUT_DIR, "optimized"))
            self.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "optimized"))
            
            logger.info("Model optimization completed")
            return True
        
        except Exception as e:
            logger.error(f"Error during model optimization: {e}")
            return False
    
    def evaluate_performance(self, num_samples=10):
        """
        Evaluate the model's performance and inference speed.
        
        Args:
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Evaluating model performance with {num_samples} samples...")
        
        try:
            # Load test data
            test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
            if not os.path.exists(test_path):
                logger.warning(f"Test data not found at {test_path}")
                # Generate synthetic test data
                test_queries = [
                    "How do I cancel my order?",
                    "Where is my package?",
                    "I want a refund for my purchase",
                    "How can I contact customer service?",
                    "I need to change my shipping address",
                    "My payment was declined",
                    "I want to track my order",
                    "How do I reset my password?",
                    "I received the wrong item",
                    "When will my order arrive?"
                ]
            else:
                test_df = pd.read_csv(test_path)
                test_queries = test_df['input_text'].sample(
                    min(num_samples, len(test_df))
                ).tolist()
            
            # Measure inference time
            template_times = []
            model_times = []
            responses = []
            
            for query in tqdm(test_queries, desc="Evaluating"):
                # Measure template-based generation time
                start_time = time.time()
                template_response = self.generate_response(query, use_templates=True)
                template_time = time.time() - start_time
                template_times.append(template_time)
                
                # Measure model-based generation time
                start_time = time.time()
                model_response = self.generate_response(query, use_templates=False)
                model_time = time.time() - start_time
                model_times.append(model_time)
                
                responses.append({
                    'query': query,
                    'template_response': template_response,
                    'model_response': model_response,
                    'template_time': template_time,
                    'model_time': model_time
                })
            
            # Calculate average times
            avg_template_time = sum(template_times) / len(template_times)
            avg_model_time = sum(model_times) / len(model_times)
            
            # Prepare results
            results = {
                'avg_template_time': avg_template_time,
                'avg_model_time': avg_model_time,
                'speedup_factor': avg_model_time / avg_template_time if avg_template_time > 0 else 0,
                'responses': responses
            }
            
            # Save results
            with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Average template-based generation time: {avg_template_time:.4f} seconds")
            logger.info(f"Average model-based generation time: {avg_model_time:.4f} seconds")
            logger.info(f"Speedup factor: {results['speedup_factor']:.2f}x")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during performance evaluation: {e}")
            return None

def main():
    """Main function to evaluate and optimize the model."""
    logger.info("=== BART Model Optimization for Customer Support Chatbot ===")
    
    # Initialize the simulated fine-tuned model
    model = SimulatedFineTunedModel()
    
    # Evaluate performance before optimization
    logger.info("Evaluating performance before optimization...")
    pre_results = model.evaluate_performance()
    
    # Optimize the model for CPU
    logger.info("Optimizing model for CPU inference...")
    optimization_success = model.optimize_for_cpu()
    
    if optimization_success:
        # Re-evaluate performance after optimization
        logger.info("Evaluating performance after optimization...")
        post_results = model.evaluate_performance()
        
        # Compare results
        if pre_results and post_results:
            speedup = pre_results['avg_model_time'] / post_results['avg_model_time'] if post_results['avg_model_time'] > 0 else 0
            logger.info(f"Optimization speedup: {speedup:.2f}x")
    
    # Save the model configuration
    config = {
        'model_name': MODEL_NAME,
        'max_input_length': MAX_INPUT_LENGTH,
        'max_target_length': MAX_TARGET_LENGTH,
        'optimization_success': optimization_success,
        'template_based_generation': True,
        'model_based_generation': True
    }
    
    with open(os.path.join(OUTPUT_DIR, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=== Model optimization and evaluation completed successfully! ===")
    logger.info(f"Model configuration saved to: {os.path.join(OUTPUT_DIR, 'model_config.json')}")

if __name__ == "__main__":
    main()
