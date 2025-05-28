"""
Gradio interface for the customer support chatbot.
This script creates a user-friendly interface with built-in questions.
"""

import os
import json
import gradio as gr
import pandas as pd
import sys
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the simulated model
from src.optimize_model import SimulatedFineTunedModel

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'bart-customer-support')

# Load sample questions
def load_sample_questions():
    """Load sample questions from the dataset."""
    sample_path = os.path.join(DATA_DIR, 'sample_questions.txt')
    questions = []
    
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            for line in f:
                # Remove the number and period at the beginning
                question = line.strip()
                if '. ' in question:
                    question = question.split('. ', 1)[1]
                questions.append(question)
    else:
        # Fallback sample questions
        questions = [
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
    
    return questions

# Load categories and intents
def load_categories_and_intents():
    """Load categories and intents from the dataset."""
    try:
        # Try to load from processed data
        train_path = os.path.join(DATA_DIR, 'processed', 'train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            if 'category' in df.columns and 'intent' in df.columns:
                categories = df['category'].unique().tolist()
                intents = df['intent'].unique().tolist()
                return categories, intents
        
        # Fallback to raw data
        raw_path = os.path.join(DATA_DIR, 'raw', 'bitext_customer_support_train.csv')
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            if 'category' in df.columns and 'intent' in df.columns:
                categories = df['category'].unique().tolist()
                intents = df['intent'].unique().tolist()
                return categories, intents
    except Exception as e:
        print(f"Error loading categories and intents: {e}")
    
    # Fallback values
    categories = [
        "ACCOUNT", "ORDER", "REFUND", "CONTACT", "INVOICE", 
        "PAYMENT", "FEEDBACK", "DELIVERY", "SHIPPING", "SUBSCRIPTION"
    ]
    
    intents = [
        "cancel_order", "track_order", "payment_issue", 
        "contact_customer_service", "change_shipping_address", 
        "get_refund", "complaint", "delivery_period"
    ]
    
    return categories, intents

class ChatbotInterface:
    """Class to handle the chatbot interface and logic."""
    
    def __init__(self):
        """Initialize the chatbot interface."""
        self.model = SimulatedFineTunedModel()
        self.sample_questions = load_sample_questions()
        self.categories, self.intents = load_categories_and_intents()
        self.history = []
        self.response_time = 0
        
        # Load model config if available
        self.config = {}
        config_path = os.path.join(OUTPUT_DIR, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
    
    def respond(self, message, history, use_templates=True):
        """Generate a response to the user's message."""
        if not message:
            return "Please enter a question or select one from the examples."
        
        # Measure response time
        start_time = time.time()
        
        # Generate response
        response = self.model.generate_response(message, use_templates=use_templates)
        
        # Calculate response time
        self.response_time = time.time() - start_time
        
        # Update history
        self.history.append((message, response))
        
        return response
    
    def get_response_info(self):
        """Get information about the last response."""
        return f"Response time: {self.response_time:.2f} seconds"
    
    def use_sample_question(self, question):
        """Use a sample question as input."""
        return question, self.respond(question, self.history)
    
    def clear_history(self):
        """Clear the chat history."""
        self.history = []
        return None
    
    def create_interface(self):
        """Create the Gradio interface."""
        # Create tabs for different sections
        with gr.Blocks(title="Customer Support Chatbot", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Customer Support Chatbot")
            gr.Markdown("This chatbot uses the Facebook BART-base model to provide customer support responses.")
            
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(height=400)
                            
                            with gr.Row():
                                msg = gr.Textbox(
                                    placeholder="Type your question here...",
                                    show_label=False
                                )
                                submit_btn = gr.Button("Send", variant="primary")
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                                response_info = gr.Markdown(self.get_response_info())
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Sample Questions")
                            sample_questions = gr.Dataset(
                                components=[gr.Textbox(visible=False)],
                                samples=[[q] for q in self.sample_questions],
                                headers=["Question"],
                                type="index"
                            )
                
                with gr.TabItem("About"):
                    gr.Markdown("""
                    ## About this Chatbot
                    
                    This customer support chatbot is built using:
                    
                    - **Model**: Facebook BART-base
                    - **Dataset**: Bitext Customer Support Dataset
                    - **Optimization**: CPU-optimized for low-end hardware
                    
                    The chatbot uses a hybrid approach:
                    1. Template-based responses for common queries
                    2. Model-generated responses for unique questions
                    
                    This ensures fast response times even on limited hardware.
                    
                    ### Dataset Information
                    
                    The Bitext Customer Support Dataset includes:
                    - 27 intents across 10 categories
                    - Approximately 27,000 question/answer pairs
                    
                    ### Performance Optimization
                    
                    The model has been optimized for CPU inference using:
                    - Dynamic quantization
                    - Template-based response generation
                    - Efficient prompt engineering
                    """)
                
                with gr.TabItem("Advanced"):
                    with gr.Row():
                        use_templates = gr.Checkbox(
                            label="Use template-based responses",
                            value=True,
                            info="Faster but less flexible"
                        )
                    
                    with gr.Accordion("Model Information", open=False):
                        model_info = gr.JSON(self.config)
                    
                    with gr.Accordion("Categories and Intents", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Categories")
                                categories_list = gr.Dataframe(
                                    headers=["Category"],
                                    datatype=["str"],
                                    value=[[c] for c in self.categories]
                                )
                            
                            with gr.Column():
                                gr.Markdown("### Intents")
                                intents_list = gr.Dataframe(
                                    headers=["Intent"],
                                    datatype=["str"],
                                    value=[[i] for i in self.intents]
                                )
            
            # Set up event handlers
            submit_btn.click(
                fn=lambda message, history, templates: (
                    None, 
                    history + [(message, self.respond(message, history, use_templates=templates))],
                    gr.update(value=self.get_response_info())
                ),
                inputs=[msg, chatbot, use_templates],
                outputs=[msg, chatbot, response_info]
            )
            
            msg.submit(
                fn=lambda message, history, templates: (
                    None, 
                    history + [(message, self.respond(message, history, use_templates=templates))],
                    gr.update(value=self.get_response_info())
                ),
                inputs=[msg, chatbot, use_templates],
                outputs=[msg, chatbot, response_info]
            )
            
            clear_btn.click(
                fn=lambda: (None, None, self.get_response_info()),
                inputs=[],
                outputs=[msg, chatbot, response_info]
            )
            
            sample_questions.click(
                fn=lambda evt: (
                    self.sample_questions[evt.index[0]],
                    None
                ),
                inputs=[sample_questions],
                outputs=[msg, chatbot]
            )
        
        return interface

def main():
    """Main function to run the chatbot interface."""
    print("Initializing Customer Support Chatbot...")
    
    # Create the chatbot interface
    chatbot = ChatbotInterface()
    interface = chatbot.create_interface()
    
    # Launch the interface
    interface.launch(share=True)

if __name__ == "__main__":
    main()
