#!/usr/bin/env python3
"""
Simple Groq API Test Script for ONC AI Assistant
"""

import os
from groq import Groq

def test_groq_connection():
    """Test basic Groq API connection"""
    
    # Initialize Groq client
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        print("Please set it with: export GROQ_API_KEY='your_api_key_here'")
        return False
    
    try:
        client = Groq(api_key=api_key)
        print("Groq client initialized successfully")
        return client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return False

def simple_chat_test(client):
    """Test basic chat functionality"""
    
    print("\nTesting basic chat...")
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello! Can you help me with ocean data?"}
            ],
            model="llama3-8b-8192",  # Fast model for testing
            temperature=0.1,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content
        print(f"Chat test successful!")
        print(f"Response: {answer}")
        return True
        
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

def onc_specific_test(client):
    """Test with ONC-specific query"""
    
    print("\nTesting ONC-specific query...")
    
    system_prompt = """You are an AI assistant for Ocean Networks Canada (ONC). 
You help users understand oceanographic data and instruments. 
Keep responses concise and scientific."""
    
    user_query = "What is a CTD instrument and what does it measure?"
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=200
        )
        
        answer = response.choices[0].message.content
        print(f"ONC test successful!")
        print(f"Query: {user_query}")
        print(f"Response: {answer}")
        return True
        
    except Exception as e:
        print(f"ONC test failed: {e}")
        return False

def interactive_mode(client):
    """ONC Console Chat Mode"""
    
    system_prompt = """You are an AI assistant for Ocean Networks Canada (ONC), specializing in oceanographic data analysis and ocean monitoring systems.

Key Guidelines:
- Focus on Cambridge Bay Coastal Observatory data when relevant
- Explain oceanographic instruments (CTD, hydrophones, ice profilers, etc.)
- Provide scientific information about ocean conditions, temperature, salinity, ice, and marine life
- Discuss ONC's mission of ocean monitoring and research
- Always be accurate and cite when you're providing general oceanographic knowledge vs specific data
- If asked about current/real-time data, explain that you need access to live ONC APIs for the most current information

You help researchers, students, educators, and community members understand ocean science and ONC's monitoring capabilities."""
    
    while True:
        try:
            # Get user input with a nice prompt
            user_input = input("\nAsk about ocean data: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\nThanks for using ONC AI Assistant!")
                break
            
            if not user_input:
                print("Please ask a question about ocean data or type 'quit' to exit.")
                continue
            
            # Show thinking indicator
            print("Analyzing your ocean data question...")
            
            # Query the model
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                model="llama3-70b-8192",  # Use the better model for ONC queries
                temperature=0.1,
                max_tokens=500  # Allow longer responses for detailed explanations
            )
            
            answer = response.choices[0].message.content
            print(f"\nONC Assistant: {answer}")
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\nThanks for using ONC AI Assistant!")
            break
        except Exception as e:
            print(f"\nError processing your question: {e}")
            print("Please try asking again or check your internet connection.")

def main():
    """Main function - Direct to ONC console chat"""
    print("ONC AI Assistant - Console Chat")
    print("=" * 40)
    print("Ask questions about ocean data, Cambridge Bay, or oceanographic instruments")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    # Test connection
    client = test_groq_connection()
    if not client:
        return
    
    # Go straight to interactive mode
    interactive_mode(client)

if __name__ == "__main__":
    main()