import ollama

def generate_response(prompt, base_url='http://127.0.0.1:11434', model='qwen2.5:1.5b'):
    """Generates a response using a locally hosted model with ollama1.invoke."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response."
