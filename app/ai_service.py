from transformers import pipeline, set_seed
import torch

class HuggingFaceService:
    def __init__(self):
        print("Loading GPT-2 model...")
        device = -1  # Use CPU
        self.generator = pipeline(
            "text-generation",
            model="gpt2",
            device=device
        )
        set_seed(42)
        print("Model loaded successfully!")
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.9):
        try:
            result = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=50256
            )
            return {
                "generated_text": result[0]["generated_text"],
                "model": "gpt2",
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")