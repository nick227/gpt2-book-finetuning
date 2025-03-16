import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
from typing import Dict, List

class TextGenerator:
    def __init__(self, model_path: str):
        """Initialize the text generator with the fine-tuned model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the fine-tuned model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        
        print(f"Model loaded on {self.device}")
        
    def generate(self,
                prompt: str,
                max_length: int = 200,
                num_return_sequences: int = 1,
                temperature: float = 0.7,
                top_k: int = 50,
                top_p: float = 0.95) -> List[str]:
        """Generate text based on the provided prompt."""
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            'input_ids': inputs,
            'max_length': max_length,
            'num_return_sequences': num_return_sequences,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'pad_token_id': self.tokenizer.eos_token_id,
            'do_sample': True,
            'no_repeat_ngram_size': 2
        }
        
        # Generate text
        outputs = self.model.generate(**gen_kwargs)
        
        # Decode and return the generated texts
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts

def main():
    parser = argparse.ArgumentParser(description='Generate text using fine-tuned GPT-2')
    parser.add_argument('--model_path', type=str, default='./fine_tuned_model',
                       help='Path to the fine-tuned model')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt to generate from')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximum length of generated text')
    parser.add_argument('--num_sequences', type=int, default=1,
                       help='Number of sequences to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for text generation')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(args.model_path)
    
    # Generate text
    generated_texts = generator.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
        temperature=args.temperature
    )
    
    # Print generated texts
    print("\nGenerated texts:")
    for i, text in enumerate(generated_texts, 1):
        print(f"\nSequence {i}:")
        print(text)
        print("-" * 50)

if __name__ == "__main__":
    main() 