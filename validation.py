import os
from typing import Dict, List, Optional
import torch
from transformers import GPT2Tokenizer

class PipelineValidator:
    @staticmethod
    def validate_pdf(file_path: str) -> bool:
        """Validate PDF file integrity."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'%PDF'):
                    return False
                # Check file size
                f.seek(0, 2)
                if f.tell() < 100:  # Suspiciously small
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def validate_extracted_text(text: str) -> Dict[str, bool]:
        """Validate extracted text quality."""
        results = {
            'has_content': bool(text.strip()),
            'reasonable_length': len(text) > 100,
            'has_sentences': '.' in text or '!' in text or '?' in text,
            'valid_characters': all(ord(c) < 128 for c in text)  # Basic ASCII check
        }
        return results

    @staticmethod
    def validate_chunk(chunk: str, tokenizer: GPT2Tokenizer) -> Dict[str, bool]:
        """Validate text chunk quality."""
        tokens = tokenizer.encode(chunk)
        results = {
            'valid_length': 100 <= len(tokens) <= 1024,
            'complete_sentences': chunk.strip()[-1] in '.!?',
            'valid_tokens': all(t in tokenizer.get_vocab() for t in tokenizer.encode(chunk))
        }
        return results

    @staticmethod
    def validate_model_output(text: str) -> Dict[str, bool]:
        """Validate model-generated text."""
        results = {
            'has_content': bool(text.strip()),
            'reasonable_length': 10 <= len(text.split()) <= 1000,
            'complete_sentences': text.strip()[-1] in '.!?',
            'valid_characters': all(ord(c) < 128 for c in text)
        }
        return results

def validate_pipeline_step(step_name: str, data: any) -> Dict[str, any]:
    """Validate individual pipeline steps."""
    validator = PipelineValidator()
    
    if step_name == "pdf_extraction":
        if not isinstance(data, str) or not os.path.exists(data):
            return {"valid": False, "error": "Invalid PDF path"}
        return {"valid": validator.validate_pdf(data)}
    
    elif step_name == "text_cleaning":
        if not isinstance(data, str):
            return {"valid": False, "error": "Invalid text data"}
        return {"valid": True, "results": validator.validate_extracted_text(data)}
    
    elif step_name == "chunk_validation":
        if not isinstance(data, tuple) or len(data) != 2:
            return {"valid": False, "error": "Invalid chunk data"}
        chunk, tokenizer = data
        return {"valid": True, "results": validator.validate_chunk(chunk, tokenizer)}
    
    elif step_name == "model_output":
        if not isinstance(data, str):
            return {"valid": False, "error": "Invalid model output"}
        return {"valid": True, "results": validator.validate_model_output(data)}
    
    return {"valid": False, "error": "Unknown validation step"} 