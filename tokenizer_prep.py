import os
import json
from typing import List, Dict, Generator
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset
import numpy as np
from validation import validate_pipeline_step

class TextDatasetPrep:
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 1024, batch_size: int = 1000):
        """Initialize the dataset preparation with directories and parameters."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Add padding token to tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        os.makedirs(output_dir, exist_ok=True)
        
    def text_generator(self, text: str, overlap_tokens: int = 50) -> Generator[str, None, None]:
        """Generate text chunks with overlap for better coherence."""
        words = text.split()
        current_chunk = []
        current_length = 0
        overlap_buffer = []  # Store overlapping tokens
        
        for word in words:
            word_tokens = len(self.tokenizer.encode(word))
            
            if current_length + word_tokens > self.chunk_size:
                if current_chunk:
                    # Create chunk with proper ending
                    chunk_text = ' '.join(current_chunk)
                    if not any(chunk_text.rstrip().endswith(p) for p in '.!?'):
                        chunk_text += '.'
                    yield chunk_text
                    
                    # Keep overlap_tokens number of tokens for next chunk
                    overlap_start = max(0, len(current_chunk) - overlap_tokens)
                    overlap_buffer = current_chunk[overlap_start:]
                    
                    # Start new chunk with overlap
                    current_chunk = overlap_buffer + [word]
                    current_length = sum(len(self.tokenizer.encode(w)) for w in current_chunk)
                else:
                    current_chunk = [word]
                    current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens
        
        # Handle the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if not any(chunk_text.rstrip().endswith(p) for p in '.!?'):
                chunk_text += '.'
            yield chunk_text

    def prepare_dataset(self) -> int:
        """Prepare dataset with validation and memory efficiency."""
        try:
            metadata_path = os.path.join(self.input_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            all_chunks = []
            total_tokens = 0
            max_tokens = 1_000_000  # Limit total tokens to prevent memory issues
            
            for item in metadata:
                if total_tokens >= max_tokens:
                    print(f"Reached token limit ({max_tokens}). Stopping processing.")
                    break
                    
                text_path = os.path.join(self.input_dir, item['output_file'])
                
                # Process text in batches
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Validate text before processing
                text_validation = validate_pipeline_step("text_cleaning", text)
                if not text_validation.get("results", {}).get("has_content", False):
                    print(f"Skipping invalid text file: {text_path}")
                    continue

                # Process text in memory-efficient way
                batch_chunks = []
                for chunk in self.text_generator(text):
                    # Validate chunk
                    chunk_validation = validate_pipeline_step("chunk_validation", (chunk, self.tokenizer))
                    if chunk_validation.get("results", {}).get("valid_length", False):
                        batch_chunks.append(chunk)
                        total_tokens += len(self.tokenizer.encode(chunk))
                    
                    # Save batch if it reaches the size limit
                    if len(batch_chunks) >= self.batch_size:
                        all_chunks.extend(batch_chunks)
                        batch_chunks = []
                        
                        # Check token limit
                        if total_tokens >= max_tokens:
                            print(f"Reached token limit ({max_tokens}). Stopping processing.")
                            break
                
                # Add remaining chunks
                if batch_chunks and total_tokens < max_tokens:
                    all_chunks.extend(batch_chunks)

            # Save chunks with token count
            output_path = os.path.join(self.output_dir, 'training_chunks.json')
            metadata = {
                'chunks': all_chunks,
                'total_tokens': total_tokens,
                'chunk_size': self.chunk_size,
                'num_chunks': len(all_chunks)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f)

            return len(all_chunks)

        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            return 0

class GPT2Dataset(Dataset):
    def __init__(self, chunks_file: str, tokenizer: GPT2Tokenizer, max_length: int = 1024):
        """Initialize the dataset with chunks and tokenizer."""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.tokenizer = tokenizer
            self.max_length = max_length
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            self.chunks = []
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        try:
            chunk = self.chunks[idx]
            encodings = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze()
            }
        except Exception as e:
            print(f"Error processing chunk {idx}: {str(e)}")
            # Return empty tensors as fallback
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

def main():
    # Define directories
    input_dir = './processed_texts'
    output_dir = './tokenized_data'
    
    # Validate input directory and files
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
        
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"Metadata file not found: {metadata_path}")
        
    # Validate metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if not metadata or not isinstance(metadata, list):
                raise ValueError("Invalid metadata format")
            
            # Check if all referenced files exist
            for item in metadata:
                text_path = os.path.join(input_dir, item['output_file'])
                if not os.path.exists(text_path):
                    raise ValueError(f"Referenced text file not found: {text_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid metadata JSON format")
    
    # Initialize and run dataset preparation
    prep = TextDatasetPrep(input_dir, output_dir)
    num_chunks = prep.prepare_dataset()
    
    if num_chunks == 0:
        raise ValueError("No valid chunks were created")
        
    # Validate output
    chunks_path = os.path.join(output_dir, 'training_chunks.json')
    if not os.path.exists(chunks_path):
        raise ValueError("Failed to create chunks file")
        
    try:
        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)
            if isinstance(chunks_data, dict):
                if not chunks_data.get('chunks'):
                    raise ValueError("No chunks in output file")
            elif not chunks_data:
                raise ValueError("Empty chunks file")
    except json.JSONDecodeError:
        raise ValueError("Invalid chunks JSON format")
    
    print(f"Dataset preparation complete. Created {num_chunks} chunks.")

if __name__ == "__main__":
    main() 