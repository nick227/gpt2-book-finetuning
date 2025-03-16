import os
from pdfminer.high_level import extract_text
import re
from typing import List, Dict
import json
from validation import validate_pipeline_step
import fcntl
import time

class PDFProcessor:
    def __init__(self, pdf_dir: str, output_dir: str, max_file_size_mb: int = 100):
        """Initialize the PDF processor with input and output directories."""
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        os.makedirs(output_dir, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean text with enhanced validation."""
        # Remove common PDF artifacts
        text = re.sub(r'(\f|\r)', '\n', text)  # Form feeds and carriage returns
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Split into paragraphs and filter
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for p in paragraphs:
            p = p.strip()
            # Skip if too short or looks like a header/footer
            if len(p) < 30 or p.isupper() or re.match(r'^\d+$', p):
                continue
            cleaned_paragraphs.append(p)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF file with validation and error handling."""
        try:
            # Basic file validation
            if not os.path.exists(pdf_path):
                raise ValueError(f"PDF file not found: {pdf_path}")
            
            # Size and format validation
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                raise ValueError("PDF file is empty")
            if file_size > self.max_file_size:
                raise ValueError(f"PDF file too large: {file_size / (1024*1024):.2f}MB")
            
            # Check if file is actually a PDF
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'%PDF'):
                    raise ValueError("File is not a valid PDF")

            # Extract text with proper encoding handling
            raw_text = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    raw_text = extract_text(pdf_path, codec=encoding)
                    if raw_text and raw_text.strip():
                        break
                except Exception:
                    continue
            
            if not raw_text or not raw_text.strip():
                raise ValueError("Failed to extract text with any encoding")

            # Basic content validation
            if len(raw_text.split()) < 50:  # Suspiciously short
                raise ValueError("Extracted text too short - possible extraction failure")

            cleaned_text = self.clean_text(raw_text)
            if not cleaned_text.strip():
                raise ValueError("Cleaning resulted in empty text")

            return {
                'filename': os.path.basename(pdf_path),
                'text': cleaned_text,
                'original_length': len(raw_text),
                'cleaned_length': len(cleaned_text),
                'encoding_used': encoding
            }

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs in the input directory."""
        processed_texts = []
        
        for filename in os.listdir(self.pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_dir, filename)
                result = self.process_pdf(pdf_path)
                if result:
                    processed_texts.append(result)
        
        return processed_texts
    
    def save_processed_texts(self, processed_texts: List[Dict[str, str]]):
        """Save processed texts with file locking."""
        for item in processed_texts:
            output_path = os.path.join(self.output_dir, 
                                     f"{os.path.splitext(item['filename'])[0]}.txt")
            
            # Use file locking for thread/process safety
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        f.write(item['text'])
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    break
                except IOError as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise e
        
        # Save metadata with locking
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        metadata = [{
            'filename': item['filename'],
            'output_file': f"{os.path.splitext(item['filename'])[0]}.txt",
            'original_length': item['original_length'],
            'cleaned_length': item['cleaned_length'],
            'encoding_used': item['encoding_used']
        } for item in processed_texts]
        
        with open(metadata_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(metadata, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def main():
    # Define directories
    pdf_dir = './pdfs'
    output_dir = './processed_texts'
    
    # Validate directories
    if not os.path.exists(pdf_dir):
        raise ValueError(f"PDF directory not found: {pdf_dir}")
    
    if not any(f.endswith('.pdf') for f in os.listdir(pdf_dir)):
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    # Initialize processor
    processor = PDFProcessor(pdf_dir, output_dir)
    
    # Process PDFs
    print("Processing PDFs...")
    processed_texts = processor.process_all_pdfs()
    
    if not processed_texts:
        raise ValueError("No texts were successfully processed")
    
    # Save results
    print("Saving processed texts...")
    processor.save_processed_texts(processed_texts)
    
    print(f"Processing complete. Processed {len(processed_texts)} files.")

if __name__ == "__main__":
    main() 