# GPT-2 Fine-tuning Pipeline

This project provides a complete pipeline for fine-tuning GPT-2 on text extracted from PDF books. The pipeline includes text extraction, preprocessing, tokenization, model fine-tuning, and text generation.

## Project Structure

```
.
├── pdfs/                      # Directory containing source PDF files
├── processed_texts/           # Directory for extracted and cleaned texts
├── tokenized_data/           # Directory for tokenized and chunked data
├── fine_tuned_model/        # Directory for the fine-tuned model
├── pdf_extractor.py         # Script for PDF text extraction
├── tokenizer_prep.py        # Script for text tokenization
├── train_gpt2.py           # Script for GPT-2 fine-tuning
├── generate_text.py        # Script for text generation
└── requirements.txt        # Python dependencies
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your PDF files in the `pdfs/` directory.

## Usage

### 1. Extract Text from PDFs
```bash
python pdf_extractor.py
```
This will:
- Extract text from all PDFs in the `pdfs/` directory
- Clean and preprocess the text
- Save the processed texts in `processed_texts/`

### 2. Prepare Dataset
```bash
python tokenizer_prep.py
```
This will:
- Load the processed texts
- Tokenize and chunk the text
- Save the prepared dataset in `tokenized_data/`

### 3. Fine-tune GPT-2
```bash
python train_gpt2.py
```
This will:
- Load the prepared dataset
- Fine-tune GPT-2 on the data
- Save the fine-tuned model in `fine_tuned_model/`

### 4. Generate Text
```bash
python generate_text.py --prompt "Your prompt here" --max_length 200 --num_sequences 1
```
Arguments:
- `--prompt`: Text prompt to generate from (required)
- `--max_length`: Maximum length of generated text (default: 200)
- `--num_sequences`: Number of sequences to generate (default: 1)
- `--temperature`: Temperature for text generation (default: 0.7)

## Notes

- The pipeline is configured for GPT-2 small by default. For larger models, adjust the batch size and other parameters accordingly.
- GPU is recommended for training, but the code will run on CPU if no GPU is available.
- The chunking process ensures that all training examples are of equal length (1024 tokens).
- Text generation parameters can be adjusted in `generate_text.py` for different creative outputs. 