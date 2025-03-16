import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer
)
from tokenizer_prep import GPT2Dataset
import json
from validation import validate_pipeline_step

class GPT2Trainer:
    def __init__(self, model_name: str = 'gpt2', output_dir: str = './fine_tuned_model'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize with error handling
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # Set up memory efficient training
            if torch.cuda.is_available():
                self.model = self.model.half()  # Use FP16 if on GPU
                
            os.makedirs(output_dir, exist_ok=True)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def prepare_training_args(self, 
                            num_train_epochs: int = 3,
                            batch_size: int = 4,
                            gradient_accumulation_steps: int = 4) -> TrainingArguments:
        """Prepare optimized training arguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=500,
            save_total_limit=2,
            learning_rate=5e-5,
            max_grad_norm=1.0,
            logging_steps=100,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            # Stability optimizations
            fp16=torch.cuda.is_available(),
            fp16_opt_level='O2',  # Mixed precision for better stability
            dataloader_num_workers=min(4, os.cpu_count() or 1),
            dataloader_pin_memory=True,
            gradient_checkpointing=True,  # Memory efficient training
            # Learning rate schedule
            warmup_steps=100,
            lr_scheduler_type="cosine",
            # Early stopping
            early_stopping_patience=3,
            # Additional stability settings
            ddp_find_unused_parameters=False,
            full_determinism=True,
            seed=42
        )

    def train(self, train_dataset: GPT2Dataset, training_args: TrainingArguments = None):
        """Train with proper error handling and memory management."""
        try:
            if training_args is None:
                training_args = self.prepare_training_args()

            # Memory cleanup and optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Enable gradient checkpointing for memory efficiency
                self.model.gradient_checkpointing_enable()

            # Create validation split with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, 
                [train_size, val_size],
                generator=generator
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )

            # Train with checkpoint recovery
            checkpoint = None
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) 
                             if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])

            trainer.train(resume_from_checkpoint=checkpoint)
            
            # Save with error handling
            try:
                trainer.save_model()
                self.tokenizer.save_pretrained(self.output_dir)
            except Exception as e:
                print(f"Warning: Error saving model: {str(e)}")
                # Try emergency save
                emergency_path = os.path.join(self.output_dir, "emergency_save")
                os.makedirs(emergency_path, exist_ok=True)
                trainer.save_model(emergency_path)

        except Exception as e:
            print(f"Training error: {str(e)}")
            # Try to save emergency checkpoint
            try:
                emergency_path = os.path.join(self.output_dir, "emergency_checkpoint")
                self.model.save_pretrained(emergency_path)
            except:
                print("Failed to save emergency checkpoint")
            raise

class GPT2Dataset(Dataset):
    def __init__(self, chunks_file: str, tokenizer: GPT2Tokenizer, max_length: int = 1024):
        """Initialize dataset with memory-efficient loading."""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.chunks = data['chunks'] if isinstance(data, dict) else data
                self.total_tokens = data.get('total_tokens', 0) if isinstance(data, dict) else 0
            
            self.tokenizer = tokenizer
            self.max_length = max_length
            
            # Validate dataset size
            if self.total_tokens > 1_000_000:  # Arbitrary limit for safety
                print(f"Warning: Large dataset detected ({self.total_tokens} tokens)")
                
        except Exception as e:
            print(f"Error initializing dataset: {str(e)}")
            self.chunks = []
            self.total_tokens = 0

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        try:
            chunk = self.chunks[idx]
            
            # Memory-efficient encoding
            encodings = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False  # Not needed for GPT-2
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
    try:
        input_dir = './tokenized_data'
        output_dir = './fine_tuned_model'
        chunks_file = os.path.join(input_dir, 'training_chunks.json')

        # Validate directories and files
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        if not os.path.exists(chunks_file):
            raise ValueError(f"Chunks file not found: {chunks_file}")
            
        # Validate chunks file format
        try:
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                if isinstance(chunks_data, dict):
                    if not chunks_data.get('chunks'):
                        raise ValueError("No chunks found in data file")
                    chunks = chunks_data['chunks']
                else:
                    chunks = chunks_data
                if not chunks:
                    raise ValueError("Empty chunks data")
        except json.JSONDecodeError:
            raise ValueError("Invalid chunks JSON format")

        # Initialize with proper error handling
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
            
        dataset = GPT2Dataset(chunks_file, tokenizer)
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
            
        # Validate CUDA availability if needed
        if torch.cuda.is_available():
            try:
                # Test CUDA memory allocation
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                raise RuntimeError(f"CUDA available but not working: {str(e)}")
            
        # Check dataset size and set batch size
        if dataset.total_tokens > 1_000_000:
            print("Warning: Large dataset detected. Using conservative batch size.")
            batch_size = 1
        else:
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                batch_size = max(1, min(4, int(total_memory / (1024**3) * 2)))
            else:
                batch_size = 1

        gradient_accumulation_steps = max(1, 16 // batch_size)

        # Initialize trainer
        trainer = GPT2Trainer(output_dir=output_dir)
        training_args = trainer.prepare_training_args(
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Validate output directory
        os.makedirs(output_dir, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory not writable: {output_dir}")
        
        # Start training
        trainer.train(dataset, training_args)
        
        # Validate saved model
        if not os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
            raise ValueError("Model file not found after training")
        if not os.path.exists(os.path.join(output_dir, 'config.json')):
            raise ValueError("Model config not found after training")

    except Exception as e:
        print(f"Fatal error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 