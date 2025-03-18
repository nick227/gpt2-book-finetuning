import os
import json
from typing import Dict, Any
import torch
from genetic_optimizer import GeneticAlgorithm
from train_gpt2 import GPT2Trainer
from tokenizer_prep import GPT2Dataset
from transformers import GPT2Tokenizer
import numpy as np

class HyperparameterOptimizer:
    def __init__(
        self,
        input_dir: str = './tokenized_data',
        output_dir: str = './optimized_model',
        population_size: int = 30,
        num_generations: int = 10,
        checkpoint_dir: str = './optimization_checkpoints'
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.population_size = population_size
        self.num_generations = num_generations
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize genetic algorithm
        self.ga = GeneticAlgorithm(population_size=population_size)
        
        # Setup directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def evaluate_individual(self, individual: Dict[str, float], dataset: GPT2Dataset) -> float:
        """Evaluate a single set of hyperparameters."""
        try:
            # Create a temporary directory for this evaluation
            temp_dir = os.path.join(self.output_dir, f'temp_{hash(str(individual))}')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize trainer with the hyperparameters
            trainer = GPT2Trainer(output_dir=temp_dir)
            
            # Prepare training arguments
            training_args = trainer.prepare_training_args(
                batch_size=int(individual['batch_size']),
                num_train_epochs=int(individual['num_epochs']),
                learning_rate=individual['learning_rate'],
                warmup_steps=int(individual['warmup_steps']),
                gradient_accumulation_steps=int(individual['gradient_accumulation_steps'])
            )
            
            # Train and get validation loss
            validation_loss = trainer.train(dataset, training_args, return_val_loss=True)
            
            # Calculate fitness (higher is better)
            fitness = 10000 / (1 + validation_loss)  # Transform loss to fitness
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating individual: {str(e)}")
            return 0.0  # Penalty for failed evaluation
            
    def optimize(self) -> Dict[str, Any]:
        """Run the optimization process."""
        # Load dataset
        chunks_file = os.path.join(self.input_dir, 'training_chunks.json')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        dataset = GPT2Dataset(chunks_file, tokenizer)
        
        # Initialize population
        self.ga.initialize_population()
        
        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")
            
            # Evaluate current population
            fitness_scores = []
            for individual in self.ga.population:
                fitness = self.evaluate_individual(individual.hyperparameters, dataset)
                fitness_scores.append(fitness)
            
            # Update population fitness
            self.ga.update_population_fitness(fitness_scores)
            
            # Print statistics
            stats = self.ga.get_statistics()
            print("\n================================================================================")
            print(f"Generation {stats['generation']} - Statistics")
            print("================================================================================\n")
            print(f"Population Size: {stats['population_size']}")
            print(f"Diversity: {stats['diversity']}%\n")
            print("Fitness Scores:")
            print(f"- Maximum: {stats['fitness_max']:.2f}")
            print(f"- Average: {stats['fitness_avg']:.2f}")
            print(f"- Minimum: {stats['fitness_min']:.2f}\n")
            print("Current Rates:")
            print(f"- Mutation: {stats['mutation_rate']}%")
            print(f"- Crossover: {stats['crossover_rate']}%")
            print(f"- Random: {stats['random_rate']}%\n")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'generation_{generation}.json'
            )
            self.ga.save_state(checkpoint_path)
            
            # Create next generation
            if generation < self.num_generations - 1:
                self.ga.create_next_generation()
        
        # Return best hyperparameters
        return {
            "best_hyperparameters": self.ga.best_individual.hyperparameters,
            "best_fitness": self.ga.best_individual.fitness,
            "final_stats": self.ga.get_statistics()
        }

def main():
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        input_dir='./tokenized_data',
        output_dir='./optimized_model',
        population_size=30,
        num_generations=10
    )
    
    # Run optimization
    try:
        results = optimizer.optimize()
        
        # Save results
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        print("\nOptimization completed successfully!")
        print("\nBest Hyperparameters:")
        for param, value in results['best_hyperparameters'].items():
            print(f"- {param}: {value}")
        print(f"\nBest Fitness Score: {results['best_fitness']:.2f}")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")

if __name__ == "__main__":
    main() 