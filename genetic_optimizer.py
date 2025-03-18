import numpy as np
from typing import List, Dict, Tuple, Optional
import random
import json
import os

class Individual:
    def __init__(self, hyperparameters: Dict[str, float], fitness: float = 0.0):
        self.hyperparameters = hyperparameters
        self.fitness = fitness

    def __str__(self):
        return f"Fitness: {self.fitness}, Params: {self.hyperparameters}"

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 30,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        random_rate: float = 0.3,
        param_ranges: Dict[str, Tuple[float, float]] = None
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_rate = random_rate
        
        # Default parameter ranges if none provided
        self.param_ranges = param_ranges or {
            'learning_rate': (1e-5, 1e-3),
            'batch_size': (1, 32),
            'num_epochs': (1, 10),
            'warmup_steps': (50, 500),
            'gradient_accumulation_steps': (1, 8)
        }
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            hyperparameters = {}
            for param_name, (min_val, max_val) in self.param_ranges.items():
                if param_name in ['batch_size', 'num_epochs', 'warmup_steps', 'gradient_accumulation_steps']:
                    value = int(random.uniform(min_val, max_val + 1))
                else:
                    value = random.uniform(min_val, max_val)
                hyperparameters[param_name] = value
            self.population.append(Individual(hyperparameters))

    def calculate_diversity(self) -> float:
        """Calculate population diversity as average pairwise Euclidean distance."""
        if not self.population:
            return 0.0
            
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = self._calculate_distance(self.population[i], self.population[j])
                distances.append(dist)
                
        return np.mean(distances) if distances else 0.0

    def _calculate_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate Euclidean distance between two individuals."""
        squared_diff_sum = 0
        for param in self.param_ranges.keys():
            # Normalize the values to [0,1] range
            min_val, max_val = self.param_ranges[param]
            val1 = (ind1.hyperparameters[param] - min_val) / (max_val - min_val)
            val2 = (ind2.hyperparameters[param] - min_val) / (max_val - min_val)
            squared_diff_sum += (val1 - val2) ** 2
        return np.sqrt(squared_diff_sum)

    def _select_parent(self) -> Individual:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1
            
        child_params = {}
        for param_name in self.param_ranges.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1.hyperparameters[param_name]
            else:
                child_params[param_name] = parent2.hyperparameters[param_name]
        return Individual(child_params)

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        mutated_params = individual.hyperparameters.copy()
        
        for param_name, (min_val, max_val) in self.param_ranges.items():
            if random.random() < self.mutation_rate:
                if param_name in ['batch_size', 'num_epochs', 'warmup_steps', 'gradient_accumulation_steps']:
                    mutated_params[param_name] = int(random.uniform(min_val, max_val + 1))
                else:
                    # For continuous parameters, use Gaussian mutation
                    std_dev = (max_val - min_val) * 0.1
                    value = mutated_params[param_name] + random.gauss(0, std_dev)
                    mutated_params[param_name] = max(min_val, min(max_val, value))
                    
        return Individual(mutated_params)

    def create_next_generation(self) -> None:
        """Create the next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: Keep the best individual
        if self.best_individual:
            new_population.append(Individual(
                self.best_individual.hyperparameters.copy(),
                self.best_individual.fitness
            ))
        
        # Generate new individuals
        while len(new_population) < self.population_size:
            if random.random() < self.random_rate:
                # Create random individual
                hyperparameters = {}
                for param_name, (min_val, max_val) in self.param_ranges.items():
                    if param_name in ['batch_size', 'num_epochs', 'warmup_steps', 'gradient_accumulation_steps']:
                        value = int(random.uniform(min_val, max_val + 1))
                    else:
                        value = random.uniform(min_val, max_val)
                    hyperparameters[param_name] = value
                new_population.append(Individual(hyperparameters))
            else:
                # Selection and crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
        
        self.population = new_population
        self.generation += 1

    def get_statistics(self) -> Dict:
        """Get current generation statistics."""
        if not self.population:
            return {
                "generation": self.generation,
                "population_size": self.population_size,
                "diversity": 0,
                "fitness_max": 0,
                "fitness_avg": 0,
                "fitness_min": 0,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "random_rate": self.random_rate
            }
            
        fitness_scores = [ind.fitness for ind in self.population]
        
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "diversity": round(self.calculate_diversity() * 100, 2),
            "fitness_max": round(max(fitness_scores), 2),
            "fitness_avg": round(np.mean(fitness_scores), 2),
            "fitness_min": round(min(fitness_scores), 2),
            "mutation_rate": round(self.mutation_rate * 100, 2),
            "crossover_rate": round(self.crossover_rate * 100, 2),
            "random_rate": round(self.random_rate * 100, 2)
        }

    def save_state(self, filepath: str) -> None:
        """Save the current state of the genetic algorithm."""
        state = {
            "generation": self.generation,
            "population": [
                {
                    "hyperparameters": ind.hyperparameters,
                    "fitness": ind.fitness
                }
                for ind in self.population
            ],
            "best_individual": {
                "hyperparameters": self.best_individual.hyperparameters,
                "fitness": self.best_individual.fitness
            } if self.best_individual else None,
            "param_ranges": self.param_ranges,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "random_rate": self.random_rate
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load a previously saved state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.generation = state["generation"]
        self.population = [
            Individual(
                hyperparameters=ind["hyperparameters"],
                fitness=ind["fitness"]
            )
            for ind in state["population"]
        ]
        
        if state["best_individual"]:
            self.best_individual = Individual(
                hyperparameters=state["best_individual"]["hyperparameters"],
                fitness=state["best_individual"]["fitness"]
            )
            
        self.param_ranges = state["param_ranges"]
        self.mutation_rate = state["mutation_rate"]
        self.crossover_rate = state["crossover_rate"]
        self.random_rate = state["random_rate"]

    def update_population_fitness(self, fitness_scores: List[float]) -> None:
        """Update the fitness scores of the current population."""
        if len(fitness_scores) != len(self.population):
            raise ValueError("Number of fitness scores must match population size")
            
        for individual, fitness in zip(self.population, fitness_scores):
            individual.fitness = fitness
            
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if not self.best_individual or current_best.fitness > self.best_individual.fitness:
            self.best_individual = Individual(
                current_best.hyperparameters.copy(),
                current_best.fitness
            ) 