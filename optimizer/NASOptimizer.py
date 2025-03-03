import random
from typing import List, Tuple
import copy
import wandb

import torch
from torch.utils.data import DataLoader

from optimizer.Chromosome import Chromosome
from training_utils import eval_model, train_model, save_model

class NASOptimizer:
    def __init__(self,
                 input_shape: tuple,
                 num_classes: int,
                 population_size: int = 20,
                 num_generations: int = 10,
                 mutation_rate: float = 0.3,
                 elite_size: int = 2,
                 project_name: str = "NAS",
                 group_name: str = "MNIST",
                 epoch: int = 2,
                 min_layers: int = 2,
                 max_layers: int = 3
                 ):
        """
        Inicjalizacja optymalizatora NAS.

        Args:
            input_shape: Kształt danych wejściowych (channels, height, width)
            num_classes: Liczba klas wyjściowych
            population_size: Rozmiar populacji
            num_generations: Liczba generacji
            mutation_rate: Współczynnik mutacji
            elite_size: Liczba najlepszych osobników zachowanych bez zmian
            project_name: Nazwa projektu w wandb
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = []
        self.best_fitness = float('-inf')
        self.best_chromosome = None
        self.project_name = project_name
        self.group_name = group_name
        self.epoch = epoch
        self.min_layers = min_layers
        self.max_layers = max_layers

    def initialize_population(self):
        """Inicjalizacja początkowej populacji losowymi chromosomami."""
        self.population = []
        for _ in range(self.population_size):
            try:
                chromosome = Chromosome.generate_random(
                    self.input_shape,
                    self.num_classes,
                    min_layers=self.min_layers,
                    max_layers=self.max_layers
                )
                self.population.append(chromosome)
            except Exception as e:
                print(f"Error during chromosome generation: {str(e)}")

        if len(self.population) == 0:
            raise RuntimeError("Failed to initialize population")

    def select_parents(self, fitness_scores: List[float]) -> Tuple[Chromosome, Chromosome]:
        """
        Selekcja rodziców metodą turnieju.

        Args:
            fitness_scores: Lista wartości fitness dla populacji

        Returns:
            Tuple zawierająca dwa wybrane chromosomy-rodzice
        """

        def tournament_selection():
            tournament_size = 3
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            return self.population[winner_idx]

        parent1 = tournament_selection()
        parent2 = tournament_selection()
        while parent2 is parent1:
            parent2 = tournament_selection()

        return parent1, parent2

    def create_next_generation(self, fitness_scores: List[float]):
        """
        Tworzy następną generację populacji.

        Args:
            fitness_scores: Lista wartości fitness dla aktualnej populacji
        """
        sorted_indices = sorted(range(len(fitness_scores)),
                                key=lambda k: fitness_scores[k],
                                reverse=True)

        new_population = [copy.deepcopy(self.population[i])
                          for i in sorted_indices[:self.elite_size]]

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)

            try:
                child = parent1.crossover(parent2)
                if random.random() < self.mutation_rate:
                    child = child.mutate()

                if child.is_correct()[0]:
                    new_population.append(child)
            except Exception as e:
                print(f"Error creating new chromosome: {str(e)}")

        self.population = new_population
            
    def optimize(self, train_loader: DataLoader, val_loader : DataLoader, save_initial = False, save_medium = False, device: str = 'cuda') -> Chromosome:
        """
        Główna pętla optymalizacji.

        Args:
            train_loader: DataLoader z danymi treningowymi
            val_loader: DataLoader z danymi walidycyjnymi
            device: Urządzenie do wykonywania obliczeń

        Returns:
            Chromosome: Najlepszy znaleziony chromosom
        """
        print("Initializing population...")
        self.initialize_population()

        if save_initial:
            save_model(self.population[0], f"saved_models/{self.group_name}_initial.pth")

        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")
            if save_medium and generation == self.num_generations // 2 + 1:
                save_model(self.population[0], f"saved_models/{self.group_name}_medium.pth")
            fitness_scores = []
            for i, chromosome in enumerate(self.population):
                print(f"\nEvaluating chromosome {i + 1}/{self.population_size}")
                model = chromosome.to_nn_module().to(device)
                run =  wandb.init(project=self.project_name, group=self.group_name, name=f"{self.project_name}_{self.group_name}({generation} - {i})", tags=[self.group_name])
                run.config.update({
                    "input_shape": self.input_shape,
                    "num_classes": self.num_classes,
                    "population_size": self.population_size,
                    "num_generations": self.num_generations,
                    "mutation_rate": self.mutation_rate,
                    "elite_size": self.elite_size,
                    "generation": generation,
                    "chromosome_id": i,
                })
                run.log({
                    "architecture": str(chromosome),
                    f"total_parameters": sum(p.numel() for p in model.parameters()),
                    f"num_layers": len(chromosome.layers)
                })
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                training_loss, training_acc = train_model(model, train_loader,self.epoch, run, optimizer,device)
                print(f"Training loss: {training_loss}, Training accuracy: {training_acc}")
                _, fitness = eval_model(model, val_loader, run, device, "Validation")
                fitness_scores.append(fitness)
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = copy.deepcopy(chromosome)
                run.finish()

            if generation < self.num_generations - 1:
                self.create_next_generation(fitness_scores)

        return self.best_chromosome



