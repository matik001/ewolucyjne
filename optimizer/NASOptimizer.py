import random
from typing import List, Tuple
import copy
import numpy as np
import wandb

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from optimizer.Layers import Layer, LinearLayer, Conv2dLayer, MaxPool2dLayer, DropoutLayer, ReluLayer
from optimizer.Chromosome import Chromosome


class NASOptimizer:
    def __init__(self,
                 input_shape: tuple,
                 num_classes: int,
                 population_size: int = 20,
                 num_generations: int = 10,
                 mutation_rate: float = 0.3,
                 elite_size: int = 2,
                 project_name: str = "NAS-Optimization",
                 epoch: int = 2):
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
        self.epoch = epoch

    def initialize_population(self):
        """Inicjalizacja początkowej populacji losowymi chromosomami."""
        self.population = []
        for _ in range(self.population_size):
            try:
                chromosome = Chromosome.generate_random(
                    self.input_shape,
                    self.num_classes,
                    min_layers=3,
                    max_layers=10
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

    def train_and_evaluate_chromosome(self,
                                      chromosome: Chromosome,
                                      train_loader: DataLoader,
                                      val_loader: DataLoader,
                                      test_loader: DataLoader,
                                      generation: int,
                                      chromosome_id: int,
                                      device: str = 'cuda') -> float:
        """
        Trenuje i ocenia pojedynczy chromosom.

        Args:
            chromosome: Chromosom do wytrenowania
            train_loader: DataLoader z danymi treningowymi
            test_loader: DataLoader z danymi testowymi
            generation: Numer aktualnej generacji
            chromosome_id: ID chromosomu w populacji
            run: Obiekt wandb.Run do logowania
            device: Urządzenie do wykonywania obliczeń

        Returns:
            float: Wartość fitness (dokładność na zbiorze testowym)
        """
        try:
            with wandb.init(
                    project=self.project_name,
                    name=f"NAS_optimization_MNIST({generation} - {chromosome_id})",
                    mode="disabled",
                    entity="matik001",
                    config={
                        "input_shape": self.input_shape,
                        "num_classes": self.num_classes,
                        "population_size": self.population_size,
                        "num_generations": self.num_generations,
                        "mutation_rate": self.mutation_rate,
                        "elite_size": self.elite_size,
                        "generation": generation,
                        "chromosome_id": chromosome_id
                    },
            ) as run:
                model = chromosome.to_nn_module().to(device)

                run.log({
                    "architecture": str(chromosome)
                })

                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.NLLLoss()

                total_params = sum(p.numel() for p in model.parameters())
                run.log({
                    f"total_parameters": total_params,
                    f"num_layers": len(chromosome.layers)
                })

                for epoch in range(self.epoch):
                    # Trening
                    model.train()
                    train_loss = 0
                    correct = 0
                    total = 0

                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        if batch_idx % 100 == 0:
                            run.log({
                                f"train/batch_loss": loss.item(),
                                f"train/batch_accuracy": 100. * correct / total,
                                "epoch": epoch,
                                "batch": batch_idx
                            })
                            # print(f"Training, Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Accuracy: {100. * correct / total}")

                    # Ewaluacja
                    model.eval()
                    validation_loss = 0
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)

                            validation_loss += loss.item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()

                    epoch_train_loss = train_loss / len(train_loader)
                    epoch_validation_loss = validation_loss / len(test_loader)
                    epoch_validation_acc = 100. * correct / total

                    run.log({
                        f"train/epoch_loss": epoch_train_loss,
                        f"validation/epoch_loss": epoch_validation_loss,
                        f"validation/accuracy": epoch_validation_acc,
                        "epoch": epoch
                    })
                    print(f"Validation, Epoch: {epoch}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_validation_loss}, Validation Accuracy: {epoch_validation_acc}")

                return epoch_validation_acc

        except Exception as e:
            print(f"Error in training chromosome {chromosome_id}: {str(e)}")
            run.log({f"chromosome_{chromosome_id}/error": str(e)})
            return 0.0

    def optimize(self, train_loader: DataLoader, val_loader : DataLoader, test_loader: DataLoader, device: str = 'cuda') -> Chromosome:
        """
        Główna pętla optymalizacji.

        Args:
            train_loader: DataLoader z danymi treningowymi
            test_loader: DataLoader z danymi testowymi
            device: Urządzenie do wykonywania obliczeń

        Returns:
            Chromosome: Najlepszy znaleziony chromosom
        """

        print("Initializing population...")
        self.initialize_population()

        for generation in range(self.num_generations):
            print(f"\nGeneration {generation + 1}/{self.num_generations}")

            fitness_scores = []
            for i, chromosome in enumerate(self.population):
                print(f"\nEvaluating chromosome {i + 1}/{self.population_size}")

                fitness = self.train_and_evaluate_chromosome(
                    chromosome, train_loader, val_loader, test_loader,
                    generation, i, device
                )
                fitness_scores.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = copy.deepcopy(chromosome)
                    # run.log({
                    #     "best_fitness": self.best_fitness,
                    #     "best_generation": generation,
                    #     "best_chromosome": i,
                    #     "best_architecture": wandb.Html(str(chromosome))
                    # })

            # Logowanie statystyk generacji
            # run.log({
            #     "generation": generation,
            #     "mean_fitness": np.mean(fitness_scores),
            #     "max_fitness": max(fitness_scores),
            #     "min_fitness": min(fitness_scores),
            #     "fitness_std": np.std(fitness_scores),
            # })

            if generation < self.num_generations - 1:
                self.create_next_generation(fitness_scores)

        # Logowanie końcowych wyników
        # run.log({
        #     "final_best_fitness": self.best_fitness,
        #     "final_architecture": wandb.Html(str(self.best_chromosome))
        # })

        return self.best_chromosome


def save_best_model(chromosome: Chromosome, path: str):
    """
    Zapisuje najlepszy model do pliku.

    Args:
        chromosome: Chromosom do zapisania
        path: Ścieżka do pliku
    """
    model = chromosome.to_nn_module()
    torch.save({
        'state_dict': model.state_dict(),
        'architecture': str(chromosome)
    }, path)
