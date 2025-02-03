import copy
import random

from optimizer.Chromosome import Chromosome


class NASOptimizer:
    def __init__(self,
                 input_shape: tuple,
                 num_classes: int,
                 population_size: int = 20,
                 num_generations: int = 10,
                 mutation_rate: float = 0.3,
                 elite_size: int = 2):
        """
        Inicjalizacja optymalizatora NAS.

        Args:
            input_shape: Kształt danych wejściowych (channels, height, width)
            num_classes: Liczba klas wyjściowych
            population_size: Rozmiar populacji
            num_generations: Liczba generacji
            mutation_rate: Współczynnik mutacji
            elite_size: Liczba najlepszych osobników zachowanych bez zmian
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

    def initialize_population(self):
        """Inicjalizacja początkowej populacji losowymi chromosomami."""
        self.population = []
        for _ in range(self.population_size):
            chromosome = Chromosome.generate_random(
                self.input_shape,
                self.num_classes,
                min_layers=3,
                max_layers=10
            )
            self.population.append(chromosome)

    def select_parents(self, fitness_scores):
        """
        Selekcja rodziców metodą turnieju.

        Args:
            fitness_scores: Lista wartości fitness dla populacji

        Returns:
            Dwa wybrane chromosomy-rodzice
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

    def create_next_generation(self, fitness_scores):
        """
        Tworzy następną generację populacji.

        Args:
            fitness_scores: Lista wartości fitness dla aktualnej populacji
        """
        # Sortowanie populacji według fitness
        sorted_indices = sorted(range(len(fitness_scores)),
                                key=lambda k: fitness_scores[k],
                                reverse=True)

        # Zachowanie elit
        new_population = [copy.deepcopy(self.population[i])
                          for i in sorted_indices[:self.elite_size]]

        # Uzupełnienie populacji nowymi osobnikami
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)

            # Krzyżowanie
            child = parent1.crossover(parent2)

            # Mutacja
            if random.random() < self.mutation_rate:
                child = child.mutate()

            new_population.append(child)

        self.population = new_population

    def optimize(self, train_loader, test_loader, device='cuda'):
        """
        Główna pętla optymalizacji.

        Args:
            train_loader: DataLoader z danymi treningowymi
            test_loader: DataLoader z danymi testowymi
            device: Urządzenie do wykonywania obliczeń (cuda/cpu)

        Returns:
            Najlepszy znaleziony chromosom
        """
        print("Inicjalizacja populacji początkowej...")
        self.initialize_population()

        for generation in range(self.num_generations):
            print(f"\nGeneracja {generation + 1}/{self.num_generations}")

            # Obliczenie fitness dla każdego chromosomu
            fitness_scores = []
            for i, chromosome in enumerate(self.population):
                print(f"\nOcena chromosomu {i + 1}/{self.population_size}")
                fitness = chromosome.get_fitness(train_loader, test_loader,
                                                 num_epochs=2, device=device)
                fitness_scores.append(fitness)

                # Aktualizacja najlepszego znalezionego rozwiązania
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = copy.deepcopy(chromosome)
                    print(f"Nowy najlepszy fitness: {self.best_fitness}")

            # Statystyki generacji
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            best_gen_fitness = max(fitness_scores)
            print(f"\nStatystyki generacji {generation + 1}:")
            print(f"Średni fitness: {avg_fitness:.4f}")
            print(f"Najlepszy fitness: {best_gen_fitness:.4f}")
            print(f"Najlepszy fitness ogółem: {self.best_fitness:.4f}")

            # Tworzenie następnej generacji
            if generation < self.num_generations - 1:
                self.create_next_generation(fitness_scores)

        return self.best_chromosome