import random
from typing import List, Tuple
import copy

from torch import nn

from optimizer.Layers import Layer, LinearLayer, Conv2dLayer, MaxPool2dLayer, DropoutLayer, ReluLayer


class Chromosome:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.layers: List[Layer] = []
        self.current_shape = input_shape
        self.max_tries = 10
        self.num_classes = num_classes


    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        layer.input_shape = copy.deepcopy(self.current_shape)
        self.current_shape = layer.calc_output_shape()

    def get_output_shape(self):
        return self.current_shape if self.layers else self.input_shape

    def mutate(self, mutation_rate=0.3):
        for i in range(self.max_tries):
            new_chromosome = copy.deepcopy(self)

            if random.random() < mutation_rate:
                mutation_type = random.choice(['add', 'remove', 'modify'])

                if mutation_type == 'add' and len(new_chromosome.layers) > 0:
                    insert_position = random.randint(0, len(new_chromosome.layers))
                    current_shape = (new_chromosome.input_shape if insert_position == 0
                                     else new_chromosome.layers[insert_position - 1].calc_output_shape())

                    layer_type = random.choice([LinearLayer, Conv2dLayer, MaxPool2dLayer, DropoutLayer, ReluLayer])

                    if layer_type == LinearLayer:
                        out_features = random.choice([32, 64, 128, 256])
                        new_layer = LinearLayer(current_shape, out_features)
                    elif layer_type == Conv2dLayer:
                        out_channels = random.choice([16, 32, 64])
                        kernel_size = random.choice([3, 5])
                        new_layer = Conv2dLayer(current_shape, out_channels, kernel_size)
                    elif layer_type == MaxPool2dLayer:
                        kernel_size = random.choice([2, 3])
                        new_layer = MaxPool2dLayer(current_shape, kernel_size)
                    elif layer_type == DropoutLayer:
                        p = random.choice([0.3, 0.5])
                        new_layer = DropoutLayer(current_shape, p)
                    else:  # ReluLayer
                        new_layer = ReluLayer(current_shape)

                    new_chromosome.layers.insert(insert_position, new_layer)

                elif mutation_type == 'remove' and len(new_chromosome.layers) > 1:
                    remove_idx = random.randint(0, len(new_chromosome.layers) - 1)
                    new_chromosome.layers.pop(remove_idx)

                elif mutation_type == 'modify' and len(new_chromosome.layers) > 0:
                    modify_idx = random.randint(0, len(new_chromosome.layers) - 1)
                    current_layer = new_chromosome.layers[modify_idx]

                    if isinstance(current_layer, LinearLayer):
                        out_features = random.choice([32, 64, 128, 256])
                        new_chromosome.layers[modify_idx] = LinearLayer(current_layer.input_shape, out_features)
                    elif isinstance(current_layer, Conv2dLayer):
                        out_channels = random.choice([16, 32, 64])
                        kernel_size = random.choice([3, 5])
                        new_chromosome.layers[modify_idx] = Conv2dLayer(current_layer.input_shape,
                                                                        out_channels, kernel_size)
                    elif isinstance(current_layer, MaxPool2dLayer):
                        kernel_size = random.choice([2, 3])
                        new_chromosome.layers[modify_idx] = MaxPool2dLayer(current_layer.input_shape, kernel_size)
                    elif isinstance(current_layer, DropoutLayer):
                        p = random.choice([0.3, 0.5])
                        new_chromosome.layers[modify_idx] = DropoutLayer(current_layer.input_shape, p)

                new_chromosome._update_shapes()
            if new_chromosome.is_correct()[0]:
                return new_chromosome
            else:
                print("Bad result of mutation, retrying",  new_chromosome.is_correct()[1])
        return copy.deepcopy(self)

    def crossover(self, other: 'Chromosome'):
        """
        Wykonuje crossover z innym chromosomem.
        Warstwy tego samego typu są krzyżowane niezależnie od ich pozycji.
        """
        if not isinstance(other, Chromosome):
            raise ValueError("Crossover must be performed with another Chromosome")

        for i in range(self.max_tries):
            new_chromosome = Chromosome(self.input_shape, self.num_classes)

            idx1 = 0  # indeks dla pierwszego chromosomu
            idx2 = 0  # indeks dla drugiego chromosomu

            while idx1 < len(self.layers) or idx2 < len(other.layers):
                # Sprawdź czy któryś z indeksów przekroczył długość
                if idx1 >= len(self.layers):
                    # Dodaj pozostałe warstwy z drugiego chromosomu
                    new_layer = copy.deepcopy(other.layers[idx2])
                    new_chromosome.add_layer(new_layer)
                    idx2 += 1
                    continue

                if idx2 >= len(other.layers):
                    # Dodaj pozostałe warstwy z pierwszego chromosomu
                    new_layer = copy.deepcopy(self.layers[idx1])
                    new_chromosome.add_layer(new_layer)
                    idx1 += 1
                    continue

                layer1 = self.layers[idx1]
                layer2 = other.layers[idx2]

                # Sprawdź czy warstwy są tego samego typu
                if type(layer1) == type(layer2):
                    # Wykonaj crossover i dodaj nową warstwę
                    new_layer = layer1.crossover(layer2)
                    new_chromosome.add_layer(new_layer)
                    idx1 += 1
                    idx2 += 1
                else:
                    if random.randint(1, 2) == 1:
                        idx1 += 1
                    else:
                        idx2 += 1
            if new_chromosome.is_correct()[0]:
                return new_chromosome
            else:
                print("Bad result of crossover, retrying", new_chromosome.is_correct()[1])
        return copy.deepcopy(self)
    def _update_shapes(self):
        current_shape = self.input_shape
        for i, layer in enumerate(self.layers):
            layer.input_shape = current_shape
            layer.output_shape = layer.calc_output_shape()
            current_shape = layer.output_shape
        self.current_shape = current_shape

    def is_correct(self) -> Tuple[bool, str]:
        """
        Sprawdza czy architektura sieci jest poprawna.

        Returns:
            Tuple[bool, str]: (czy_poprawna, komunikat_błędu)
        """
        if not self.layers:
            return False, "Brak warstw w sieci"

        # Sprawdź każdą warstwę po kolei
        flatten_occurred = False  # Flaga wskazująca czy wystąpiła operacja spłaszczenia (np. przez Linear)

        for i, current_layer in enumerate(self.layers):
            # 1. Sprawdź czy wymiary się zgadzają
            if i > 0:
                prev_layer = self.layers[i - 1]
                if current_layer.input_shape != prev_layer.calc_output_shape():
                    return False, (f"Niezgodność wymiarów między warstwami {i - 1} i {i}: "
                                   f"{prev_layer.calc_output_shape()} vs {current_layer.input_shape}")

            # 2. Sprawdź ograniczenia dla warstw konwolucyjnych
            if isinstance(current_layer, Conv2dLayer):
                if flatten_occurred:
                    return False, f"Warstwa Conv2d (indeks {i}) nie może wystąpić po spłaszczeniu danych"

                # Sprawdź czy dane wejściowe są 3D (channels, height, width)
                if len(current_layer.input_shape) != 3:
                    return False, f"Warstwa Conv2d (indeks {i}) wymaga 3D input shape, otrzymano: {current_layer.input_shape}"

            # 3. Sprawdź ograniczenia dla warstw MaxPool
            elif isinstance(current_layer, MaxPool2dLayer):
                if flatten_occurred:
                    return False, f"Warstwa MaxPool2d (indeks {i}) nie może wystąpić po spłaszczeniu danych"

                if len(current_layer.input_shape) != 3:
                    return False, f"Warstwa MaxPool2d (indeks {i}) wymaga 3D input shape, otrzymano: {current_layer.input_shape}"

            # 4. Sprawdź ograniczenia dla warstw Linear
            elif isinstance(current_layer, LinearLayer):
                flatten_occurred = True

                # Jeśli to pierwsza warstwa Linear, sprawdź czy wymiary się zgadzają
                # if i > 0 and not isinstance(self.layers[i - 1], LinearLayer):
                #     prev_output = self.layers[i - 1].calc_output_shape()
                #     if len(prev_output) > 1:
                #         flattened_size = prev_output[0]
                #         for dim in prev_output[1:]:
                #             flattened_size *= dim
                #         if flattened_size != current_layer.input_shape[0]:
                #             return False, (f"Niepoprawny rozmiar wejścia warstwy Linear (indeks {i}): "
                #                            f"oczekiwano {flattened_size}, otrzymano {current_layer.input_shape[0]}")

            # 5. Sprawdź ograniczenia dla warstw aktywacji i dropout
            elif isinstance(current_layer, (ReluLayer, DropoutLayer)):
                # Warstwy aktywacji i dropout powinny zachować wymiary
                if current_layer.input_shape != current_layer.calc_output_shape():
                    return False, (f"Warstwa {current_layer.__class__.__name__} (indeks {i}) "
                                   f"zmienia wymiary: {current_layer.input_shape} -> {current_layer.calc_output_shape()}")

        # 6. Sprawdź czy ostatnia warstwa ma odpowiedni wymiar wyjściowy
        if not isinstance(self.layers[-1], LinearLayer):
            return False, "Ostatnia warstwa powinna być typu Linear"

        if self.layers[-1].calc_output_shape()[0] != self.num_classes:
            return False, (f"Ostatnia warstwa powinna mieć {self.num_classes} wyjść, "
                           f"ma {self.layers[-1].calc_output_shape()[0]}")

        # 7. Sprawdź czy sieć nie jest zbyt głęboka lub zbyt płytka
        if len(self.layers) < 2:
            return False, "Sieć jest zbyt płytka (minimum 2 warstwy)"

        if len(self.layers) > 50:  # można dostosować limit
            return False, "Sieć jest zbyt głęboka (maksimum 50 warstw)"

        return True, "Architektura jest poprawna"

    @staticmethod
    def generate_random(input_shape: tuple, num_classes: int,
                        min_layers: int = 3, max_layers: int = 10) -> 'Chromosome':
        """
        Generuje losowy, poprawny chromosom.

        Args:
            input_shape: Krotka określająca wymiary wejścia (channels, height, width)
            num_classes: Liczba klas wyjściowych
            min_layers: Minimalna liczba warstw
            max_layers: Maksymalna liczba warstw

        Returns:
            Chromosome: Losowo wygenerowany, poprawny chromosom
        """
        max_attempts = 50  # Maksymalna liczba prób wygenerowania poprawnego chromosomu

        for attempt in range(max_attempts):
            chrom = Chromosome(input_shape, num_classes)
            current_shape = input_shape

            # Losowa liczba warstw
            num_layers = random.randint(min_layers, max_layers)

            # Flaga wskazująca czy nastąpiło spłaszczenie
            flattened = False

            # Lista możliwych warstw przed spłaszczeniem
            conv_layers = [
                lambda shape: Conv2dLayer(
                    shape,
                    out_channels=random.choice([16, 32, 64, 128]),
                    kernel_size=random.choice([3, 5]),
                    padding=random.choice([0, 1, 2]),
                    stride=random.choice([1, 2])
                ),
                lambda shape: MaxPool2dLayer(
                    shape,
                    kernel_size=2,
                    stride=2
                ),
                lambda shape: ReluLayer(shape),
                lambda shape: DropoutLayer(shape, p=random.uniform(0.1, 0.5))
            ]

            # Lista możliwych warstw po spłaszczeniu
            linear_layers = [
                lambda shape: LinearLayer(
                    shape,
                    out_features=random.choice([64, 128, 256, 512, 1024])
                ),
                lambda shape: ReluLayer(shape),
                lambda shape: DropoutLayer(shape, p=random.uniform(0.1, 0.5))
            ]

            try:
                # Dodawaj warstwy, aż osiągniemy wymaganą liczbę
                for i in range(num_layers - 1):  # -1 bo ostatnia warstwa musi być Linear do num_classes
                    if not flattened and len(current_shape) > 1:
                        # Przed spłaszczeniem - warstwy konwolucyjne i pooling
                        if random.random() < 0.3:  # 30% szans na spłaszczenie
                            flattened = True
                            layer_creator = linear_layers[0]
                        else:
                            layer_creator = random.choice(conv_layers)
                    else:
                        # Po spłaszczeniu - tylko warstwy liniowe
                        flattened = True
                        layer_creator = random.choice(linear_layers)

                    # Stwórz i dodaj warstwę
                    new_layer = layer_creator(current_shape)
                    chrom.add_layer(new_layer)
                    current_shape = new_layer.calc_output_shape()


                final_layer = LinearLayer(current_shape, num_classes)
                chrom.add_layer(final_layer)
                is_valid, message = chrom.is_correct()
                if is_valid:
                    return chrom
                else:
                    print("Not valid", message)
                    print()

            except Exception as e:
                print(f"Próba {attempt + 1} nie powiodła się: {str(e)}")
                continue

        raise RuntimeError("Nie udało się wygenerować poprawnego chromosomu")

    def __str__(self):
        result = f"Input shape: {self.input_shape}\n"
        for i, layer in enumerate(self.layers):
            result += f"Layer {i + 1}: {layer.__class__.__name__}\n"
            result += f"  Input shape: {layer.input_shape}\n"
            result += f"  Output shape: {layer.calc_output_shape()}\n"
        return result



################################################
####          EWALUCJA PYTORCHEM            ####
################################################

    def to_nn_module(self):
        layers = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer) and len(layer.input_shape) > 1:
                layers.append(nn.Flatten())
            if (isinstance(layer, Conv2dLayer) or isinstance(layer, MaxPool2dLayer))  and len(layer.input_shape) == 1:
                print("conv2d after linear!!! :(")

            layers.append(layer.to_nn_layer())

        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)

    # def train(
    #         self,
    #         train_loader: DataLoader,
    #         learning_rate: float = 0.001,
    #         num_epochs: int = 10,
    #         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ) -> nn.Module:
    #     """
    #     Trenuje model i zwraca wytrenowany moduł PyTorch.

    #     Args:
    #         train_loader: DataLoader z danymi treningowymi
    #         learning_rate: współczynnik uczenia
    #         num_epochs: liczba epok treningu
    #         device: urządzenie na którym trenować (cuda/cpu)

    #     Returns:
    #         Wytrenowany model PyTorch
    #     """
    #     model = self.to_nn_module()
    #     model = model.to(device)
    #     model.train()
    #     print(model)
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #     criterion = nn.NLLLoss()

    #     for epoch in range(num_epochs):
    #         running_loss = 0.0
    #         correct = 0
    #         total = 0

    #         for batch_idx, (inputs, targets) in enumerate(train_loader):
    #             inputs, targets = inputs.to(device), targets.to(device)

    #             # Zerowanie gradientów
    #             optimizer.zero_grad()

    #             try:
    #                 # Forward pass
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, targets)

    #                 # Backward pass i optymalizacja
    #                 loss.backward()
    #                 optimizer.step()

    #                 # Statystyki
    #                 running_loss += loss.item()
    #                 _, predicted = outputs.max(1)
    #                 total += targets.size(0)
    #                 correct += predicted.eq(targets).sum().item()

    #                 # Wyświetl postęp co 100 batchy
    #                 if (batch_idx + 1) % 100 == 0:
    #                     print(f'Epoch: {epoch + 1}/{num_epochs} | '
    #                           f'Batch: {batch_idx + 1}/{len(train_loader)} | '
    #                           f'Loss: {running_loss / (batch_idx + 1):.3f} | '
    #                           f'Acc: {100. * correct / total:.2f}%')

    #             except Exception as e:
    #                 print(f"Błąd podczas treningu w epoce {epoch + 1}, batch {batch_idx + 1}: {str(e)}")
    #                 return None

    #         # Statystyki na koniec epoki
    #         epoch_loss = running_loss / len(train_loader)
    #         epoch_acc = 100. * correct / total
    #         print(f'Epoch {epoch + 1} finished | Loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}%')

    #     return model

    # def eval(
    #         self,
    #         model: nn.Module,
    #         test_loader: DataLoader,
    #         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ) -> Tuple[float, float]:
    #     """
    #     Ocenia wytrenowany model na zbiorze testowym.

    #     Args:
    #         model: wytrenowany model PyTorch
    #         test_loader: DataLoader z danymi testowymi
    #         device: urządzenie na którym wykonać ewaluację

    #     Returns:
    #         Tuple (accuracy, loss)
    #     """
    #     if model is None:
    #         return 0.0, float('inf')

    #     model = model.to(device)
    #     model.eval()

    #     total_loss = 0
    #     correct = 0
    #     total = 0

    #     criterion = nn.NLLLoss()

    #     with torch.no_grad():
    #         for inputs, targets in test_loader:
    #             inputs, targets = inputs.to(device), targets.to(device)

    #             try:
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, targets)

    #                 total_loss += loss.item()
    #                 _, predicted = outputs.max(1)
    #                 total += targets.size(0)
    #                 correct += predicted.eq(targets).sum().item()

    #             except Exception as e:
    #                 print(f"Błąd podczas ewaluacji: {str(e)}")
    #                 return 0.0, float('inf')

    #     accuracy = 100. * correct / total
    #     avg_loss = total_loss / len(test_loader)

    #     return accuracy, avg_loss

    # def get_fitness(
    #         self,
    #         train_loader: DataLoader,
    #         test_loader: DataLoader,
    #         learning_rate: float = 0.001,
    #         num_epochs: int = 10,
    #         device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ) -> float:
    #     """
    #     Trenuje model i zwraca jego fitness na podstawie wyników na zbiorze testowym.

    #     Args:
    #         train_loader: DataLoader z danymi treningowymi
    #         test_loader: DataLoader z danymi testowymi
    #         learning_rate: współczynnik uczenia
    #         num_epochs: liczba epok treningu
    #         device: urządzenie na którym wykonać obliczenia

    #     Returns:
    #         Wartość fitness (wyższa = lepsza)
    #     """
    #     trained_model = self.train(train_loader, learning_rate, num_epochs, device)
    #     accuracy, loss = self.eval(trained_model, test_loader, device)

    #     scaled_loss = loss * 100

    #     accuracy_weight = 0.7
    #     loss_weight = 0.3

    #     fitness = (accuracy_weight * accuracy) - (loss_weight * scaled_loss)

    #     print(f"Final Results - Accuracy: {accuracy:.2f}% | Loss: {loss:.4f} | Fitness: {fitness:.4f}")

    #     return fitness