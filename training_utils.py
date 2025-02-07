import torch
import wandb

from optimizer.Chromosome import Chromosome

import traceback



def eval_model(model, data_loader, run, device, mode):
    model.eval()
    criterion = torch.nn.NLLLoss()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        loss /= len(data_loader)
        acc = 100. * correct / total
        run.log({
            f"{mode}/loss": loss,
            f"{mode}/accuracy": acc,
        })
        print(f"{mode}, {mode} Loss: {loss}, {mode} Accuracy: {acc}%")
        return loss, acc
    
def train_model(model, train_loader, epochs, run, device: str = "cuda"):
        """
        Trenuje model.

        Args:
            modele: model do wytrenowania
            train_loader: DataLoader z danymi treningowymi
            generation: Numer aktualnej generacji
            chromosome_id: ID chromosomu w populacji
            run: Obiekt wandb.Run do logowania
            device: Urządzenie do wykonywania obliczeń

        Returns:
            float: Wartość fitness (dokładność na zbiorze testowym)
        """
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.NLLLoss()
            for epoch in range(epochs):
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

                epoch_train_loss = train_loss / len(train_loader)
                epoch_train_acc = 100. * correct / total
                run.log({
                    f"train/epoch_loss": epoch_train_loss,
                    f"train/epoch_acc": epoch_train_acc,
                    "epoch": epoch
                })

            return epoch_train_loss, epoch_train_acc

        except Exception as e:
            print(f"Error in training: {str(e)}")
            traceback.print_exc()
            # run.log({f"chromosome_{chromosome_id}/error": str(e)})
            return 0.0

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