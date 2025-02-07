import torch
import wandb

from optimizer.Chromosome import Chromosome

import traceback



def eval_model(model, data_loader, device, mode):
    model.eval()
    criterion = torch.nn.NLLLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss.backward()
                        # optimizer.step()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / len(data_loader)
        test_acc = 100. * correct / total

        # run.log({
        #     f"test/loss": test_loss,
        #     f"test/accuracy": test_acc,
        # })
        print(f"{mode}, {mode} Loss: {test_loss}, {mode} Accuracy: {test_acc}%")
        return test_loss, test_acc
    
def train_model(model, train_loader, epochs, device: str = "cuda"):
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
            # with wandb.init(
            #         project=self.project_name,
            #         name=f"NAS_optimization_MNIST({generation} - {chromosome_id})",
            #         mode="disabled",
            #         entity="matik001",
            #         config={
            #             "input_shape": self.input_shape,
            #             "num_classes": self.num_classes,
            #             "population_size": self.population_size,
            #             "num_generations": self.num_generations,
            #             "mutation_rate": self.mutation_rate,
            #             "elite_size": self.elite_size,
            #             "generation": generation,
            #             "chromosome_id": chromosome_id
            #         },
            # ) as run:
            #     a = 5

            #     run.log({
            #         # "architecture": str(chromosome)
                    
            #     })

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = torch.nn.NLLLoss()

                total_params = sum(p.numel() for p in model.parameters())
                # run.log({
                #     f"total_parameters": total_params,
                #     f"num_layers": len(chromosome.layers)
                # })

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

                        # if batch_idx % 100 == 0:
                            # run.log({
                            #     f"train/batch_loss": loss.item(),
                            #     f"train/batch_accuracy": 100. * correct / total,
                            #     "epoch": epoch,
                            #     "batch": batch_idx
                            # })
                            # print(f"Training, Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, Accuracy: {100. * correct / total}")

                    epoch_train_loss = train_loss / len(train_loader)
                    epoch_train_acc = 100. * correct / total
                    # run.log({
                    #     f"train/epoch_loss": epoch_train_loss,
                    #     f"validation/epoch_loss": epoch_validation_loss,
                    #     f"validation/accuracy": epoch_validation_acc,
                    #     "epoch": epoch
                    # })

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