import torch
from torch import nn
import torch.nn.functional as F


def eval_model(model, data_loader, device, mode = "Test"):
    model.eval()
    criterion = nn.NLLLoss()
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
    

def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.NLLLoss()
    validation_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validation_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_validation_loss = validation_loss / len(data_loader)
    epoch_validation_acc = 100. * correct / total
    return epoch_validation_loss, epoch_validation_acc

