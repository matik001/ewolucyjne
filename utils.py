import torch
from torch import nn

def eval(model, data_loader, device, mode = "Test"):
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