import numpy as np
import torch
from algorithms.configs import SEEDS, DEVICE, BASE_LR, FP4_LR, PRECISIONS, RESULTS_DIR, configs
from training.data.load_everything import download_mushrooms_dataset, train_loader, test_loader
from training.models import ResNet50
from quantization.quant import apply_gradient_quantization
from algorithms.optim import train_saga

import torch.nn as nn


def run_experiments():
    for seed in SEEDS:
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_tr, X_te, y_tr, y_te = download_mushrooms_dataset()

        X_tr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
        y_tr = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
        X_te = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        y_te = torch.tensor(y_te, dtype=torch.float32, device=DEVICE)

        for precision in PRECISIONS:
            lrs = [BASE_LR]
            if precision == "fp4":
                lrs.append(FP4_LR)

            for lr in lrs:
                w = torch.zeros(X_tr.shape[1], device=DEVICE)

                losses, accs, qs = train_saga(
                    w, X_tr, y_tr, X_te, y_te,
                    lr=lr,
                    precision=precision
                )

                fname = f"{RESULTS_DIR}/sag_{precision}_lr{lr:.0e}_seed{seed}.npz"
                np.savez(
                    fname,
                    train_loss=losses,
                    test_acc=accs,
                    shrinkage_q=qs,
                )
                print(f"Saved {fname}")


def train_model(model, trainloader, testloader, precision, lr, epochs=100):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    train_losses = []
    test_accuracies = []
    
    model.to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            apply_gradient_quantization(model, precision)
            
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        
        if epoch % 1 == 0: 
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{epochs}], Precision: {precision}, '
                  f'Train Loss: {avg_train_loss:.4f}, Test Acc: {accuracy:.2f}%, LR: {lr}')
    
    return train_losses, test_accuracies

def train_baseline():

    results = {}
    for config in configs:
        print(f"\nTraining with {config['name']}")

        model = ResNet50(num_classes=10)
        
        train_losses, test_accuracies = train_model(
            model=model,
            trainloader=train_loader,
            testloader=test_loader,
            precision=config['precision'],
            lr=config['lr'],
            epochs=100 
        )
        
        results[config['name']] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
    return results