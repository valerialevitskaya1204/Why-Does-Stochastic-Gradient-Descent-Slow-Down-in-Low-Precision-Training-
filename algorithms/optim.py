from configs import DEVICE, EPOCHS
import numpy as np
import torch
from training.metrics import logistic_loss_and_grad, accuracy_pm1
from quantization.quant import quantize_grad_proper

def train_saga(w, X, y, X_test, y_test, lr, precision):
    n, d = X.shape
    grad_memory = torch.zeros((n, d), device=DEVICE)  #∇f_i(φ_i) for each sample
    grad_avg = torch.zeros(d, device=DEVICE)  

    for i in range(n):
        xi = X[i]
        yi = y[i]
        _, grad = logistic_loss_and_grad(w, xi, yi)
        grad_q, _, _ = quantize_grad_proper(grad, precision)
        grad_memory[i] = grad_q
        grad_avg += grad_q / n  

    train_losses = []
    test_accs = []
    shrink_qs = []  #q

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        q_vals = []

        for i in perm:
            xi = X[i]
            yi = y[i]

            # ∇f_{i_k}(x^k)
            loss, grad = logistic_loss_and_grad(w, xi, yi)
            epoch_loss += loss.item()

            #quantization
            grad_q, q, _ = quantize_grad_proper(grad, precision)
            q_vals.append(float(q.item()))

            old_grad_q = grad_memory[i].clone()

            #SAGA update
            update = grad_q - old_grad_q + grad_avg

            w -= lr * update

            #update gradient table
            grad_memory[i] = grad_q

            grad_avg += (grad_q - old_grad_q) / n

        q_mean = float(np.mean(q_vals))
        shrink_qs.append(q_mean)

        train_losses.append(epoch_loss / n)
        test_accs.append(accuracy_pm1(w, X_test, y_test))

        print(
            f"[{precision.upper()} | lr={lr:.0e}] "
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"loss={train_losses[-1]:.6f} | acc={test_accs[-1]:.4f} | mean q={q_mean:.4f}"
        )

    return np.array(train_losses), np.array(test_accs), np.array(shrink_qs)