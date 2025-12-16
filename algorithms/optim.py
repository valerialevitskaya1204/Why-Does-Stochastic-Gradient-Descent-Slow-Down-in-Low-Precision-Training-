from configs import DEVICE, EPOCHS
import numpy as np
import torch
from training.metrics import logistic_loss_and_grad, accuracy_pm1
from quantization.quant import quantize_grad_proper


def train_saga(w, X, y, X_test, y_test, lr, precision):
    n, d = X.shape
    grad_memory = torch.zeros((n, d), device=DEVICE)  # ∇f_i(φ_i) for each sample
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
    shrink_qs = []  # q

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        q_vals = []

        for i in perm:
            xi = X[i]
            yi = y[i]

            loss, grad = logistic_loss_and_grad(w, xi, yi)
            epoch_loss += loss.item()

            grad_q, q, _ = quantize_grad_proper(grad, precision)
            q_vals.append(float(q.item()))

            old_grad_q = grad_memory[i].clone()

            update = grad_q - old_grad_q + grad_avg

            w -= lr * update

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


def train_adam(
    w: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr: float,
    precision: str,
    stochastic_round: bool = True,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    n, d = X.shape

    m = torch.zeros(d, device=DEVICE)
    v = torch.zeros(d, device=DEVICE)
    t = 0

    train_losses = []
    test_accs = []
    shrink_qs = []

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        q_vals = []

        for i in perm:
            xi = X[i]
            yi = y[i]

            loss, grad = logistic_loss_and_grad(w, xi, yi)

            grad_q, q, _ = quantize_grad_proper(
                grad, precision, stochastic_round=stochastic_round
            )
            q_vals.append(float(q.item()))

            t += 1
            m = beta1 * m + (1.0 - beta1) * grad_q
            v = beta2 * v + (1.0 - beta2) * (grad_q**2)

            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            w -= lr * m_hat / (torch.sqrt(v_hat) + eps)

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / n)
        test_accs.append(accuracy_pm1(w, X_test, y_test))
        shrink_qs.append(float(np.mean(q_vals)))

        print(
            f"[ADAM | {precision.upper()} | lr={lr:.0e}] "
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"loss={train_losses[-1]:.6f} | "
            f"acc={test_accs[-1]:.4f} | "
            f"mean q={shrink_qs[-1]:.4f}"
        )

    return (
        np.array(train_losses),
        np.array(test_accs),
        np.array(shrink_qs),
    )


def train_sag(w, X, y, X_test, y_test, lr, precision, stochastic_round=True):
    n, d = X.shape
    grad_memory = torch.zeros((n, d), device=DEVICE)
    grad_avg = torch.zeros(d, device=DEVICE)

    train_losses = []
    test_accs = []
    shrink_qs = []

    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        q_vals = []

        for i in perm:
            xi = X[i]
            yi = y[i]

            loss, grad = logistic_loss_and_grad(w, xi, yi)

            grad_q, q, _ = quantize_grad_proper(
                grad, precision, stochastic_round=stochastic_round
            )
            q_vals.append(float(q.item()))

            grad_avg += (grad_q - grad_memory[i]) / n
            grad_memory[i] = grad_q
            w -= lr * grad_avg

            epoch_loss += loss.item()

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
