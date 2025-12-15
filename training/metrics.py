import torch
import torch.nn.functional as F

@torch.no_grad()
def accuracy_pm1(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    preds = torch.where(X @ w >= 0, 1.0, -1.0)
    return (preds == y).float().mean().item()

def logistic_loss_and_grad(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    # y in {-1, +1}
    t = y * torch.dot(w, x)
    loss = F.softplus(-t)
    grad = -(y * x) * torch.sigmoid(-t)
    return loss, grad