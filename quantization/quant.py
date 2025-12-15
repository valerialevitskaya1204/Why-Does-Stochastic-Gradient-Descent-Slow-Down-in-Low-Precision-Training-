import torch

@torch.no_grad()
def quantize_symmetric_uniform(
    g: torch.Tensor,
    bits: int,
    stochastic_round: bool = True,
):
    """
    Symmetric uniform quantization using per-vector max(|g|) scaling.
    If stochastic_round=True, uses stochastic rounding for (approximately) unbiased noise.
    """
    max_val = g.abs().max()
    if max_val == 0:
        return g.clone()

    qmax = (2 ** (bits - 1)) - 1 
    scale = max_val / qmax
    y = g / scale

    if stochastic_round:
        y_floor = torch.floor(y)
        frac = y - y_floor
        u = torch.rand_like(frac)
        yq = y_floor + (u < frac).to(g.dtype)
    else:
        yq = torch.round(y)

    yq = torch.clamp(yq, -qmax, qmax)
    return yq * scale

@torch.no_grad()
def decompose_shrinkage(g: torch.Tensor, g_tilde: torch.Tensor, eps: float = 1e-12):
    """
    Shrinkage factor as in the paper:
        q = ||g_tilde||_2 / ||g||_2

    Returns:
        q: scalar shrinkage factor (can be < 1 or = 1)
        eps_vec: residual noise g_tilde - q * g
    """
    g_norm = g.norm()
    if g_norm < eps:
        q = torch.tensor(1.0, device=g.device, dtype=g.dtype)
        eps_vec = g_tilde - g
        return q, eps_vec

    g_tilde_norm = g_tilde.norm()
    q = g_tilde_norm / (g_norm + eps)

    eps_vec = g_tilde - q * g
    return q, eps_vec



@torch.no_grad()
def quantize_fp_like(g, mant_bits, exp_bits):
    abs_g = g.abs()
    mask = abs_g > 0

    exp = torch.zeros_like(g)
    exp[mask] = torch.floor(torch.log2(abs_g[mask]))

    #  экспонента
    max_exp = 2 ** (exp_bits - 1) - 1
    min_exp = -max_exp
    exp = torch.clamp(exp, min_exp, max_exp)

    # мантисса
    mant = g / (2.0 ** exp)
    mant_levels = 2 ** mant_bits
    mant_q = torch.round(mant * mant_levels) / mant_levels
    mant_q = torch.clamp(mant_q, -2.0, 2.0)

    return mant_q * (2.0 ** exp)



@torch.no_grad()
def quantize_grad_proper(grad: torch.Tensor, precision: str):
    """
    Returns:
      grad_tilde: quantized gradient
      q: shrinkage factor in (0,1]
      eps: residual noise in decomposition grad_tilde = q*grad + eps
    """
    if precision == "fp4":
        g_tilde = quantize_fp_like(grad, mant_bits=1, exp_bits=2)
    elif precision == "fp8":
        g_tilde = quantize_fp_like(grad, mant_bits=3, exp_bits=4)
    elif precision == "fp16":
        g_tilde = grad.half().float()
    elif precision == "fp32":
        g_tilde = grad
    else:
        raise ValueError(f"Unknown precision: {precision}")

    q, eps = decompose_shrinkage(grad, g_tilde) #подсчет q
    return g_tilde, q, eps


def quantize_symmetric(grad, bits):
    max_val = grad.abs().max()
    if max_val == 0:
        return grad
    qmax = (2 ** (bits - 1)) - 1
    scale = max_val / qmax
    q = torch.round(torch.clamp(grad / scale, -qmax, qmax))
    return q * scale

def apply_gradient_quantization(model, precision='fp32'):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                if precision == "fp4":
                    return quantize_symmetric(param.grad, 4)
                elif precision == "fp8":
                    return quantize_symmetric(param.grad, 8)
                elif precision == "fp16":
                    return param.grad.half().float()
                else:
                    return param.grad