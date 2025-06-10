import torch

def l2_relative_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """
    Compute L2 relative error between prediction and simulated data
    
    Args:
        u_pred: Predicted density
        u_true: Simulation-based density
        
    Returns:
        Relative error (float)
    """
    err = torch.norm(u_pred - u_true, p=2)
    denom = torch.norm(u_true, p=2)
    return (err/denom).item() if denom.item() > 1e-12 else err.item()