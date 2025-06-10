
import torch 

# ----------------------------------------------------------------------
    # Method for dynamic collocation re-sampling
    # ----------------------------------------------------------------------
def adaptive_resample(pinn, x_current, t_current, n_new=500):
    """
    Evaluate PDE residual on a large random set of points, pick those
    with highest residual, and append them to the existing collocation set.

    Args:
        x_current (Tensor): existing collocation x points, shape [N, 1]
        t_current (Tensor): existing collocation t points, shape [N, 1]
        n_new (int): number of new points to select from random sampling

    Returns:
        (x_updated, t_updated): The updated collocation set.
    """
    # 1) Generate many random points in [x_min, x_max] x [t_min, t_max]
    N_eval = 10000
    x_rand = torch.rand(N_eval, 1)*(pinn.x_max - pinn.x_min) + pinn.x_min
    t_rand = torch.rand(N_eval, 1)*(pinn.t_max - pinn.t_min) + pinn.t_min

    # 2) Compute PDE residual at these random points
    residuals = pinn.compute_residuals(x_rand, t_rand)

    # --------------------- FIX STARTS HERE ---------------------
    
    # Convert residuals from shape (N,1) -> (N,)
    residuals = residuals.view(-1)

    # Ensure n_new does not exceed residuals.size(0)
    n_new = min(n_new, residuals.size(0))

    # 3) Select the top 'n_new' points with the largest PDE residual
    vals, idx_top = torch.topk(residuals, n_new)
    
    # --------------------- FIX ENDS HERE -----------------------

    x_new = x_rand[idx_top]
    t_new = t_rand[idx_top]

    # 4) Append these new points to existing collocation set
    x_updated = torch.cat([x_current, x_new], dim=0)
    t_updated = torch.cat([t_current, t_new], dim=0)

    print(f"[adaptive_resample] Added {n_new} points with highest residuals. "
        f"New collocation set size = {x_updated.shape[0]}.")

    return x_updated, t_updated