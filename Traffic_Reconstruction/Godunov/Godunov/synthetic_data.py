#%%
from .GodunovSimulation import SimuGodunov  
import numpy as np
import torch

############################################################################
#           GENERATE SYNTHETIC DATA FROM GODUNOV SIMULATION
############################################################################

def generate_synthetic_data(
        Vf, gamma, L, Tmax, Nx, p, rhoBar, rhoSigma,
        zMin=0, zMax=1, noise=False, greenshield=True
    ):
    # Generate empty probe positions and times to avoid using probes
    xiPos = np.array([])
    xiT = np.array([])

    godunov = SimuGodunov(
        Vf=Vf, gamma=gamma, xiPos=xiPos, xiT=xiT,
        zMin=zMin, zMax=zMax, L=L, Tmax=Tmax,
        Nx=Nx, rhoBar=rhoBar, rhoSigma=rhoSigma, greenshield=greenshield
    )
    z_res = godunov.simulation()
    godunov.plot()
    # Define grid parameters 
    delta_L = L / 100       # ΔL = L / number_of_points
    delta_t = Tmax / 100    # Δt = Tmax / number_of_points
    
    # Spatial grid: x = [ΔL, 2ΔL, ..., L - ΔL]
    x_start = delta_L
    x_end = L - delta_L
    x_sel_grid = np.linspace(x_start, x_end, int((L - 2*delta_L)/delta_L) + 1)
    
    # Time grid: t = [Δt, 2Δt, ..., Tmax]
    t_start = delta_t
    t_end = Tmax
    t_sel_grid = np.linspace(t_start, t_end, int((Tmax - delta_t)/delta_t) + 1)
    
    # Create meshgrid for spatiotemporal sampling
    X, T = np.meshgrid(x_sel_grid, t_sel_grid)
    X_flat = X.flatten()
    T_flat = T.flatten()
    
    # Convert to indices in the simulation grid
    x_indices = np.clip((X_flat / godunov.sim.deltaX).astype(int), 0, godunov.sim.Nx - 1)
    t_indices = np.clip((T_flat / godunov.sim.deltaT).astype(int), 0, godunov.sim.Nt - 1)
    
    # Extract density values
    rho_sel_flat = z_res[x_indices, t_indices]
    
    # Apply noise if required
    if noise:
        noise_vals = np.random.normal(0, 0.02, rho_sel_flat.shape)
        rho_sel_flat = np.clip(rho_sel_flat + noise_vals, 0.0, 1.0)
    
    # Convert to tensors (shape [num_points, 1])
    x_sel = torch.tensor(X_flat.reshape(-1, 1), dtype=torch.float32)
    t_sel = torch.tensor(T_flat.reshape(-1, 1), dtype=torch.float32)
    rho_sel = torch.tensor(rho_sel_flat.reshape(-1, 1), dtype=torch.float32)
    
    # Collocation points
    N_colloc = 10000 
    x_f = torch.rand(N_colloc, 1) * L
    t_f = torch.rand(N_colloc, 1) * Tmax

    # Test data 
    Nx_test = 500
    Nt_test = 500
    x_lin = np.linspace(0, L, Nx_test)
    t_lin = np.linspace(0, Tmax, Nt_test)
    Xg, Tg = np.meshgrid(x_lin, t_lin)
    Xg_1d = Xg.flatten()
    Tg_1d = Tg.flatten()
    X_idx = np.clip((Xg_1d / godunov.sim.deltaX).astype(int), 0, godunov.sim.Nx - 1)
    T_idx = np.clip((Tg_1d / godunov.sim.deltaT).astype(int), 0, godunov.sim.Nt - 1)
    u_true_flat = z_res[X_idx, T_idx]

    x_test = torch.tensor(Xg_1d, dtype=torch.float32).view(-1, 1)
    t_test = torch.tensor(Tg_1d, dtype=torch.float32).view(-1, 1)
    u_test = torch.tensor(u_true_flat, dtype=torch.float32).view(-1, 1)

    return x_sel, t_sel, rho_sel, x_f, t_f, x_test, t_test, u_test, godunov
