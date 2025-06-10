#%% Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from ..config import device

#%%####################################################################################
#        Additional function to visualize the contribution of each ResNet block 
#######################################################################################
def plot_residual_block_contributions(model, L, Tmax, Nx=500, Nt=500):
    x_vals = np.linspace(0, L, Nx)
    t_vals = np.linspace(0, Tmax, Nt)
    X, T = np.meshgrid(x_vals, t_vals)

    x_tensor = torch.FloatTensor(X.ravel().reshape(-1, 1)).to(device)
    t_tensor = torch.FloatTensor(T.ravel().reshape(-1, 1)).to(device)
    x_arg = torch.cat([x_tensor, t_tensor], dim=1)

    out_old = model.first_layer(x_arg)
    residuals_list = []

    for i in range(model.n_stacked_mf_layers):
        res_block = model.layers[i]
        residual_input = torch.cat([x_arg, out_old], dim=1)
        residual = res_block.res_mlp(residual_input)

        residual_2D = residual.detach().cpu().numpy().reshape(Nt, Nx)
        residuals_list.append(residual_2D)

        alpha_i = torch.abs(model.alpha[i])
        out_old = out_old + alpha_i * residual

    plt.figure(figsize=(6 * model.n_stacked_mf_layers, 5))

    for i, block_residual in enumerate(residuals_list):
        plt.subplot(1, model.n_stacked_mf_layers, i+1)
        sns.set(style="whitegrid")  # Clean background
        plt.rcParams["text.usetex"] = True
        plt.pcolormesh(T, X, block_residual, shading='auto', cmap='seismic')
        plt.colorbar()
        plt.xlabel(r"$x$", fontsize=14, labelpad=10)
        plt.ylabel(r"$t$", fontsize=14, labelpad=10)
        plt.title(rf"Residual Block {i+1}", fontsize=16, pad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    plt.suptitle(r"Contribution of Each ResNet Block (F([x, out_old]))")
    plt.tight_layout()
    plt.show()
####################################################################################
#           Spatio-temporal error between PINN prediction and godunov
####################################################################################
def plot_spatiotemporal_error(pinn, godunov_sim):
    # Retrieve L and Tmax from the Godunov simulation object
    L = godunov_sim.sim.L
    Tmax = godunov_sim.sim.Tmax

    # Calculate adjusted endpoints to exclude the final value
    num_points = 100
    x_end = L - (L / num_points)
    t_end = Tmax - (Tmax / num_points)

    # Generate test points using PyTorch's linspace without endpoint
    x_test = torch.linspace(0, x_end, num_points).view(-1, 1)
    t_test = torch.linspace(0, t_end, num_points).view(-1, 1)

    # Create meshgrid with 'ij' indexing for correct dimensions
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    X_flat = X.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    
    # Get predictions from the model
    with torch.no_grad():
        U_pred = pinn.predict(X_flat, T_flat)
    
    # Obtain true density values from the Godunov simulation
    U_true = godunov_sim.getDatas(X_flat.numpy(), T_flat.numpy())
    
    # Compute absolute error
    error = np.abs(U_pred.cpu().numpy() - U_true)
    
    # Plot the spatiotemporal error
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(X.numpy(), T.numpy(), error.reshape(X.shape), 
                   shading='auto', cmap='rainbow')
    plt.colorbar(label=r'Absolute Error')
    plt.xlabel(r'Position', fontsize=14, labelpad=10)
    plt.ylabel(r'Time', fontsize=14, labelpad=10)
    plt.title(r'Spatiotemporal Absolute Error: PINN vs Godunov', fontsize=16, pad=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#%%##################################################################
# Real Density vs. Predicted Density Plot In each stacked Layer
#####################################################################

def plot_four_stages(pinn, godunov_sim, 
                     fname_true="true_density.eps",
                     fname_layer1="layer1_density.eps",
                     fname_layer2="layer2_density.eps",
                     fname_layer3="layer3_density.eps"):
    """
    Creates 4 separate EPS plots:
     1) Godunov true density
     2) PINN first stacked layer
     3) PINN second stacked layer
     4) PINN final stacked layer
    """
    # Retrieve dimensions from the Godunov simulation
    L = godunov_sim.sim.L
    Tmax = godunov_sim.sim.Tmax
    Nx = godunov_sim.sim.Nx
    Nt = godunov_sim.sim.Nt
    
    # Create space-time mesh that matches the Godunov data shape
    x_vals = np.linspace(0, L, Nx)
    t_vals = np.linspace(0, Tmax, Nt)
    X, T = np.meshgrid(x_vals, t_vals, indexing='ij')  # X.shape = (Nx, Nt)
    
    # Flatten for PINN prediction
    X_flat = X.ravel().reshape(-1,1)
    T_flat = T.ravel().reshape(-1,1)
    
    # 1) True Godunov density
    U_true = godunov_sim.z  # shape (Nx, Nt)
    
    # 2) Model predictions at each stacked layer
    #    Note: layer indices are 1-based in  StackedRes
    #          i=1 → first stacked layer
    #          i=2 → second stacked layer
    #          i=3 → final (third) stacked layer
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    T_tensor = torch.tensor(T_flat, dtype=torch.float32)
    with torch.no_grad():
        U_pred_1 = pinn.predict(X_tensor, T_tensor, layer_index=1).cpu().numpy().reshape(X.shape)
        U_pred_2 = pinn.predict(X_tensor, T_tensor, layer_index=2).cpu().numpy().reshape(X.shape)
        U_pred_3 = pinn.predict(X_tensor, T_tensor, layer_index=3).cpu().numpy().reshape(X.shape)
    U_pred_1 = U_pred_1.reshape(X.shape)
    U_pred_2 = U_pred_2.reshape(X.shape)
    U_pred_3 = U_pred_3.reshape(X.shape)
    # -- PLOT 1: True Godunov --
    plt.figure()
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(T, X, U_true, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)", fontsize=12)
    plt.ylabel(r"Position (km)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label=r"Density")
    plt.savefig(fname_true, format='eps')
    plt.close()
    
    # -- PLOT 2: PINN layer 1 --
    plt.figure()
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(T, X, U_pred_1, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)", fontsize=12)
    plt.ylabel(r"Position (km)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label=r"Density")
    plt.savefig(fname_layer1, format='eps')
    plt.close()
    
    # -- PLOT 3: PINN layer 2 --
    plt.figure()
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(T, X, U_pred_2, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)", fontsize=12)
    plt.ylabel(r"Position (km)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label=r"Density")
    plt.savefig(fname_layer2, format='eps')
    plt.close()
    
    # -- PLOT 4: PINN layer 3 (final) --
    plt.figure()
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(T, X, U_pred_3, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)", fontsize=12)
    plt.ylabel(r"Position (km)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label=r"Density")
    plt.savefig(fname_layer3, format='eps')
    plt.close()

#%%###############################################################
# visualise signed error  \hat u^{(i)} - u  per layer
# ################################################################
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

def plot_error_stages(pinn, godunov_sim,
                      fname_true="true_density.eps",
                      fname_layer1="layer1_error.eps",
                      fname_layer2="layer2_error.eps",
                      fname_layer3="layer3_error.eps"):
    """
    Creates 4 separate EPS plots:
     1) Godunov true density
     2) error first stacked layer
     3) error second stacked layer
     4) error final stacked layer
    """
    # Retrieve dimensions from the Godunov simulation
    L = godunov_sim.sim.L
    Tmax = godunov_sim.sim.Tmax
    Nx = godunov_sim.sim.Nx
    Nt = godunov_sim.sim.Nt

    # Create space-time mesh that matches the Godunov data shape
    x_vals = np.linspace(0, L, Nx)
    t_vals = np.linspace(0, Tmax, Nt)
    X, T = np.meshgrid(x_vals, t_vals, indexing='ij')  # X.shape = (Nx, Nt)

    # Flatten for PINN prediction
    X_flat = X.ravel().reshape(-1,1)
    T_flat = T.ravel().reshape(-1,1)
    X_tensor = torch.tensor(X_flat, dtype=torch.float32)
    T_tensor = torch.tensor(T_flat, dtype=torch.float32)

    # True Godunov solution
    U_true = np.asarray(godunov_sim.z).reshape(X.shape)

    # Predict at each stacked layer and compute error maps
    preds = []
    for layer_idx in (1, 2, 3):
        with torch.no_grad():
            U_pred = pinn.predict(X_tensor, T_tensor, layer_index=layer_idx)
        preds.append(U_pred.cpu().numpy().reshape(X.shape))

    # Compute error maps
    error_maps = [pred - U_true for pred in preds]
    # Compute global maximum absolute error for consistent color scaling
    v_max = max(np.max(np.abs(err)) for err in error_maps)
    norm = TwoSlopeNorm(vmin=-v_max, vcenter=0.0, vmax=v_max)

    # Common plotting setup
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True

    # 1) True Godunov density
    plt.figure()
    plt.pcolormesh(T, X, U_true, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)", fontsize=12)
    plt.ylabel(r"Position (km)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.colorbar(label=r"Density")
    plt.savefig(fname_true, format='eps')
    plt.close()

    # Layer-wise error plots
    fnames = [fname_layer1, fname_layer2, fname_layer3]
    labels = [r"$\hat u^{(1)} - u$", r"$\hat u^{(2)} - u$", r"$\hat u^{(3)} - u$"]
    for err_map, fname, label in zip(error_maps, fnames, labels):
        plt.figure()
        plt.pcolormesh(T, X, err_map, cmap='seismic', norm=norm, shading='auto')
        plt.xlabel(r"Time (h)")
        plt.ylabel(r"Position (km)")
        plt.colorbar(label=label)
        plt.savefig(fname, format='eps')
        plt.close()

#%% #############################################################################
# Residual Contribution 
##############################################################################
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm  


def plot_base_and_resblocks_separate_eps(model, L, Tmax, 
                                         Nx=200, Nt=200, 
                                         output_dir="res_blocks_plots"):

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a mesh in x and t
    x_vals = np.linspace(0, L, Nx)
    t_vals = np.linspace(0, Tmax, Nt)
    X, T = np.meshgrid(x_vals, t_vals)  # X, T shape = (Nt, Nx)

    
    x_tensor = torch.FloatTensor(X.ravel().reshape(-1, 1)).to(device)
    t_tensor = torch.FloatTensor(T.ravel().reshape(-1, 1)).to(device)
    x_arg = torch.cat([x_tensor, t_tensor], dim=1)

    # 1) Compute the base (initial) output
    with torch.no_grad():
        out_old = model.first_layer(x_arg)  
    base_output_2D = out_old.cpu().numpy().reshape(Nt, Nx)

    # ---- Save the base output plot ----
    plt.figure()
    sns.set(style="whitegrid")  # Clean background
    plt.rcParams["text.usetex"] = True
    plt.pcolormesh(T, X, base_output_2D, shading='auto', cmap='rainbow')
    plt.xlabel(r"Time (h)")
    plt.ylabel(r"Position (km)")
    plt.colorbar(label=r"Density")
    eps_base = os.path.join(output_dir, "base_pinn_output.eps")
    plt.savefig(eps_base, format='eps')
    plt.close()

    # 2) For each residual block, compute alpha_i * residual
    for i in range(model.n_stacked_mf_layers):
        block = model.layers[i]
        residual_input = torch.cat([x_arg, out_old], dim=1)
        with torch.no_grad():
            residual = block.res_mlp(residual_input)
            alpha_i = torch.abs(model.alpha[i])
            correction = alpha_i * residual

        correction_2D = correction.cpu().numpy().reshape(Nt, Nx)
        

         # Plot and save the block's correction
        plt.figure()
        sns.set(style="whitegrid")  # Clean background
        plt.rcParams["text.usetex"] = True
        plt.pcolormesh(T, X, correction_2D, shading='auto', cmap='seismic')
        plt.colorbar(label=r"Residual Correction")
        plt.xlabel(r"Time (h)")
        plt.ylabel(r"Position (km)")
        eps_file = os.path.join(output_dir, f"block_{i+1}_correction.eps")
        plt.savefig(eps_file, format='eps')
        plt.close()

        # Update out_old for the next iteration
        out_old = out_old + correction

    # To save the final solution after all blocks:
    final_output_2D = out_old.cpu().numpy().reshape(Nt, Nx)
    plt.figure()
    plt.pcolormesh(T, X, final_output_2D, shading='auto', cmap='rainbow')
    plt.colorbar(label="Final Output")
    plt.xlabel(r"Time (h)")
    plt.ylabel(r"Position (km)")
    plt.title("Final Output (After All Residual Blocks)")
    eps_final = os.path.join(output_dir, "final_stacked_res_pinn_output.eps")
    plt.savefig(eps_final, format='eps')
    plt.close()

    print(f"All EPS files saved in '{output_dir}/'")
