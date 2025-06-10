from config import device
from Godunov import generate_synthetic_data, SimuGodunov
from models.stacked_res import StackedRes
from pinn.VanishingStackedRes import StackedPINN
from utils.Visualization import plot_error_stages, plot_four_stages, plot_base_and_resblocks_separate_eps
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    # Parameters setup
    Vf = 1.5
    gamma_ = 0.0
    L = 5
    Tmax = 2
    Nx = 500
    p = 1/20
    rhoBar = 0.5
    rhoSigma = 0.3
    noise = False
    
    def flux_unscaled(u):
        return Vf * (1 - 2*u)

    # Generate data
    (x_data, t_data, u_data,
     x_colloc, t_colloc,
     x_test, t_test, u_test, godunov_sim) = generate_synthetic_data(
        Vf=Vf, gamma=gamma_,
        L=L, Tmax=Tmax, Nx=Nx,
        p=p, rhoBar=rhoBar, rhoSigma=rhoSigma,
        zMin=0, zMax=1, noise=noise, greenshield=True
    )    
    # Initialize model
    model = StackedRes(
        insize=2,
        outsize=1,
        h_sf_sizes=[40,40,40],
        n_stacked_mf_layers=3,
        h_res_sizes=[50,50,50,50,50],
        alpha_init=0.1
    ).to(device)
    
    # Initialize and train vanishing stacked-residual PINN
    pinn = StackedPINN(
        model_density=model,
        optimizer=None,
        F_func=flux_unscaled,
        t_min=0.0, t_max=Tmax,
        x_min=0.0, x_max=L, 
        gamma_init= 0.1,
    )
    all_params = list(model.parameters())
    optimizer = optim.Adam(all_params, lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=1000)
    pinn.set_optimizer(optimizer)

    pinn.train(
        x_data, t_data, u_data,
        x_colloc, t_colloc,
        x_test, t_test, u_test,
        epochs_adam=15000,
    )

    # Visualization
    plot_error_stages(pinn, godunov_sim)
    plot_four_stages(pinn, godunov_sim)
    plot_base_and_resblocks_separate_eps(model, L, Tmax, 
                                         Nx=200, Nt=200, 
                                         output_dir="res_blocks_plots")

if __name__ == "__main__":
    main()