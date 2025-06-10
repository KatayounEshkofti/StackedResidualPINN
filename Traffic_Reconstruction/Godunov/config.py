import torch 

# Device configuration

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA.")
else:
    device = torch.device('cpu')
    print("Using CPU.")

# Set seeds for reproducibility
SEED = 12345
torch.manual_seed(SEED)

# Export device as a module-level variable
__all__ = ['device', 'SEED']