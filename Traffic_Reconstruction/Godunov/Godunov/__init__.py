'''
 Godunov Simulation Module 
 
 Contains: 
 
 GodunovSim: Godunov Simulation
 generate_synthetic_data: Creates training and test data

'''

from .GodunovSimulation import SimuGodunov
from .synthetic_data import generate_synthetic_data

__all__ = [
    'SimuGodunov',
    'generate_synthetic_data'
]