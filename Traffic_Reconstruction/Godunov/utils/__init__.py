# utils/__init__.py
"""
Utility Functions

Contains:
- resampling: Adaptive collocation point sampling
- l2error : Computing l2 relative error 
- visualization: Plotting and visualization tools
"""

from .resampling import adaptive_resample
from .l2error import l2_relative_error
from .Visualization import (
    plot_residual_block_contributions,
    plot_spatiotemporal_error,
    plot_four_stages,
    plot_error_stages, 
    plot_base_and_resblocks_separate_eps
)

__all__ = [
    'adaptive_resample',
    'l2_relative_error',
    'plot_residual_block_contributions',
    'plot_spatiotemporal_error',
    'plot_four_stages',
    'plot_error_stages',
    'plot_base_and_resblocks_separate_eps',
]