from .utils import initialize, visualize
from . import distributions
from .distributions import priors, selection
from . import samplers, simulators, solvers
from .utils.preprocessing import preprocess_data
import sys

# Make these accessible as proper top-level submodules
sys.modules[__name__ + ".initialize"] = initialize
sys.modules[__name__ + ".visualize"] = visualize

__all__ = [
    'initialize',
    'visualize',
    'distributions',
    'priors',
    'selection',
    'samplers',
    'simulators',
    'solvers',
    'preprocess_data',
]

from importlib.metadata import version
__version__ = version('simplebayesn')
