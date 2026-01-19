from .utils import initialize, visualize, io
from . import distributions
from .distributions import priors, selection
from . import samplers, simulators, solvers
from .distributions.likelihood import marginal_loglikelihood
from .utils.preprocessing import preprocess_data
import sys

# Make these accessible as proper top-level submodules
sys.modules[__name__ + ".initialize"] = initialize
sys.modules[__name__ + ".visualize"] = visualize
sys.modules[__name__ + ".io"] = io

__all__ = [
    'initialize',
    'visualize',
    'io',
    'distributions',
    'priors',
    'selection',
    'samplers',
    'simulators',
    'solvers',
    'marginal_loglikelihood',
    'preprocess_data',
]

from importlib.metadata import version
__version__ = version('simplebayesn')
