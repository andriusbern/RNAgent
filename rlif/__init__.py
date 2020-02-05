import sys
from .settings import get_parameters
# Loading compatibility
import rlif
import rlif.learning as learning
import rlif.learning.networks as networks
sys.modules['rlfold'] = rlif
sys.modules['rlfold.baselines'] = learning
sys.modules['rlfold.baselines.CustomPolicies'] = networks


# from .learning import RLIF