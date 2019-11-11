import sys

# Loading compatibility
import rlif
import rlif.learning as learning
import rlif.learning.networks as networks
sys.modules['rlfold'] = rlif
sys.modules['rlfold.baselines'] = learning
sys.modules['rlfold.baselines.CustomPolicies'] = networks
