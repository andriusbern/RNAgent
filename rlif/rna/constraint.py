"""
A Target object can have multiple contraint objects associated with it
When a new Solution gets initialized it inherits and adds all the constraints 
automatically
"""

class Constraint:
    def __init__(self, seq=''):
        self.start = 0
        self.sequence = self.parse(seq)

    def parse(self, seq):
        """
        Possible implementations:
            Each nucl has forbidden bases (at an action check available bases at i and j)
            Insert only available actions (could have deadlocks (some of the seq constraints could make all of them impossible))
            Make checks for constraints?
        """
        alphabet = {
            'A':'a'
        }

    def __len__(self):
        return len(self.sequence)
