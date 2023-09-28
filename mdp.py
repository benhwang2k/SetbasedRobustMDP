

"""
This file contains the class definition of Markov decision processes
"""

class mdp:

    def __init__(self, S, U, C, alpha, P):
        self.S = S              # set of states
        self.U = U              # set of controls
        self.C = C              # cost function C: (state,action) -> cost
        self.alpha = alpha      # number in (0,1)
        self.P = P              # probability density P(next_state, curr_state, action) -> prob of next state
        self.pi = lambda s : list(self.U)[0]
        self.name = "default"

    def setName(self, name):
        self.name = name

    def setPolicy(self, pi):
        self.pi = pi

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"{self.name}"









