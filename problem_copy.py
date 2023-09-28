import math
import random
from functools import partial
import mdp

DEBUG = True


def make_simple(V, h):
    def simpleV(myV, x):
        x_index = math.floor(x/h)
        return myV[x_index]
    return partial(simpleV, V)


class FintieProblem:

    def __init__(self, n, m, G, alpha, p):
        self.n = n
        self.m = m
        self.states = list(range(n))
        self.controls = list(range(m))
        self.G = G  # costs are given as a set of continuous functions
        self.alpha = alpha
        self.p = p

        self.max_cost = 0

    # def caculate_optimal_value(self, tolerance):
    #     """
    #     This function calculates the optimal value of being at any state in the space of the problem.
    #     The tolerance dictates how far into the future the simulation will be run.
    #     :param tolerance: = (1/(1-alpha)) - finite_calculation
    #     :return: map of states to values
    #     """
    #     Xi = {tuple([0]*self.n)}
    #     Xi_next =

    def Bellman(self, V, g):
        """
        Applies the Bellman Operator (T) to the value function V
        :param V: tuple of values associated with states in order
        :param g: stage-cost
        :return: T(V)
        """
        V_next = [0] * self.n
        for j in range(self.n):
            V_next[j] = g(self.states[j], self.controls[0]) + self.alpha * self.expectedValue(V, self.states[j],
                                                                                              self.controls[0])
            for u in self.controls:
                W = g(self.states[j], u) + self.alpha * self.expectedValue(V, self.states[j], u)
                if W < V_next[j]:
                    V_next[j] = W
        return tuple(V_next)

    def set_Bellman(self, V_0, horizon):
        V = set([V_0])
        for k in range(horizon):
            V_next = []
            for v in V:
                for g in self.G:
                    V_next.append(self.Bellman(v, g))
            V = set(V_next)
        return V

    def expectedValue(self, V, x, u):
        """
        Gives the expected value of value function V given control u.
        :param V:
        :param u:
        :return:
        """
        E = 0
        for y in range(self.n):
            E += self.p(y, x, u) * V[y]
        return E


class ContinuousProblem:
    def __init__(self, h, G, alpha, p):
        self.h = h
        self.n = math.ceil(1 / h)
        self.states = []
        self.controls = []
        for i in range(self.n):
            # pick random state for th representative
            upper = min([i*h + h, 1.0])
            self.states.append(random.random() * (upper-i*h) + i * h)
            # use corners as controls
            self.controls.append(i * h)

        # create discrete costs
        self.G = []  # costs are given as a set of continuous functions need discrete

        def gi(g, x, u):
            x_disc = self.states[math.floor(x / h)]
            return g(x_disc, u)

        for g in G:
            self.G.append(partial(gi, g))

        self.alpha = alpha

        # discrete transition
        def p_disc(y, x, u):
            y_disc = self.states[math.floor(y / h)]
            x_disc = self.states[math.floor(x / h)]
            return h * p(y_disc, x_disc, u) / sum([h * p(z, x_disc, u) for z in self.states])

        self.p = p_disc

    def get_mdp_set(self):
        MDPs = []
        for g in self.G:
            MDPs.append(mdp.mdp(self.states, self.controls, g, self.alpha, self.p))
        return MDPs


    # def caculate_optimal_value(self, tolerance):
    #     """
    #     This function calculates the optimal value of being at any state in the space of the problem.
    #     The tolerance dictates how far into the future the simulation will be run.
    #     :param tolerance: = (1/(1-alpha)) - finite_calculation
    #     :return: map of states to values
    #     """
    #     Xi = {tuple([0]*self.n)}
    #     Xi_next =

    def Bellman(self, V, g):
        """
        Applies the Bellman Operator (T) to the value function V
        :param V: tuple of values associated with states in order
        :param g: stage-cost
        :return: T(V)
        """
        V_next = [0] * self.n
        for j in range(self.n):
            V_next[j] = g(self.states[j], self.controls[0]) + self.alpha * self.expectedValue(V, self.states[j],
                                                                                              self.controls[0])
            for u in self.controls:
                W = g(self.states[j], u) + self.alpha * self.expectedValue(V, self.states[j], u)
                if W < V_next[j]:
                    V_next[j] = W
        return tuple(V_next)

    def set_Bellman(self, V_set, horizon):
        V = V_set
        for k in range(horizon):
            V_next = []
            for v in V:
                for g in self.G:
                    V_next.append(self.Bellman(v, g))
            V = set(V_next)
        return V

    def expectedValue(self, V, x, u):
        """
        Gives the expected value of value function V given control u.
        :param V:
        :param u:
        :return:
        """
        E = 0
        for y_ind in range(self.n):
            E += self.h * self.p(self.states[y_ind], x, u) * V[y_ind]
        return E
