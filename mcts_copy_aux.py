import math
import random
import mdp
import matplotlib.pyplot as plt

DEBUG = False
"""
Implementation of Monte Carlo Tree search for the approximate
evaluation of Value functions.

This implementation is for infinite horizon, discounted MDP
with stochastic state transitions in discrete state
and action space.
"""


class Tree:
    def __init__(self, depth, parent, dist, seq, proc: mdp.mdp):
        self.depth = depth  # number with depth in the tree
        self.dist = dist  # probabiliy density function of current state | dist(state) = number in (0,1)
        self.seq = seq
        self.value = None  # this is the predicted cost of all future actions in the horizon (aka Value)
        self.visits = 0
        self.parent = parent
        self.children = []
        self.proc = proc
        self.running_cost = 0  # this is the cost incurred from the single previous action
        if depth > 0:
            self.running_cost = self.stage_cost(seq[depth - 1])

    def add_child(self, action):
        """

        :rtype: child
        """
        distribution = {}
        for state in self.proc.S:
            distribution[state] = 0
            for prev in self.proc.S:
                distribution[state] += self.dist[prev] * self.proc.P(state, prev, action)
        if DEBUG:
            print(f"distribution of new state: {[distribution[s] for s in self.proc.S]}")
        child = Tree(self.depth + 1, self, distribution, self.seq + [action], self.proc)
        self.children.append(child)
        return child

    def stage_cost(self, action):
        cost = 0
        for state in self.proc.S:
            cost += self.proc.C(state, action) * self.parent.dist[state]
        return cost

    def visit(self, episode, horizon, defaultValue):
        """
        Simulate being at this node and choosing the action.
        This assumes that selection is complete with action chosen

        steps:
        1) Select the next action to traverse down the tree
        2) Expansion - add a leaf (means adding a visit)
        3) Simulation - rollout to the horizon without adjusting visit counters
        4) Back-propogation - for the constructed tree backprop with mean values
        :return: Q value of last action
        """
        value = None
        self.visits += 1
        # is this the last in the horizon?
        if DEBUG:
            print(f"depth: {self.depth}")
        if self.depth == horizon:
            if DEBUG:
                print(f"Horizon: {horizon}")
            """return the default Value = Expected value of this distribution"""
            if self.value is not None:
                value = self.value
            else:
                value = 0
                for state in self.proc.S:
                    value += self.dist[state] * defaultValue(state)
                self.value = value
            return self.running_cost + self.proc.alpha * value

        # Is this the leaf?
        is_leaf = self.visits == 0


        # if this is a leaf then simulate the rest of the horizon with random actions
        if is_leaf:
            """Simulate"""
            value = self.simulate(episode, horizon, defaultValue)
        else:
            # if not a leaf, then select next action
            """ Select """
            selected_node = self.select()
            value = selected_node.visit(episode, horizon, defaultValue)

        """Back-propogate: self's value is the weighted average of child values"""
        if self.value is None:
            self.value = value
        else:
            self.value = ((self.visits - 1) / self.visits) * self.value + (1 / self.visits) * value
        return self.running_cost + self.proc.alpha * self.value

    def select(self):
        minB = math.inf
        selected_node = None
        child_actions = []
        for child in self.children:
            if child.visits == 0:
                return child
            if DEBUG:
                print(f"select from children: {child} in children: {self.children}")
            child_actions.append(child.seq[self.depth])
            Q = child.running_cost + self.proc.alpha * child.value
            B = Q - math.sqrt(2 * math.log(self.visits) / child.visits)
            if B < minB:
                minB = B
                selected_node = child
        for action in self.proc.U:
            if not (action in child_actions):
                selected_node = self.add_child(action)
                break
        return selected_node

    def simulate(self, episode, horizon, defaultValue):
        # go down the tree until you hit the horizon recursively
        value = None
        if self.depth == horizon:
            """return the default Value = Expected value of this distribution"""
            if self.value is not None:
                value = self.value
            else:
                value = 0
                for state in self.proc.S:
                    value += self.dist[state] * defaultValue(state)
        else:
            """traverse the tree"""
            # choose a random action
            next_action = list(self.proc.U)[random.randint(0, len(self.proc.U) - 1)]
            for child in self.children:
                if child.seq[self.depth] == next_action:
                    value = child.simulate(episode, horizon, defaultValue)
                    break
            if value is None:
                # create the requested child
                child = self.add_child(next_action)
                value = child.simulate(episode, horizon, defaultValue)

        return self.running_cost + self.proc.alpha * value


    def to_string(self, depth):
        if depth == 0:
            return ""
        s = f"{self.seq} : v = {self.value}\n"
        for child in self.children:
            s += child.to_string(depth-1)
        return s


def get_value_function():
    pass

# S = {0, 1}
# U = {0, 1}
# C = lambda state, action: abs(state - action)
# alpha = 0.9
# P = lambda next, curr, action: 0.75 if next == action else 0.25
# proc = mdp.mdp(S, U, C, alpha, P)
#
# V = lambda state : 0
# dist = {0:1, 1:0}
# seq = []
#
# vals = []
# for j in range(100):
#     root = Tree(0, None, dist, seq, proc)
#     for i in range(500):
#         root.visit(i+1, j, {0: 0, 1: 0})
#     vals.append(root.value)
#     print(f"{j}", end='\r')
#
# plt.plot(vals)
# plt.show()