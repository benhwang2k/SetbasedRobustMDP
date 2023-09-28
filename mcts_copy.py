import math
import random
import mdp
import matplotlib.pyplot as plt
import time

DEBUG = False
"""
Implementation of Monte Carlo Tree search for the approximate
evaluation of Value functions.

This implementation is for infinite horizon, discounted MDP
with stochastic state transitions in discrete state
and action space.
"""


class Tree:
    def __init__(self, depth, parent, dist, action, mdpseq):
        self.depth = depth  # number with depth in the tree
        self.dist = dist  # probabiliy density function of current state | dist(state) = number in (0,1)
        self.action = action  # this is the action that resulted in getting here
        self.value = None  # this is the predicted cost of all future actions in the horizon (aka Value)
        self.visits = 0
        self.parent = parent
        self.children = []
        self.mdpseq = mdpseq
        self.proc = mdpseq[self.depth]
        self.running_cost = 0  # this is the cost incurred from the single previous action
        if not action is None:
            for state in self.proc.S:
                self.running_cost += self.parent.dist[state] * self.proc.C(state, action)

    def add_child(self, action):
        next_dist = {}
        for state in self.proc.S:
            next_dist[state] = sum([(self.dist[prev] * self.proc.P(state, prev, action)) for prev in self.proc.S])
        child = Tree(self.depth + 1, self, next_dist, action, self.mdpseq)
        self.children.append(child)
        return child

    def rollout(self, horizon, defaultvalue):
        """
        simulate the rest of the process using random actions
        :return: the accumulated cost of the applying a random policy for the rest of the horizon
        return value = self's value which is = next runnign cost + alpha * next value
        """
        if self.depth == horizon:
            if self.value is None:
                self.value = sum([(self.dist[state]*defaultvalue(state)) for state in self.proc.S])
            return self.value
        else:
            # choose random action
            a = list(self.proc.U)[random.randint(0,len(self.proc.U)-1)]
            for child in self.children:
                if child.action == a:
                    v = child.running_cost + child.proc.alpha * child.rollout(horizon, defaultvalue)
                    return v
            child = self.add_child(a)
            v = child.running_cost + child.proc.alpha * child.rollout(horizon, defaultvalue)
            return v

    def select(self):
        """
        Select process from MCTS
        :return: Best action to explore next
        """
        minB = math.inf
        selected_node = None
        child_actions = []
        for child in self.children:
            if child.visits == 0:
                return child
            child_actions.append(child.action)
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

    def visit(self, horizon, defaultvalue):
        """
        recursively updates the value of this node by traversing the tree ->

        Select : select which action branch to explore
        Expand : add new nodes up to the horizon
        Rollout: simulate the rest of the horizon with random actions to get an approximate value for this node
        Backup: update values of the tree along the explored branch

        :param horizon: time horizon is the max depth of the tree starting at 0
        :param defaultvalue: value used as the value for the nodes at the horizon
        :return: return the updated value of the tree
        """
        self.visits += 1
        value = 0
        # If this is a leaf node, then expand
        if self.visits == 1 or self.depth == horizon - 1:
            value = self.rollout(horizon, defaultvalue)
        else:   # If not a leaf then Select a branch down which to explore
            value = self.select().visit(horizon, defaultvalue)
        # update this value
        if self.value is None:
            self.value = value
        else:
            self.value = ((self.visits - 1)/self.visits)*self.value + (1/self.visits)*value
        # return Q for previous nodes to update
        return self.running_cost + self.proc.alpha * self.value


    def get_info(self):
        s = f"action={self.action}\n"
        s += f"visits={self.visits}\n"
        s += f"value={self.value}\n"
        s += f"running cost={self.running_cost}\n"
        return s

    def print_dist(self):
        x = []
        y = []
        for state in self.proc.S:
            x.append(state)
            y.append(self.dist[state])
        plt.xlabel(f"running cost : {self.running_cost}")
        plt.plot(x, y, 'ro')

    def print_tree(self):
        fig, ax = plt.subplots()
        plt.xlim(-10,10)
        plt.ylim(-0.5,10)
        self.print_tree_helper(ax, 0.4)


    def print_tree_helper(self, ax, x):
        text = f"a : {round(self.action,3) if self.action else self.action} \n l = {round(self.running_cost,3) if self.running_cost else self.running_cost} \n v = {round(self.value,3) if self.value else self.value} "
        ax.text(x, 0.2*self.depth, text, fontsize=10, color='black',
                bbox=dict(facecolor='none', edgecolor='black'))
        separation = 5
        y = 0.2*self.depth
        i = 0
        l = len(self.children) - 1
        offset = l/2 * (separation/(self.depth+1))
        for child in self.children:
            x_next = x - offset + (separation*i/(self.depth+1))
            y_next = 0.2*child.depth
            plt.plot([x, x_next], [y, y_next], "blue")
            child.print_tree_helper(ax, x - offset + (separation*i/(self.depth+1)))
            i += 1


def rollout_policy(pi, mdp, horizon):
    V = {}
    for s in mdp.S:
        V[s] = 0
    for s0 in mdp.S:
        dist = {}
        for state in mdp.S:
            dist[state] = 1 if state == s0 else 0
        for h in range(horizon):
            V[s0] += (mdp.alpha ** h) * sum([(dist[s]*mdp.C(s,pi[s])) for s in mdp.S])
            next_dist = {}
            for state in mdp.S:
                next_dist[state] = sum([dist[prev]*mdp.P(state,prev,pi[prev]) for prev in mdp.S])
            dist = next_dist
    return V

def rollout_time_policy(pis, mdp, horizon):
    V = {}
    for s in mdp.S:
        V[s] = 0
    for s0 in mdp.S:
        dist = {}
        for state in mdp.S:
            dist[state] = 1 if state == s0 else 0
        for h in range(horizon):
            V[s0] += (mdp.alpha ** h) * sum([(dist[s]*mdp.C(s,pis[h][s])) for s in mdp.S])
            next_dist = {}
            for state in mdp.S:
                next_dist[state] = sum([dist[prev]*mdp.P(state,prev,pis[h][prev]) for prev in mdp.S])
            dist = next_dist
    return V


def construct_path(root: Tree):
    pi = []
    a = []
    curr = root
    while not (curr.children == []):
        minV = math.inf
        best_action = None
        bset_child = None
        for child in curr.children:
            if child.value is None:
                continue
            transition_value = child.running_cost + child.proc.alpha * child.value
            if transition_value < minV:
                minV = transition_value
                best_action = child.action
                best_child = child
        pi.append(best_child)
        a.append(best_action)
        curr = best_child
    # follow the path set by pi, collecting running costs
    print(f"policy : {a}")
    V = 0
    depth = 0
    for node in pi:
        V += (node.proc.alpha ** depth) * node.running_cost
    return V


def get_policy(mdpseq, budget, horizon):
    """
    :param horizon: 
    :param mdpseq: 
    :param budget: seconds : time budget for each value calculation
    :return: policy dictionary key state => optimal action
    """
    states = mdpseq[0].S
    pi = {}
    defaultvalue = lambda state: 0
    for initial_state in states:
        dist = {}
        for state in proc.S:
            dist[state] = 1 if state == initial_state else 0
        root = Tree(0,None,dist,None,mdpseq)
        tic = time.time()
        n = 0
        while time.time() - tic < budget:
            print(f"time elapsed: {round(time.time() - tic, 4)}, budget: {budget}", end='\r')
            root.visit(horizon, defaultvalue)
            n += 1
        print(f"time elapsed: {round(time.time() - tic, 10)}, budget: {budget}, n {n}")
        pi[initial_state] = list(mdpseq[0].U)[0]
        minQ = math.inf
        for child in root.children:
            if not child.value is None:
                Q = child.running_cost + child.proc.alpha * child.value
                if Q < minQ:
                    pi[initial_state] = child.action
                    minQ = Q
    return pi

# compare against real as calculated by Bellman
def bellman(depth):
    V = {}
    a = {}
    for state in S:
        V[state] = 0
    while depth > 0:
        print(f"depth: {depth}", end='\r')
        V_next = {}
        for state in S:
            minAction = U[0]
            minVal = math.inf
            for u in U:
                v = proc.C(state, u) + proc.alpha * (sum([(proc.P(x, state, u) * V[x]) for x in S]))
                if v < minVal:
                    minVal = v
                    minAction = u
            V_next[state] = minVal
            a[state] = minAction
        V = V_next
        depth -= 1
    # print(f"bellmanV : {V}")
    # print(f"actions : {a}")
    return V
