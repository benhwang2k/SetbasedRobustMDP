from problem_copy import ContinuousProblem
from functools import partial
from mcts import Tree
import matplotlib.pyplot as plt

"""
The start of the problem
"""


# transition
def p(y, w, u):
    if y < u:
        return (y / u) * 2.0
    else:
        return 2.0 * (1 - y) / (1 - u)


# cost
G = []

def g(z, x, u):
    return (abs(x - u) + abs(x - z))/2

for z in [0.25, 0.75]:
    G.append(partial(g, z))

alpha = 0.6

h = 0.5

prob = ContinuousProblem(h, G, alpha, p)
mdps = prob.get_mdp_set()
print(mdps)
for proc in mdps:
    V = lambda state: 0
    dist = {}
    for state in proc.S:
        dist[state] = 0
    dist[list(proc.S)[0]] = 1
    seq = []

    root = Tree(0, None, dist, seq, proc)
    n = 100
    for i in range(n):
        root.visit(i,10,V)
    print(root.to_string(5))

