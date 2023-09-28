from problem import ContinuousProblem
from functools import partial
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
    return abs(x - u) + abs(x - z)

for z in [0.25, 0.75]:
    G.append(partial(g, z))

alpha = 0.9

h = 0.5

prob = ContinuousProblem(h, G, alpha, p)
V_0 = set([tuple([0] * prob.n)])
colors = ["red", "orange", "yellow", "green", "blue", "purple"]
print(f"states: {prob.states}")
hoz2 = 4
V_set = V_0
for pow2 in range(hoz2):
    V_set = prob.set_Bellman(V_set, 2 ** pow2)
    # print(f"iteration {2 ** pow2}: {V_set}")
    x = [point[0] for point in V_set]
    y = [point[1] for point in V_set]
    plt.scatter(x, y, color=colors[pow2])

# plot a single cost version
V_0 = tuple([0] * prob.n)
V_1 = V_0
for i in range(2 ** (hoz2)):
    V_1 = prob.Bellman(V_1, prob.G[0])
    plt.plot(V_1[0], V_1[1], "mo")

# plot the y = x line
plt.plot([0, 5], [0, 5], 'c')

plt.show()
