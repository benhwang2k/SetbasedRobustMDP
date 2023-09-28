from problem import FintieProblem as Problem
import matplotlib.pyplot as plt

"""
2 states, 2 controls that encourage transition to the corresponding state, 2 costs that  drive to the corresponding state.

The start of the problem
"""
n = 2
m = 2
alpha = 0.6


# Create the transition probability function p
def p(y, x, u):
    if u == y:
        return 3 / 4
    else:
        return 1 / 4


# create the cost functions
def g0(x, u):
    return abs(x-0) + abs(x-u)


def g1(x, u):
    return abs(x-1) + abs(x-u)


def g2(x, u):
    return abs(x-2) + abs(x-u)




G = [g0, g1]

prob = Problem(n, m, G, alpha, p)
V_0 = tuple([0] * n)
colors = ["red", "orange", "yellow", "green", "blue", "purple"]
for pow2 in range(5):
    V_set = prob.set_Bellman(V_0, 2**pow2)
    print(f"iteration {2**pow2}: {V_set}")
    x = [point[0] for point in V_set]
    y = [point[1] for point in V_set]
    plt.scatter(x, y, color=colors[pow2])

# plot a single cost version
V_1 = V_0
for i in range(2**4):
    V_0 = prob.Bellman(V_0, g1)
    V_1 = prob.Bellman(V_1, g0)
    plt.plot(V_0[0], V_0[1], "co")
    plt.plot(V_1[0], V_1[1], "mo")


plt.show()