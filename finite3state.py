from problem import FintieProblem as Problem

"""
3 states, 3 controls that encourage transition to the corresponding state, 3 costs that  drive to the corresponding state.

The start of the problem
"""
n = 3
m = 3
alpha = 0.9


# Create the transition probability function p
def p(y, x, u):
    if u == y:
        return 1 / 2
    else:
        return 1 / 4


def p2(y, x, u):
    if u == y:
        return 2 / 3
    else:
        return 1 / 6


# create the cost functions
def g0(x, u):
    return abs(x-0) + abs(x-u)


def g1(x, u):
    return abs(x-1) + abs(x-u)


def g2(x, u):
    return abs(x-2) + abs(x-u)




G = [g0, g1, g2]

prob = Problem(n, m, G, alpha, p)
V_0 = tuple([0] * n)
V_set = prob.set_Bellman(V_0, 2)
print(V_set)