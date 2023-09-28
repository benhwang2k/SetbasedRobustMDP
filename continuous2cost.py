from problem_copy import ContinuousProblem
from problem import make_simple
from functools import partial
import math
import matplotlib.pyplot as plt

"""
turn a tuple into a function for comparison and evalutate at a point
"""


def eval_simple(V, h, x):
    f = make_simple(V, h)
    return f(x)


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

alpha = 0.6

h = 0.5
hlist = []
count = 0
V_final = []
while h > 0.05:
    hlist.append(h)
    prob = ContinuousProblem(h, G, alpha, p)
    V_0 = set([tuple([0] * prob.n)])
    colors = ["red", "orange", "yellow", "green", "blue", "purple"]
    print(f"states: {prob.states}")
    hoz = 8
    V_set = V_0
    for it in range(hoz):
        V_set = prob.set_Bellman(V_set, 1)
        # print(f"iteration {2 ** pow2}: {V_set}")
        x = [eval_simple(point, h, 0.25) for point in V_set]
        y = [eval_simple(point, h, 0.75) for point in V_set]
        plt.scatter(x, y, color=colors[it % 6])
        print(f" {round(math.exp(it) / math.exp(hoz) * 100)} % complete", end='\r')
    V_final.append(V_set)
    count += 1
    h = h - 0.01

# measure the max difference between the set and it's previous using the sup norm
sup = [0] * (count - 1)
for i in range(count - 1):
    print(f"processing {i} out of {count - 2}")
    for v_prev in V_final[i]:
        for v in V_final[i + 1]:
            v_prev_simp = make_simple(v_prev, hlist[i])
            v_simp = make_simple(v, hlist[i + 1])
            for x_ind in range(20):
                if abs(v_simp(0.05 * x_ind) - v_prev_simp(0.05 * x_ind)) > sup[i]:
                    sup[i] = abs(v_simp(0.05 * x_ind) - v_prev_simp(0.05 * x_ind))
print(sup)
plt.title("Comparison of cost-to-go from states 0.25 and 0.75 ")
# plot the y = x line
plt.plot([0, 2], [0, 2], 'c')
plt.ylabel("cost-to-go state: 0.75")
plt.xlabel("cost-to-go state: 0.25")

plt.figure()
plt.plot(hlist[:44], sup)
plt.title("Convergence with decreasing grid spacing")
plt.xlabel("h")
plt.ylabel("set distance between sequential trials")
plt.show()
