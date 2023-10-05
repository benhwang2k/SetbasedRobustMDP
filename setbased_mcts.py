import time
import mcts
import math
import random
import mdp
import matplotlib.pyplot as plt


class mdpTree:
    def __init__(self, mdp, parent):
        self.mdp = mdp
        self.parent = parent
        self.children = []

    def add_child(self, mdp):
        child = mdpTree(mdp, self)
        self.children.append(child)
        return child

    def getleafseqence(self):
        parent = self
        seq = []
        while not (parent.mdp is None):
            seq.insert(0, parent.mdp)
            parent = parent.parent
        return seq

    def getsequences(self):
        if not self.children:
            return [self.getleafseqence()]
        else:
            l = []
            for child in self.children:
                l += child.getsequences()
            return l

    def get_repr(self):
        if self.children == []:
            return [self.mdp]
        else:
            s = []
            for child in self.children:
                arr = child.get_repr()
                if not (arr == []):
                    s.insert(0, arr.pop(0))
                    if arr:
                        s.append(arr)

            r = [self.mdp] + [s]
            return r

    def __repr__(self):
        return str(self.get_repr())


def create_mdp_tree(root: mdpTree, mdpset, horizon):
    if horizon == 1:
        for mdp in mdpset:
            root.add_child(mdp)
        return root
    else:
        for mdp in mdpset:
            child = root.add_child(mdp)
            create_mdp_tree(child, mdpset, horizon - 1)
        return root


class SetBasedMCTS:
    def __init__(self, S, U, G, alpha, P, H, Vguess=lambda state: 0):
        self.S = S
        self.U = U
        self.alpha = alpha
        self.P = P
        self.mdpset = []
        self.H = H
        count = 0
        for g in G:
            m = mdp.mdp(set(S), set(U), g, alpha, P)
            m.setName(count)
            self.mdpset.append(m)
            count += 1
        self.defaultvalue = Vguess
        root = mdpTree(None, None)
        tree = create_mdp_tree(root, self.mdpset, H)
        self.sequences = tree.getsequences()

    def rollout_seq_policy(self, mdpseq, pis):
        key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
        V = {}

        for s in self.S:
            V[s] = 0
        for s0 in self.S:
            dist = {}
            for state in self.S:
                dist[state] = 1 if state == s0 else 0
            for h in range(self.H):
                k = key[h:self.H] + '0' * h
                V[s0] += (self.alpha ** h) * sum([(dist[s] * mdpseq[h].C(s, pis[k][s])) for s in self.S])
                next_dist = {}
                for state in self.S:
                    next_dist[state] = sum([dist[prev] * self.P(state, prev, pis[k][prev]) for prev in self.S])
                dist = next_dist
        return V

    def rollout_seq_policy_horizon(self, mdpseq, pis):
        key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
        V = {}

        for s in self.S:
            V[s] = 0
        for s0 in self.S:
            dist = {}
            for state in self.S:
                dist[state] = 1 if state == s0 else 0
            for h in range(self.H):
                k = key[h:self.H]
                V[s0] += (self.alpha ** h) * sum([(dist[s] * mdpseq[h].C(s, pis[self.H-h][k][s])) for s in self.S])
                next_dist = {}
                for state in self.S:
                    next_dist[state] = sum([dist[prev] * self.P(state, prev, pis[k][prev]) for prev in self.S])
                dist = next_dist
        return V

    def bellman_iteration(self, proc: mdp.mdp, V):
        pi = {}
        nextV = {}
        for s in proc.S:
            minQ = math.inf
            mina = list(proc.U)[0]
            for a in proc.U:

                Q = proc.C(s, a) + proc.alpha * (sum([proc.P(s_next, s, a)*V[s_next] for s_next in proc.S]))
                # if s == list(proc.S)[0]:
                #     print(f"a = {a}, Q = {Q}")
                #     print(f"stagecost = {proc.C(s, a)}")
                #     print(f"alpha time exp = {proc.alpha * sum([proc.P(s_next, s, a) * V[s_next] for s_next in proc.S])}")
                if Q < minQ:
                    minQ = Q
                    mina = a
            pi[s] = mina
            nextV[s] = minQ
        return nextV, pi
    def solveBellman(self, seqkey):
        seq = []
        for t in range(len(seqkey)):
            # print(f"key : {seqkey[t]} , mdpname = {self.mdpset[int(seqkey[t])].name}")
            seq.append(self.mdpset[int(seqkey[t])])
        V = {}
        for s in self.S:
            V[s] = 0
        V_iter = V
        for i in range(self.H):
            (V_iter, pi_iter) = self.bellman_iteration(seq[self.H-1-i], V_iter)
            # print("----------------------------------")
            # print(f"bellman h = {self.H-1-i}, mdp = {seq[self.H-1-i]}")
            # print(pi_iter)
            # print("----------------------------------")
        return (V_iter, pi_iter)

    def solve(self, budget):
        tic = time.time()
        expected_time = len(self.mdpset[0].S) * budget * len(self.sequences)
        print(f"expected duration: {expected_time} = {len(self.mdpset[0].S)} x {budget} x {len(self.sequences)}")

        policies = {}
        values = {}
        # print(f"H = {self.H}, length of seq = {len(self.sequences[0])}")
        for mdpseq in self.sequences:
            # print(f"policy for mdp {mdpseq}")
            key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
            policy = mcts.get_policy(mdpseq, budget, self.H, defaultvalue)
            policies[key] = policy
        for mdpseq in self.sequences:
            #print(f"value for mdp {mdpseq}")
            key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
            values[key] = self.rollout_seq_policy(mdpseq, policies)
        # print(f"elapsed time: {time.time() - tic}")
        return values, policies

def get_policy(prob: mdp.mdp, V):
    pi = {}
    for s in prob.S:
        minQ = math.inf
        mina = list(prob.U)[0]
        for a in prob.U:
            Q = prob.C(s, a) + prob.alpha * sum([prob.P(s_next, s, a)*V[s_next] for s_next in prob.S])
            print(f"current state: {s} action choice = {a} , Q = {Q}")
            if Q < minQ:
                minQ = Q
                mina = a
        pi[s] = mina
        input()
        print("---------")
    return pi
H = 5
alpha = 0.7
gz = lambda z: (lambda state, action: 1. * abs(state - z) + 0.0005 * abs(state-action))
target_states = [0.2, 0.8]
G = []
for target in target_states:
    G += [gz(target)]
defaultvalue = lambda state: 0
def p(y, w, u):
    if y < u:
        return (y / u) * 2.0
    else:
        return 2.0 * (1 - y) / (1 - u)


def p_disc(y, x, u):
    y_disc = S[math.floor(y / h)]
    x_disc = S[math.floor(x / h)]
    return h * p(y_disc, x_disc, u) / sum([h * p(m, x_disc, u) for m in S])

def p_deterministic(y, x, u):
    closest = S[0]
    mindist = math.inf
    for s in S:
        if abs(u-s) < mindist:
            mindist = abs(u-s)
            closest = s
    return 1 if y == closest else 0
h = 0.5
valmap = []
errmap = []
d = 2
while d < 30:
    d += 1
    h = 1.0/d
    S = []
    U = []
    x = h
    while x < 1.00000001:
        point = round(x - h/2, 3)
        S.append(point) #random.random() * h)
        U.append(point)
        x += h

    # print(f"parameters: \n S: {S} \n U: {U} \n G: {G} \n H: {H} \n def: {defaultvalue} \n")
    prob = SetBasedMCTS(S, U, G, alpha, p_disc, H, defaultvalue)
    (values, policies) = prob.solve(budget=10)
    valmap.append(values)
    if len(valmap) > 1:
        # calculate the error
        plotkeys = ['0'*H, '1'*H]
        errdict = {}
        if H > 1:
            oskey1 = '01'*int(H/2) if (H % 2 == 0) else '01'*int(H/2) + '0'
            oskey2 = '10' * int(H / 2) if (H % 2 == 0) else '10' * int((H-1) / 2) + '1'
            plotkeys.append(oskey1)
            plotkeys.append(oskey2)
        for key in plotkeys:
            e = 0
            for s in values[key]:
                nearest = math.inf
                state_error = 0
                for sclose in valmap[len(valmap)-2][key]:
                    if abs(sclose - s) < abs(nearest - s):
                        nearest = sclose
                        state_error = abs(values[key][s] - valmap[len(valmap)-2][key][nearest])
                if state_error > e:
                    e = state_error
            # e is the infinity norm of the difference between successive iterations
            errdict[key] = e
        errmap.append(errdict)

count = 2
for m in errmap:
    for key in m:
        plt.plot(count, m[key], marker='o')
    count += 1
plt.savefig("gridspaceerror.png")



# # print(policies)
# plotkeys = ['0'*H, '1'*H]
# if H > 1:
#     oskey1 = '01'*int(H/2) if (H % 2 == 0) else '01'*int(H/2) + '0'
#     oskey2 = '10' * int(H / 2) if (H % 2 == 0) else '10' * int((H-1) / 2) + '1'
#     plotkeys.append(oskey1)
#     plotkeys.append(oskey2)
# sm = 0
# for key in plotkeys:
#     #plt.title(f"tree and bellman(o),key {key}")
#     #plt.plot(S, [values[key][s] for s in S])
#     #print(f"tree policy: {policies[key]}")
#     # print(f"tree policy at h = 0: {policies[key]}")
#     (V, pi) = prob.solveBellman(key)
#     e = sum([abs(valmap[h[s] - values[key][s]) for s in S])
#     sm += e
# plt.plot(H, sm, marker='o')
#     #print(f"final result: {V}")
# plt.show()


#
# for i in range(2):
#     H = (3+1) * 2
#     # print(S)
#     problem = SetBasedMCTS(S, U, G, alpha, p_disc, H, defaultvalue)
#     (values, policies) = problem.solve(budget=0.1)
#     # print(f"values \n {values}")
#     # print(f"policies \n {policies}")
#     key = '10'*int(H/2)
#     plt.figure(0)
#     plt.title("tree and bellman(o)")
#     plt.plot(S, [values[key][s] for s in S])
#     #print(f"tree policy: {policies[key]}")
#     # print(f"tree policy at h = 0: {policies[key]}")
#     (V, pi) = problem.solveBellman(key)
#     #print(f"final result: {V}")
#     # plt.figure(1)
#     # plt.title("bellman")
#     plt.plot(S, [V[s] for s in S], marker="o")
#     err.append(sum([abs(V[s] - values[key][s]) for s in S]))
#     for h in range(H):
#         k = key[h:H] + '0' * h
#         print(f"h = {h}, key = {k} tree policy :\n {policies[k]}")
# plt.figure(2)
# plt.title("errorh4")
# plt.plot(err)
# plt.savefig("errorh4.png")
# plt.figure(0)
# plt.savefig("valuesh4.png")


