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

    def solveBellman(self, seqkey):
        seq = []
        for t in range(len(seqkey)):
            print(f"key : {seqkey[t]} , mdpname = {self.mdpset[int(seqkey[t])].name}")
            seq.append(self.mdpset[int(seqkey[t])])
        V = {}
        for s in self.S:
            V[s] = 0
        V_next = V
        for h in range(self.H):
            for s in self.S:
                val = math.inf
                minu = list(self.U)[0]
                for u in self.U:
                    exp = seq[self.H - 1 - h].C(s, u) + sum([self.P(y, s, u) * V[y] for y in self.S])
                    if exp < val:
                        val = exp
                        minu = u
                #print(f"state: {s} time: {H - h} policy {round(minu, 3)} ")
                #print(self.alpha ** (self.H - h))
                V_next[s] = (self.alpha ** (self.H - h - 1)) * val
                #print(V_next)
            V = V_next
        return V

    def solve(self, budget):
        tic = time.time()
        expected_time = len(self.mdpset[0].S) * budget * len(self.sequences)
        print(f"expected duration: {expected_time} = {len(self.mdpset[0].S)} x {budget} x {len(self.sequences)}")

        policies = {}
        values = {}
        # print(f"H = {self.H}, length of seq = {len(self.sequences[0])}")
        for mdpseq in self.sequences:
            print(f"policy for mdp {mdpseq}")
            key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
            policy = mcts.get_policy(mdpseq, budget, self.H, defaultvalue)
            policies[key] = policy
        for mdpseq in self.sequences:
            #print(f"value for mdp {mdpseq}")
            key = ''.join([format(mdp.name, 'd') for mdp in mdpseq])
            values[key] = self.rollout_seq_policy(mdpseq, policies)
        print(f"elapsed time: {time.time() - tic}")
        return values, policies


h = 0.19
S = []
U = []
x = h
while x < 1:
    S.append(x - random.random() * h)
    U.append(x - h)
    x += h

z = 0.5
gz = lambda z: (lambda state, action: abs(state - z))

alpha = 0.1


def p(y, w, u):
    if y < u:
        return (y / u) * 2.0
    else:
        return 2.0 * (1 - y) / (1 - u)


def p_disc(y, x, u):
    y_disc = S[math.floor(y / h)]
    x_disc = S[math.floor(x / h)]
    return h * p(y_disc, x_disc, u) / sum([h * p(m, x_disc, u) for m in S])


target_states = [0.3, 0.6]
G = []
for target in target_states:
    G += [gz(target)]
defaultvalue = lambda state: 0

H = 4
# print(S)
problem = SetBasedMCTS(S, U, G, alpha, p_disc, H, defaultvalue)
(values, policies) = problem.solve(budget=0.2)
# print(f"values \n {values}")
print(f"policies \n {policies}")
key = '10'*int(H/2)
plt.plot(S, [values[key][s] for s in S])

V = problem.solveBellman(key)
print(f"V[{S[4]}] = {V[S[4]]}")
plt.plot(S, [V[s] for s in S])
plt.savefig("figure.png")


# class setbasedValue:
#     def __init__(self, mdpset, initV, H):
