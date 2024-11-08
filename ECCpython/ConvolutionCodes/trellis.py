import numpy as np
from Tools.helperFuncs import bi2de
from Tools.helperFuncs import de2bi


class Trellis:
    def __init__(self, G, r):
        """
        """
        self.G = np.array(G)
        self.n, self.L_c = np.shape(G)
        self.k = 1
        self.M = self.L_c - 1
        self.systematic = False
        self.rsc_poly = np.append(1, np.zeros(self.M, dtype=int))

        if r != 0:
            self.rsc_poly = self.G[r - 1, :]
            self.G = np.delete(self.G, r - 1, axis=0)
            self.systematic = True

        self.states = np.arange(2 ** self.M)
        self.bi_states = de2bi(self.states, self.M)

        self._init_trellis()
        self._init_states()

    def __str__(self) -> str:
        return f''

    def __repr__(self) -> str:
        return f''

    def _init_trellis(self):
        next = []
        out = []
        for state in self.bi_states:
            for u in range(2):
                s = np.append(u, state)
                s[0] = self.rsc_poly @ s % 2
                next = np.append(next, s[:-1])
                if self.systematic:
                    out = np.append(out, np.append(u, (self.G @ s) % 2))
                else:
                    out = np.append(out, (self.G @ s) % 2)
        next = np.reshape(next, (-1, 2, self.M)).astype(int)
        out = np.reshape(out, (-1, 2, self.n)).astype(int)
        de_next = np.array(list(map(bi2de, next)))
        de_out = np.array(list(map(bi2de, out)))
        self.trellis = {'next': next, 'out': out}
        self.de_trellis = {'next': de_next, 'out': de_out}

    def _init_states(self):
        # dictionary of next states for each state
        self.state_nexts = {s: [self.de_trellis['next'][s, 0], self.de_trellis['next'][s, 1]] for s in self.states}
        # dictionary of previous states for each state
        self.state_prevs = {s: [s_ for s_, nexts in self.state_nexts.items() if s in nexts] for s in self.states}
        # dictionary of input states for each state pair
        self.state_ins = {(s1, s2): u for s1 in self.states for u, s2 in enumerate(self.state_nexts[s1])}
        # dictionary of output states for each state pair
        self.state_outs = {(s1, s2): self.trellis['out'][s1, u] for s1 in self.states for u, s2 in
                           enumerate(self.state_nexts[s1])}


########################################################################################################
if __name__ == "__main__":
    generator_poly = [[1, 1, 1], [1, 0, 1]]
    rsc_poly = 0
    T = Trellis(generator_poly, rsc_poly)
    print(T.de_trellis['next'])
