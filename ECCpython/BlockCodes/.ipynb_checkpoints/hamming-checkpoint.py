import numpy as np
from BlockCodes.blockcode import BlockCode
from HelperFuncs import de2bi
from HelperFuncs import bi2de


class Hamming(BlockCode):
    def __init__(self,
                 r: int = 3):
        """
        """
        self.r = r
        self.n = 2 ** r - 1
        self.k = self.n - self.r

        self.type = 'Hamming'

        H = de2bi(np.arange(1, self.n + 1), self.r).T  # generate arbitrary parity check matrix
        # compute parity check matrix for systematic encoder
        # find columns with Hamming weight 1
        ptr = np.flatnonzero(np.sum(H, axis=0) - 1)
        P_T = H[:, ptr]

        self.H = np.hstack((P_T, np.eye(self.r)))  # generate parity check matrix
        self.G = np.hstack((np.eye(self.k), P_T.T))  # generate generator matrix

        super().__init__(self.n, self.k, self.G, self.H, self.type, 3)

    def message_passing(self, Lin, depth, algorithm='approximation'):
        """
        """
        alg = {'approximation': lambda L: np.prod(np.sign(L)) * np.min(np.abs(L)),
               'exact': lambda L: 2 * np.arctan(np.prod(np.tanh(L / 2)))}

        Lc = np.multiply(self.H, Lin)
        Le = np.zeros((self.r, self.n))
        Snode = {s: [v for v, h in enumerate(self.H[s, :]) if h == 1] for s in range(self.r)}
        Vnode = {v: [s for s, h in enumerate(self.H[:, v]) if h == 1] for v in range(self.n)}

        for _ in range(depth):
            for s in Snode:
                for v in Snode[s]:
                    input_to_Snode = np.array([Lc[s, i] for i in Snode[s] if i != v])
                    Le[s, v] = alg[algorithm](input_to_Snode)
            for v in Vnode:
                for s in Vnode[v]:
                    input_to_Vnode = np.array([Le[j, v] for j in Vnode[v] if j != s])
                    Lc[s, v] = Lin[v] + np.sum(input_to_Vnode)

        Lc = (Lin + np.sum(Le, axis=0))

        return Lc

    def make_trellis(self, dual=True, p=None):
        """
            generate trellis from parity check matrix self.H
            self.H must be binary matrix with n-k rows and n columns

            INPUTS
                p          -   integer indicating the column to be neglected by generating the trellis
                                 necessary for decoding with extended Battail algorithm
                                 if p == n, no column is omitted

            Outputs
                trellis        array with 2^(n-k) rows and n columns (n-1 columns for p > 0)
                                row i represents current state i,
                                contents of row i and column j describes next state at time j + 1 for info=1
                                (info = 0 remains in the same state)
        """
        self.dual = dual

        if self.dual:
            H = self.G
            # number of parity bits of dual code
            m = self.k
        else:
            H = self.H
            # number of parity bits
            m = self.n - self.k

        # number of states in trellis
        M = 2 ** m
        n = self.n

        if p is None:
            p = self.n

        # vector containing states in decimal representation
        state = np.arange(M)
        # matrix containing all states in binary representation (each row contains one state)
        state_bin = de2bi(state, m, leftMSB=False)

        if p == self.n:
            trellis = np.zeros((M, self.n), dtype=int)
        else:
            trellis = np.zeros((M, self.n - 1), dtype=int)
            H = np.delete(H, p, axis=1)  # discard p-th column
            n -= 1

        for i in np.arange(n):
            H_i = np.tile(H[:, i], (M, 1))
            trellis[:, i] = bi2de(np.mod(state_bin + H_i, 2), leftMSB=False)

        return trellis

    def bcjr_decode(self, Lin, La=None, info=False):
        """
             soft-output decoding of linear block codes using BCJR algorithm (E.Offer)

            INPUTS

            Lin         -   vector of received data, intrinsic LLRs
            La          -   vector containing a priori LLRs for each information bit
            dual        -   True: indicates decoding with trellis of dual code,
                                        (H must be the generator matrix of the original code,
                                         that is the parity check matrix for the dual code)
                            False: decoding with original code
            info        -   True:  soft-output is calculated only for information bits 0 ... k-1
                            False: soft-output is calculated for all coded bits 0 ... n-1

            OUTPUTS
            
            L           -   vector containing LLRs for each information or code bit
            Le          -   vector containing extrinsic LLRs for each information or code bit

        """
        self.trellis = self.make_trellis(p=self.n)

        if info:
            n_out = self.k  # soft output onl for information bits --> n_out = k;
        else:
            n_out = self.n  # soft - output for all coded bits --> n_out = n;

        if np.any(La == None):
            La = np.zeros(self.k)
        n_La = len(La)

        # --------------------------------------- input LLRs ----------------------------------------------------
        Lapp = Lin
        Lapp[:n_La] += La

        # --------------------------------------  decoding via dual code ----------------------------------------
        if self.dual:
            # number of parity bits of dual code
            m = self.k
            M = 2 ** m

            # ------------------ initialize alpha ------------------------
            alpha = np.zeros((M, self.n))
            alpha[0, 0] = 1.0  # trellis starts in all-zero-state

            # ------------------ initialize beta ------------------------
            beta = np.zeros((M, self.n))
            beta[0, self.n - 1] = 1.0  # trellis ends in all-zero-state

            metric = np.tanh(Lapp / 2.0)

            # ------------------ forward recursion -----------------------
            for j in np.arange(self.n - 1):
                # alpha for info = 0
                alpha[:, j + 1] = alpha[:, j]
                # alpha for info = 1
                alpha[self.trellis[:, j], j + 1] += alpha[:, j] * metric[j]

            # ------------------ backward recursion -----------------------
            for j in np.arange(self.n - 1, 0, -1):
                # alpha for info = 0
                beta[:, j - 1] = beta[:, j]
                # alpha for info = 1
                beta[:, j - 1] += beta[self.trellis[:, j], j] * metric[j]

            # ------------------ calculating extrinsic information -----------------------
            # info = 0
            Le0 = np.sum(alpha[:, :n_out] * beta[:, :n_out], axis=0)
            # info = 1
            Le1 = np.zeros(n_out)
            for run in np.arange(n_out):
                Le1[run] = np.sum(alpha[:, run] * beta[self.trellis[:, run], run])

            Le = np.log((Le0 + Le1) / (Le0 - Le1))


        # --------------------------------------  decoding via original code -----------------------------------
        else:
            # number of parity bits
            m = self.n - self.k
            M = 2 ** m

            # ------------------ initialize alpha ------------------------
            alpha = np.zeros((M, self.n))
            alpha[0, 0] = 1.0  # trellis starts in all-zero-state

            # ------------------ initialize beta ------------------------
            beta = np.zeros((M, self.n))
            beta[0, self.n - 1] = 1.0  # trellis ends in all-zero-state

            metric = np.zeros((self.n, 2))
            # calculating transition probabilities for info = 1
            ptr = np.where(np.abs(Lapp) < 50.0)
            metric[ptr, 1] = 1.0 / (1.0 + np.exp(Lapp[ptr]))
            metric[Lapp < -50, 1] = 1.0
            # calculating transition probabilities for info = 0
            metric[ptr, 0] = metric[ptr, 1] * np.exp(Lapp[ptr])
            metric[Lapp > 50.0, 0] = 1.0

            # ------------------ forward recursion -----------------------
            norm = np.zeros(self.n)
            for j in np.arange(self.n - 1):
                # alpha for info = 0
                alpha[:, j + 1] = alpha[:, j] * metric[j, 0]
                # alpha for info = 1
                alpha[self.trellis[:, j], j + 1] += alpha[:, j] * metric[j, 1]
                norm[j] = np.sum(alpha[:, j + 1])
                alpha[:, j + 1] /= norm[j]

            # ------------------ backward recursion - ----------------------
            for j in np.arange(self.n - 1, 0, -1):
                # beta for info = 0
                beta[:, j - 1] = beta[:, j] * metric[j, 0]
                # beta for info = 1
                beta[:, j - 1] += beta[self.trellis[:, j], j] * metric[j, 1]
                beta[:, j - 1] /= norm[j - 1]

            # ------------------ calculating extrinsic information -----------------------
            # info = 0
            Le0 = np.sum(alpha[:, :n_out] * beta[:, :n_out], axis=0)
            # info = 1
            Le1 = np.zeros(n_out)
            for run in np.arange(n_out):
                Le1[run] = np.sum(alpha[:, run] * beta[self.trellis[:, run], run])

            Le = np.log(Le0 / Le1)

        # --------------------- Adding systematic, a priori and extrinsic information ------------------------
        L = Lapp[:n_out] + Le

        return L, Le

    def battail_decode(self, Lin, La=None, info=False):
        """
             soft-output decoding of linear block codes using algorithm of Battail (E.Offer)

            INPUTS

            Lin         -   vector of received data, intrinsic LLRs
            La          -   vector containing a priori LLRs for each information bit
            info        -   True:  soft-output is calculated only for information bits 0 ... k-1
                            False: soft-output is calculated for all coded bits 0 ... n-1

            OUTPUTS

            L           -   vector containing LLRs for each information or code bit
            Le          -   vector containing extrinsic LLRs for each information or code bit

        """
        self.trellis = list()
        for i in np.arange(self.n):  # calculating extrinsic information for each information bit
            self.trellis.append(self.make_trellis(p=i))  # generating trellis discarding i - th column of H
            if self.dual:
                self.H_dec = bi2de(self.G.transpose(), leftMSB=False)
            else:
                self.H_dec = bi2de(self.H.transpose(), leftMSB=False)

        if info:
            n_out = self.k  # soft output onl for information bits --> n_out = k;
        else:
            n_out = self.n  # soft - output for all coded bits --> n_out = n;

        if np.any(La == None):
            La = np.zeros(self.k)
        n_La = len(La)

        # ---------------------------- calculation of probabilities from input LLRs -------------------------------
        Lapp = Lin
        Lapp[:n_La] += La

        # --------------------------------------  decoding via dual code ---------------------------------------
        if self.dual:
            # number of parity bits of dual code
            m = self.k
            M = 2 ** m

            metric = np.tanh(Lapp / 2.0)

            for i in np.arange(n_out):  # calculating extrinsic information for each output bit
                gamma = np.delete(metric, i)

                # ------------------ initialize alpha ------------------------
                alpha = np.zeros((M, self.n - 1))
                alpha[0, 0] = 1.0  # trellis starts in all-zero-state

                for j in np.arange(self.n - 2):
                    # alpha for info=0
                    alpha[:, j + 1] = alpha[:, j]
                    # alpha for info=1
                    alpha[self.trellis[i][:, j], j + 1] += alpha[:, j] * gamma[j]

                Le[i] = 2.0 * np.arctanh(alpha[self.H_dec[i], self.n - 2] / alpha[0, self.n - 2])

        # --------------------------------------  decoding via original code -----------------------------------
        else:

            # number of parity bits
            m = self.n - self.k
            M = 2 ** m

            # ------------------ initialize alpha ------------------------
            alpha = np.zeros((M, self.n))
            alpha[0, 0] = 1.0  # trellis starts in all-zero-state

            # ------------------ initialize beta ------------------------
            beta = np.zeros((M, self.n))
            beta[0, self.n - 1] = 1.0  # trellis ends in all-zero-state

            metric = np.zeros((self.n, 2))
            # calculating transition probabilities for info = 1
            ptr = np.where(np.abs(Lapp) < 70.0)
            metric[ptr, 1] = 1.0 / (1.0 + np.exp(Lapp[ptr]))
            metric[Lapp < -70, 1] = 1.0
            # calculating transition probabilities for info = 0
            metric[ptr, 0] = metric[ptr, 1] * np.exp(Lapp[ptr])
            metric[Lapp > 70.0, 0] = 1.0

            # calculating extrinsic LLRs for each information bit
            for i in np.arange(n_out):
                gamma = np.delete(metric, i, axis=0)
                alpha = np.zeros((M, self.n - 1))
                # trellis starts in all - zero - state
                alpha[0, 0] = 1.0

                for j in np.arange(self.n - 2):
                    # alpha for info = 0
                    alpha[:, j + 1] = alpha[:, j] * gamma[j, 0]
                    # alpha for info = 1
                    alpha[self.trellis[i][:, j], j + 1] += alpha[:, j] * gamma[j, 1]
                    norm = np.sum(alpha[:, j + 1])
                    alpha[:, j + 1] /= norm

                Le[i] = np.log(alpha[self.H_dec[i], self.n - 2]) - np.log(alpha[0, self.n - 2])

        # -------------------------------------- adding systematic and a priori information ---------------------
        L = Lapp[:n_out] + Le

        return L, Le


#######################################################################################################################
if __name__ == "__main__":
    H = Hamming()
    print(H.cosets)
