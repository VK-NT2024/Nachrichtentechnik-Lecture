import numpy as np
from HelperFuncs import bi2de


class BlockCode:
    def __init__(self,
                 n: int,
                 k: int,
                 generator,
                 parity_check,
                 code_type: str = None,
                 free_distance: int = None):
        """
        class for Block Codes. Generates the code based on the generator and parity check matrices

        n:
        k:
        generator:
        parity_check:
        code_type:
        free_dsitance:
        """

        self.n, self.k = n, k
        self.G, self.H = generator, parity_check
        self.P = self.G[:, k:]
        self.r = n - k
        self.Rc = k / n

        # obtained from child classes
        if free_distance is None:
            self.d = self.n - self.k + 1  # singleton bound
        else:
            self.d = free_distance  # free distance

        self.t = (self.d - 1) // 2

        if code_type is None:
            self.type = 'Block Code'

        self.flatten = False
        self._encode = {
            True: self.encode_flat,
            False: self.encode_array
        }

        self._decode = {
            True: lambda c: self.syndrome_decode(c).ravel(),
            False: lambda c: self.syndrome_decode(c)
        }

    def __str__(self):
        return f'Code-Type: {self.type}\nCode-Bits(n): {self.n}\nInfo-Bits(k): {self.k}'

    def __repr__(self):
        return f'Generator matrix: {self.G}\nCheck Matrix: {self.H}\nCode-Type: {self.type}'

    @property
    def cosets(self):
        """
        generates cosets for syndrome decoding

        return
        ------
            coset_leaders : dict
                the leaders of each coset have the lowest hamming distance and are a result of the same syndrome
        """
        from itertools import combinations
        coset_leaders = {}
        for combination in combinations(range(self.n), self.t):
            coset_lead = [1 if i in combination else 0 for i in range(self.n)]
            syndrome = bi2de(coset_lead @ np.transpose(self.H) % 2)[0]
            coset_leaders[syndrome] = coset_lead
        coset_leaders[0] = 0
        return coset_leaders

    def encode(self, u):
        """
        input:
            u:  list or 2D array
                information bits

        returns:
            c: 2D array
                coded bits

            flatten = True
                c: list
                    coded bits
        """
        return self._encode[self.flatten](u)

    def decode(self, c):
        """
        performs encoding for all block codes
        """
        return self._decode[self.flatten](c)

    def encode_flat(self, u):
        """
        performs encoding for all block codes
        """
        try:
            u = np.reshape(u, (-1, self.k))
            c = (u @ self.G % 2).ravel()
        except Exception:
            raise ValueError(f"length of u needs to be a multiple of {self.k}")

        return c

    def encode_array(self, u):
        """
         input:
             u: matrix of information bits (one information word with k bits per row, 2nd dimension)
         """
        try:
            c = np.array(u) @ self.G % 2
        except Exception:
            raise ValueError(
                f"number of columns of u needs to be a multiple of {self.k} but input has {np.shape(u)[0]} columns")

        return c

    def syndrome_decode(self, c_hat):
        """
        performs hard-in, hard-out decoding using syndromes for all block codes
        """
        N_codewords = c_hat.shape[0]
        u_hat = np.zeros((N_codewords, self.k), dtype=int)
        syndromes = bi2de(c_hat @ self.H.T % 2)
        for i in range(N_codewords):
            if syndromes[i] != 0:
                try:
                    error = self.cosets[syndromes[i]]
                except KeyError:
                    error = 0  # error detected but not corrected
            else:
                error = np.zeros(self.n, dtype=int)
            u_hat[i, :] = (c_hat[i, :] - error)[:self.k] % 2

        if self.flatten:
            u_hat = u_hat.ravel()

        return u_hat


################################################################################################
if __name__ == '__main__':
    print(bi2de([1, 1, 1]))
