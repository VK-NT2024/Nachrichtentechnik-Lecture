import numpy as np
import random


class Interleaving:
    def __init__(self,
                 interleaver_type: str,
                 shape: tuple,
                 seed=None,
                 S_value=None):
        """
        Implemented Interleavers:
            s-random :
            random :
            block :
            none : used when an interleaver that does nothing is required
        """
        self._call = 0
        self.type = interleaver_type

        try:
            self.height = shape[0]
        except IndexError:
            self.height = 1

        try:
            self.width = shape[1]
        except IndexError:
            self.width = 1

        self.size = self.width * self.height
        self.shape = self.width, self.height
        self.seed = seed
        self.Svalue = S_value

        self.__interleavers__ = {
            's-random': self._sInterleaver,
            'random': self._rInterleaver,
            'block': self._bInterleaver,
            'none': self._nInterleaver}

        self._interleaving = self.__interleavers__[self.type]()
        self._deinterleaving = np.argsort(self._interleaving)

    # def __call__(self, data):
    #     """
    #     allows calling object directly to perform interleaving and deinterleaving
    #     """
    #     if self._call % 2 == 0:
    #         out = self.interleave(data)
    #     else:
    #         out = self.deinterleave(data)
    #     self._call += 1
    #     return out

    def _rInterleaver(self):
        """
        Random Interleaver
        """
        if self.seed is not None:
            random.seed(self.seed)

        indices = list(range(self.size))
        random.shuffle(indices)
        return list(indices)

    def _sInterleaver(self):
        """
        S Random Interleaver
        S_value <= sqrt(2N) for convergence in reasonable time
        http://www.josephboutros.org/ldpc_vs_turbo/turbo_Popovski_CL04.pdf
        """
        if self.seed is not None:
            random.seed(self.seed)

        choices = list(range(self.size))
        indices = [random.choice(choices)]
        choices.remove(indices[-1])

        while len(choices) > 0:
            k = random.choice(choices)  # candidate
            for i, j in enumerate(indices):
                if np.abs(j - k) > self.Svalue or (len(indices) - indices[i]) > self.Svalue:
                    indices = np.append(indices, k)
                    choices.remove(indices[-1])  # successful choice
                    break

        return list(indices)

    def _fInterleaver(self):
        """
        http://www.josephboutros.org/ldpc_vs_turbo/turbo_Popovski_CL04.pdf
        """
        pass

    def _bInterleaver(self):
        """
        block interleaver
        """
        indices = list(range(self.size))
        indices = (np.reshape(indices, self.shape).transpose()).flatten()
        return list(indices)

    def _nInterleaver(self):
        """
        None interleaver
        """
        return list(range(self.size))

    def interleave(self, data):
        """
        performs interleave
        """
        shape = np.shape(data)
        data_in = np.reshape(data, (-1, self.size))
        data_out = np.zeros_like(data_in)
        for i, d_in in enumerate(data_in):
            data_out[i] = d_in[self._interleaving]
        data_out = np.reshape(data_out, shape)
        return data_out

    def deinterleave(self, data):
        """
        performs deinterleave
        """
        shape = np.shape(data)
        data_in = np.reshape(data, (-1, self.size))
        data_out = np.zeros_like(data_in)
        for i, d_in in enumerate(data_in):
            data_out[i] = d_in[self._deinterleaving]
        data_out = np.reshape(data_out, shape)
        return data_out


#############################################################################################################
if __name__ == "__main__":
    pi = Interleaving('block', (3, 3))
    c = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    # c = np.array(c).ravel()
    # d = pi.interleave(c)
    # print(d)
    # c_ = pi.deinterleave(d)
    # print(c_)
    d = pi(c)
    print(d)
    print(pi(d))