import numpy as np
import random


class PI:
    def __init__(self):
        """
        Implemented Interleavers:
            s-random :
            random :
            block :
            none : used when an interleaver that does nothing is required
        """
        self.interleaver = {}
        self.__pisize__ = {}
        self.__initpi__ = {'s-random': self.sInterleaver,
                           'random': self.rInterleaver,
                           'block': self.bInterleaver,
                           'none': self.nInterleaver}

    def set_interleaver(self, interleaver_type, *args, **kwargs):
        """

        """
        return self.__initpi__[interleaver_type](*args, **kwargs)

    def rInterleaver(self, flag, width: int, seed=None, height: int = None):
        """
        Random Interleaver
        """
        if seed is not None:
            random.seed(seed)

        indices = list(range(width))
        random.shuffle(indices)
        self.interleaver[flag] = list(indices)
        self.__pisize__[flag] = width

    def sInterleaver(self, flag, width: int, S_value: int, seed=None, height: int = None):
        """
        S Random Interleaver
        S_value <= sqrt(2N) for convergence in reasonable time
        http://www.josephboutros.org/ldpc_vs_turbo/turbo_Popovski_CL04.pdf
        """
        if seed is not None:
            random.seed(seed)

        choices = list(range(width))
        indices = [random.choice(choices)]
        choices.remove(indices[-1])

        while len(choices) > 0:
            k = random.choice(choices)  # candidate
            for i, j in enumerate(indices):
                if np.abs(j - k) > S_value or (len(indices) - indices[i]) > S_value:
                    indices = np.append(indices, k)
                    choices.remove(indices[-1])  # successful choice
                    break

        self.interleaver[flag] = list(indices)
        self.__pisize__[flag] = width

    def fInterleaver(self):
        """
        http://www.josephboutros.org/ldpc_vs_turbo/turbo_Popovski_CL04.pdf
        """
        pass

    def bInterleaver(self, flag, width: int, height: int):
        """
        block interleaver
        """
        self.__pisize__[flag] = width * height
        indices = list(range(self.__pisize__[flag]))
        indices = (np.reshape(indices, (height, width)).transpose()).flatten()
        self.interleaver[flag] = list(indices)

    def nInterleaver(self, flag, width: int, height: int = None):
        """
        None interleaver
        """
        self.interleaver[flag] = list(range(width))
        self.__pisize__[flag] = width

    def interleave(self, flag, data):
        if len(data) != self.__pisize__[flag]:
            raise Exception(f"length of input is {len(data)} but interleaver has length {self.__pisize__[flag]}")
        try:
            interleaved = np.empty_like(data)
            for i, index in enumerate(self.interleaver[flag]):
                interleaved[i] = data[index]
            return interleaved
        except KeyError:
            raise Exception("no such interleaver was initialised")

    def deinterleave(self, flag, data):
        if len(data) != self.__pisize__[flag]:
            raise Exception(f"length of input is {len(data)} but interleaver has length {self.__pisize__[flag]}")
        try:
            deinterleaved = np.empty_like(data)
            for i, index in enumerate(self.interleaver[flag]):
                deinterleaved[index] = data[i]
            return deinterleaved
        except KeyError:
            raise Exception("no such interleaver was initialised")


#############################################################################################################
if __name__ == "__main__":
    c = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    PP = PI()
    PP.bInterleaver(0, 9)
    print(PP.interleaver[0])
