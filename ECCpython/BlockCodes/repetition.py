import numpy as np
from BlockCodes.blockcode import BlockCode


class Repetition(BlockCode):
    def __init__(self, n):
        """
        Repetition code class

        Parameters
        -----------
        n: int
        number of repetitions of the information bit
        """
        self.type = 'Repetition'
        self.P = np.ones(n - 1, dtype=int)
        self.G = np.array([np.ones(n, dtype=int)]).astype(int)
        self.H = np.column_stack((self.P, np.eye(n - 1))).astype(int)
        self.k = 1
        super().__init__(n, 1, self.G, self.H, code_type=self.type, free_distance=n)

    def softdecode(self, Lin, La=None, algorithm=None):
        """
        """
        Lin = np.reshape(Lin, (-1, self.n))

        if La is None:
            Lu = np.sum(Lin, axis=1, keepdims=True)
        else:
            Lu = np.sum(Lin, axis=1, keepdims=True) + La

        Lc = np.tile(Lu, (1, self.n))

        if self.flatten:
            Lu = Lu.ravel()
            Lc = Lc.ravel()

        return Lu, Lc


############################################################################################
if __name__ == "__main__":
    R = Repetition(1)
    u = np.array([1, 0, 1])
    c = R.encode(u)
    print(c)
