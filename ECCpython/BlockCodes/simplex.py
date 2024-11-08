from BlockCodes.hamming import Hamming


class Simplex(Hamming):
    def __init__(self, r):
        super().__init__(r)
        self.k, self.n = self.n, self.k
        self.G, self.H = self.H, self.G
        self.type = 'Simplex'