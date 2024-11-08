import numpy as np
from TurboCodes.pi import PI


class Turbo(PI):
    def __init__(self):
        """
        sets interleaver and puncturing
        """
        super().__init__()
        self.Rp = 1  # puncture rate
        self.m = 1  # bits per symbol

        self.modscheme = None
        self.puncturing = None
        self.piset = False
        self._Lp = None
        self._pW = None
        self._pH = None
        self._psize = None
        self._Lc = None
        self._Lc1 = None
        self._pund = None
        self._pisize = None

    def _interleaver_check(self):
        """

        """
        if not self.piset:
            raise Exception("please set an interleaver using the 'set_interleaver' method")

    def set_interleaver(self, type: str, interleaver_width: int, interleaver_height: int = None, **kwargs):
        """
        supported interleaver types:
            Random
            Block
            S-Random
            None
        """
        self.__initpi__[type]('turbo', width=interleaver_width, height=interleaver_height, **kwargs)
        self._pisize = self.__pisize__['turbo']
        self.piset = True

    def set_puncture(self, puncture_matrix):
        """
        """
        self.puncturing = np.array(puncture_matrix)
        self._pH, self._pW = np.shape(self.puncturing)
        self._psize = self.puncturing.size
        self.Rp = self._psize / np.count_nonzero(self.puncturing)

    def set_modulation(self, type):
        pass

    def merge(self, c1, c2):
        """
        """
        self._Lc = np.gcd(len(c1), len(c2))
        c1_ = np.reshape(c1, (self._Lc, -1))
        c2_ = np.reshape(c2, (self._Lc, -1))
        _, self._Lc1 = np.shape(c1_)
        c = np.hstack((c1_, c2_)).ravel()
        return c

    def demerge(self, y):
        """
        """
        y_ = np.reshape(y, (self._Lc, -1))
        c1 = y_[:, :self._Lc1].ravel()
        c2 = y_[:, self._Lc1:].ravel()
        return c1, c2

    def puncture(self, c):
        """
        """
        if self.puncturing is None:
            return c
        else:
            c_ = np.reshape(c, (-1, self._pH))
            self._Lp = len(c_)
            cp = np.zeros(0, dtype=int)
            self._pund = []
            for i, cp_ in enumerate(c_):
                for j, p in enumerate(self.puncturing[:, i % self._pW]):
                    if p != 0:
                        cp = np.append(cp, cp_[j])
                        self._pund.append((i, j))
            return cp

    def depuncture(self, y):
        """
        """
        if self.puncturing is None:
            return y
        else:
            yd = np.zeros((self._Lp, self._pH))
            for index, y_ in enumerate(y):
                (i, j) = self._pund[index]
                yd[i, j] = y_
            yd = yd.ravel()
            return yd


#####################################################
if __name__ == "__main__":
    T = Turbo()
    c1 = [0, 0, 1, 0, 0, 1]
    c2 = [0, 0, 2, 0, 0, 2]
    c = T._merge(c1, c2)
    print(c)
    T.set_puncture([[1, 1, 1], [0, 0, 1]])
    x = T._puncture(c)
    print(x)
    print(T._depuncture(x))
