import numpy as np
import galois


class Alist:
    """
    example:
        [[12, 16],
        [4, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 8, 10, 13],
        [4, 7, 9, 13],
        [2, 5, 7, 10],
        [4, 6, 11, 14],
        [3, 9, 15, 16],
        [1, 6, 9, 10],
        [4, 8, 12, 15],
        [2, 6, 12, 16],
        [1, 7, 14, 16],
        [3, 5, 12, 14],
        [2, 11, 13, 15],
        [1, 5, 8, 11],
        [6, 9, 12],
        [3, 8, 11],
        [1, 5, 10],
        [2, 4, 7],
        [3, 10, 12],
        [4, 6, 8],
        [2, 3, 9],
        [1, 7, 12],
        [2, 5, 6],
        [1, 3, 6],
        [4, 11, 12],
        [7, 8, 10],
        [1, 2, 11],
        [4, 9, 10],
        [5, 7, 11],
        [5, 8, 9]]
        
        codes can be found at http://www.inference.org.uk/mackay/codes/data.html
    """
    def __init__(self, alist: str or list):
        """
        alist_path : converts txt file to alist
        array : converts python list to alist
        """
        if isinstance(alist, str):
            with open(alist, 'r') as file:
                lines = file.readlines()
            self.alist = []
            for line in lines:
                row = list(map(int, line.strip().split()))
                row = list(filter(lambda x: x != 0, row))
                self.alist.append(row)

        elif isinstance(alist, list):
            self.alist = alist

        else:
            raise Exception("input needs to be a path to an alist txt file or a list in alist format")

        self.N, self.M = self.alist[0]  # N columns, M rows
        self.K = self.N - self.M    # information length

        self.Nmax, self.Mmax = self.alist[1]

        self.col_range = len(self.alist[2])
        self.Vdegrees, self.VdegreeDistribution = np.unique(self.alist[2], return_counts=True)

        self.row_range = len(self.alist[3])
        self.Sdegrees, self.SdegreeDistribution = np.unique(self.alist[3], return_counts=True)

        self.cols = self.alist[4:4 + self.col_range]
        self.rows = self.alist[4 + self.col_range:]

        self.Vnode = {v + 1: self.cols[v] for v in range(self.N)}
        self.Snode = {s + 1: self.rows[s] for s in range(self.M)}

    def __str__(self):
        return '\n'.join([' '.join(map(str, a)) for a in self.alist])

    def __mul__(self, other):
        """
        element-wise multiplication of a vector size N with parity check alist
        returns dictionary of result
        """
        other = np.array(other)
        if other.size != self.N:
            raise Exception(f"vector must have length: {self.N} but has length {other.size}")

        return {(s, v): other[v - 1] for v in self.Vnode for s in self.Vnode[v]}

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def __transpose__(self):
        """
        self.T
        """
        alist_T = [row for row in self.alist]
        alist_T[0][0], alist_T[0][1] = alist_T[0][1], alist_T[0][0]
        alist_T[1][0], alist_T[1][1] = alist_T[1][1], alist_T[1][0]
        alist_T[2], alist_T[3] = alist_T[3], alist_T[2]
        alist_T[4:4 + self.col_range], alist_T[4 + self.col_range:] = alist_T[4 + self.col_range:], alist_T[
                                                                                                    4:4 + self.col_range]
        return Alist(alist_T)

    def __matmul__(self, other):
        """
        binary matrix multiplication of parity check alist with non-sparse matrix
        self @ other
        """
        other = np.array(other)
        A, B = other.shape
        if A != self.N:
            raise Exception(f"input must have shape {(self.N, '?')}")

        result = np.zeros((self.M, B), dtype=int)
        for j, col in enumerate(other):
            result[:, j] = np.sum([col[s - 1] for s in self.Vnode[j - 1]]) % 2
        return result

    def __rmatmul__(self, other):
        """
        binary matrix multiplication of non-sparse matrix with parity check alist
        other @ self
        """
        other = np.array(other)
        A, B = other.shape
        if B != self.M:
            raise Exception(f"input must have shape {('?', self.M)}")

        result = np.zeros((A, self.N), dtype=int)
        for i, row in enumerate(other):
            result[i, :] = np.sum([row[v - 1] for v in self.Snode[i + 1]]) % 2
        return result

    @property
    def _sanity_check(self):
        alist_T = self.__transpose__
        return alist_T.__rmatmul__(self.G)

    @property
    def H(self):
        """
        matrix representation
        """
        H = np.zeros((self.M, self.N), dtype=int)
        for i, v in enumerate(self.Vnode):
            for s in self.Vnode[v]:
                H[s - 1, i] = 1
        return H

    @property
    def G(self):
        """
        generator matrix obtained as the nullspace of H (generally non-systematic)
        """
        G = np.array(galois.GF2(self.H).null_space())
        return G


###########################################################################################
if __name__ == "__main__":
    L = aList('Restructure\\Mackay LDPC alist\\96.3.963.txt')
    np.savetxt('96.3.963_sane', L._sanity_check, fmt='%i', delimiter=" ")
