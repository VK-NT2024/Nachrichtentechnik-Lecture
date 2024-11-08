import numpy as np


class Puncturing:
    """
    Performs puncturing of codes.
    Arguments:
        puncture_matrix: array
            used to define the puncturing pattern of inputs

        code: ECC object (default = None)
            changes the code rate of the code
    """

    def __init__(self, puncture_matrix, code=None):

        self.punc_mat = np.array(puncture_matrix, dtype=bool)
        self.punc_pattern = np.where((self.punc_mat.transpose()).ravel() == 1)[0]
        self.max_ind = max(self.punc_pattern)
        self.size = self.punc_mat.size
        self.Rp = self.size / np.count_nonzero(self.punc_mat)
        if code is not None:
            code.Rc = code.Rc * self.Rp

    def puncture(self, input):
        """
        performs puncturing

        Arguments:
            input: array or list of arbitrary size
                code word to be punctured

        Return:
            punc_out: 1 dimensional array
        """
        punc_in = np.array(input).ravel()
        reps = np.size(punc_in) // self.size
        indices = np.array([self.punc_pattern + i * self.size for i in range(reps)]).ravel()
        punc_out = punc_in[indices]
        return punc_out

    def depuncture(self, input, unpunctured_size=None):
        """
        performs depuncturing

        Arguments:
            input: array or list of arbitrary size
                code word to be depunctured

            unpunctured_size: int (default = None)
                original size of the code word before puncturing. when None it uses the length that was used during the puncturing.

        Return:
            depunc_out: 1 dimensional array of size equal to unpunctured_size. If inpunctured_size is None it calculates the size
            of the original array.
        """
        depunc_in = np.array(input).ravel()

        if unpunctured_size is None:
            unpunctured_size = np.size(depunc_in) // self.Rp

        depunc_out = np.zeros(unpunctured_size)
        reps = unpunctured_size // self.size
        indices = np.array([self.punc_pattern + i * self.size for i in range(reps)]).ravel()
        depunc_out[indices] = depunc_in
        return depunc_out


###############
if __name__ == "__main__":
    A = [1, 2, 3]
    print(A[3])
