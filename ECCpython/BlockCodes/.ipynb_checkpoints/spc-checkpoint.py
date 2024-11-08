import numpy as np
from BlockCodes.blockcode import BlockCode


class SPC(BlockCode):
    def __init__(self,
                 k: int = 3):
        """
        Single Parity Check code class

        Parameters
        ----------
        k: int
        length of the information word
        """
        self.k, self.n = k, k + 1
        self.type = 'SPC'
        self.P = np.ones(k, dtype=int)
        self.G = np.column_stack((np.eye(k, dtype=int), self.P))
        self.H = np.array([np.ones(k + 1, dtype=int)])
        super().__init__(k + 1, k, self.G, self.H, self.type, 2)

        self._funct = {
            'approximation': lambda t: np.prod(np.sign(t), axis=1) * np.min(np.abs(t), axis=1),
            'exact': lambda t: 2*np.arctanh((np.prod(np.tanh(t/2), axis=1))),
                       }
        self.default = 'approximation'
        self._funct['default'] = self._funct[self.default]

        # self.__softdecode__ = {
        #     True: lambda Lin, La=None, algorithm='approximation':
        #         tuple(LLR.ravel() for LLR in self._softdecode(Lin, La, algorithm)),
        #     False: lambda Lin, La=None, algorithm='approximation':
        #         self._softdecode(Lin, La, algorithm)
        #                         }

    # def softdecode(self, Lin, La=None, algorithm='approximation'):
    #     """
    #     Soft Input Decoder for SPC
    #     inputs
    #         Lin:   intrinsic LLRs of channel output (N x n array)
    #                 N: number of code words (rows)
    #         La:    a priori LLRs (one per information bit, N x k array, default = None)
    #         alg:   string indicating decoding algorithm (default = 'approximation')
    #                 'exact' performs the exact calculation using tanh and atanh functions
    #                 'approximation': approximation multiplying product of signs with minimum magnitude
    #     output:
    #         Lu     a posteriori LLRs of information bits (N x j array)
    #         Lc     a posteriori LLRs of code bits (N x j array), optional
    #     """
    #     return self.__softdecode__[self.flatten](Lin, La, algorithm)

    def softdecode(self, Lin, La=None, algorithm='approximation'):
        """
        Soft Input Decoder for SPC
        inputs
            Lin:   intrinsic LLRs of channel output (N x n array)
                    N: number of code words (rows)
            La:    a priori LLRs (one per information bit, N x k array, default = None)
            alg:   string indicating decoding algorithm (default = 'approximation')
                    'exact' performs the exact calculation using tanh and atanh functions
                    'approximation': approximation multiplying product of signs with minimum magnitude
        output:
            Lu     a posteriori LLRs of information bits (N x j array)
            Lc     a posteriori LLRs of code bits (N x j array), optional
        """
        try:
            funct = self._funct[algorithm]
        except KeyError:
            print('using default decoder')
            funct = self._funct['default']

        Lin = np.reshape(Lin, (-1, self.n))
        Le = np.zeros_like(Lin)

        Lyx = np.copy(Lin)
        if La is None:
            pass
        else:
            La = np.reshape(La, (-1, self.k))
            Lyx[:, :self.k] = Lyx[:, :self.k] + La

        for i in range(self.n):
            temp = np.delete(Lyx, i, axis=1)
            Le[:, i] = funct(temp)
        Lc = Lyx + Le

        Lu = Lc[:, :self.k]

        if self.flatten:
            Lu = Lu.ravel()
            Lc = Lc.ravel()

        return Lu, Lc

    # def softdecode(self, Lin, La=None, alg="approximation", mode="Lu"):
    #     """
    #     Soft Input Decoder for SPC
    #     inputs
    #         Lin:   intrinsic LLRs of channel output (N x n array)
    #                 N: number of code words (rows)
    #         La:    a priori LLRs (one per information bit, N x k array, default = None)
    #         alg:   string indicating decoding algorithm (default = 'approximation')
    #                 'exact' performs the exact calculation using tanh and atanh functions
    #                 'approximation': approximation multiplying product of signs with minimum magnitude
    #         mode:  string,
    #                 'Lu': only LLRs of information bits are computed
    #                 'Lc': LLRs of code and information bits are computed
    #     output:
    #         Lu     a posteriori LLRs of information bits (N x j array)
    #         Lc     a posteriori LLRs of code bits (N x j array), optional
    #     """
    #     if alg == "approximation":
    #         funct = lambda t: np.prod(np.sign(t), axis=1) * np.min(np.abs(t), axis=1)
    #     elif alg == "exact":
    #         funct = lambda t: 2 * np.arctanh((np.prod(np.tanh(t/2), axis=1)))
    #     else:
    #         raise Exception("Please enter: 'approximation' or 'exact'")
    #
    #     Le = np.zeros_like(Lin)
    #     Lyx = np.copy(Lin)
    #     if La is None:
    #         pass
    #     else:
    #         Lyx[:, :self.k] += La
    #
    #     for i in range(self.n):
    #         temp = np.delete(Lyx, i, axis=1)
    #         Le[:, i] = funct(temp)
    #     Lc = Lyx + Le
    #     Lu = Lc[:, :self.k]
    #
    #     if mode=='Lu':
    #         return Lu
    #     else:
    #         return Lu, Lc
    #     return Lu, Lc

    def simulate(self, info_blocks=5, trials=1000, SNR_min=0, SNR_max=10, SNR_step=1, algorithm="approximation"):
        import matplotlib.pyplot as plt
        info_length = info_blocks * self.k
        SNR_range = np.arange(SNR_min, SNR_max + 1, SNR_step)
        error = np.zeros((trials, len(SNR_range)))
        nn = 10 * np.log10(self.Rc)

        for trial in range(trials):
            for i, SNR in enumerate(SNR_range):
                u = np.random.randint(2, size=info_length)
                c = self.encode(u)
                x = 1 - 2 * c
                sigma_N = 10 ** (- SNR / 10)
                w = np.random.normal(0, np.sqrt(sigma_N), size=len(x))
                Lch = 4 / sigma_N
                Lin = Lch * (x + w)
                u_hat = self.softdecode(Lin, None, algorithm)[0]
                error[trial, i] = np.count_nonzero(u_hat - u)

        error_rate = np.mean(error, axis=0) / info_length

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.semilogy(SNR_range, error_rate)
        ax1.set_xlabel('$E_s/N_0(dB)$')
        ax1.set_ylabel('Symbol Error Rate ($P_s$)')
        ax1.set_title('Probability of Symbol Error over AWGN channel')
        ax1.set_xlim(SNR_range[0] - 1, SNR_range[-1] + 1)
        ax1.grid(True)
        ax2.semilogy(SNR_range - nn, error_rate)
        ax2.set_xlabel('$E_b/N_0(dB)$')
        ax2.set_ylabel('Bit Error Rate ($P_b$)')
        ax2.set_title('Probability of Bit Error over AWGN channel')
        ax2.set_xlim(SNR_range[0] - 1 - nn, SNR_range[-1] + 1 - nn)
        ax2.grid(True)
        plt.show()


##################################################################################################
if __name__ == "__main__":
    S3 = SPC(3)
    S3.simulate()
