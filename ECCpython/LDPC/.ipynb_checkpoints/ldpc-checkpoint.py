import numpy as np
from LDPC.alist import Alist
from info_theory import calc_MI


class LDPC(Alist):
    def __init__(self, alist: str or list):
        """

        """
        super().__init__(alist)
        self.Rc = self.K / self.N

        self._funct = {
            "approximation": lambda L: np.prod(np.sign(L)) * np.min(np.abs(L)),
            "exact": lambda L: 2 * np.arctanh(np.prod(np.tanh(L / 2)))
        }
        self.default = "approximation"
        self._funct["default"] = self._funct[self.default]

    def encode(self, message):
        return message @ self.G % 2

    def message_passing(self, Lin, iterations=10, algorithm='approximation'):
        """
        input
        Lin : Lch * y
        iterations : number of belief propagation iterations
        algorithm : approximation or exact

        output:
        Lc array
            each row contains the result from each iteration
            final row contains final result, when relevant use Lc[-1] to obtain it
        """
        try:
            funct = self._funct[algorithm]
        except KeyError:
            print(f"using default decoding algorithm: {self.default}")
            funct = self._funct["default"]

        Le_ = {}
        Lc_ = super().__mul__(Lin)
        Lc = np.zeros((iterations, self.N))

        for iter in range(iterations):
            Le = np.zeros(self.N)
            for s in self.Snode:
                for v in self.Snode[s]:
                    input_to_Snode = np.array([Lc_[(s, i)] for i in self.Snode[s] if i != v])
                    Le_[(s, v)] = funct(input_to_Snode)
            for v in self.Vnode:
                for s in self.Vnode[v]:
                    input_to_Vnode = np.array([Le_[(j, v)] for j in self.Vnode[v] if j != s])
                    Lc_[(s, v)] = Lin[v - 1] + np.sum(input_to_Vnode)
            for (s, v), Le_sv in Le_.items():
                Le[v - 1] += Le_sv
            Lc[iter] = Lin + Le

        return Lc

    def softdecode(self, Lin, iterations=10, algorithm="approximation"):
        return self.message_passing(Lin, iterations, algorithm)[-1]

    def simulate(self, trials=100, message_passing_iterations=10, SNR_min=0, SNR_max=10, SNR_step=1,
                 algorithm="approximation"):
        import matplotlib.pyplot as plt

        SNRdB_range = np.arange(SNR_min, SNR_max + 1, SNR_step)
        sigma2_N_range = 0.5 * 10 ** (- SNRdB_range / 10)
        sigma_N_range = np.sqrt(sigma2_N_range)
        error = np.zeros((trials, len(SNRdB_range)))
        nn = 10 * np.log10(self.Rc)

        for trial in range(trials):
            for i, (sigma2_N, sigma_N) in enumerate(zip(sigma2_N_range, sigma_N_range)):
                u = np.random.randint(2, size=self.K)
                c = self.encode(u)
                x = 1 - 2 * c
                w = np.random.normal(0, sigma_N, size=len(x))
                Lch = 2 / sigma2_N
                Lin = Lch * (x + w)
                Lc, _ = self.softdecode(Lin, message_passing_iterations, algorithm)
                c_hat = Lc <= 0
                error[trial, i] = np.count_nonzero(c_hat - u)

        error_rate = np.mean(error, axis=0) / self.K
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.semilogy(SNRdB_range, error_rate, 'b-', label=f'{algorithm}')
        ax1.set_xlabel('$E_s/N_0(dB)$')
        ax1.set_ylabel('Symbol Error Rate ($P_s$)')
        ax1.set_title('Probability of Symbol Error over AWGN channel')
        ax1.set_xlim(SNRdB_range[0] - 1, SNRdB_range[-1] + 1)
        ax1.legend()
        ax1.grid(True)
        ax2.semilogy(SNRdB_range - nn, error_rate, 'b-', label=f'{algorithm}')
        ax2.set_xlabel('$E_b/N_0(dB)$')
        ax2.set_ylabel('Bit Error Rate ($P_b$)')
        ax2.set_title('Probability of Bit Error over AWGN channel')
        ax2.set_xlim(SNRdB_range[0] - 1 - nn, SNRdB_range[-1] + 1 - nn)
        ax2.legend()
        ax2.grid(True)
        plt.show()


###############################################################################################
if __name__ == "__main__":
    pass
