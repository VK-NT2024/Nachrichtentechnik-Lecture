import numpy as np
from TurboCodes.turbo import Turbo


class Serial(Turbo):
    def __init__(self,
                 code1,
                 code2):
        """
        """
        super().__init__()
        self.code1, self.code2 = code1, code2

    @property
    def Rc(self):
        """
        code rate
        """
        return self.code1.Rc * self.code2.Rc * self.Rp

    @property
    def spectral_efficiency(self):
        """
        spectral efficiency
        """
        return self.Rc * self.m

    def encode(self, u):
        """
        """
        self._interleaver_check()
        u = np.asarray(u).flatten()
        c1 = self.code1.encode(u)
        c1i = self.interleave('turbo', c1)
        c2 = self.code2.encode(c1i)
        return c2

    def transmit(self, u):
        """

        """
        c = self.encode(u)
        cp = self.puncture(c)
        x = self.modulate(cp)
        return x

    def turbo_decode(self, Lin, iterations, algorith_code1=None, algorithm_code2=None):
        """
        """
        for _ in range(iterations):
            try:
                La2 = self.interleave('turbo', Le1)
                _, Lu2, Le2 = self.code2.turbo_decode(Lin, La2, algorithm_code2, mode="Lu")
            except NameError:
                _, Lu2, Le2 = self.code2.turbo_decode(Lin, None, algorithm_code2, mode="Lu")
            Lin1 = self.deinterleave('turbo', Le2)
            u_hat, _, Le1 = self.code1.turbo_decode(Lin1, None, algorithm_code2, mode="Lc")
        return u_hat


    def receive(self, Lin, iterations):
        """
        """
        Linp = self.depuncture(Lin)
        Lin_ = np.real(Linp)
        u_hat = self.turbo_decode(Lin_, iterations)
        return u_hat

    def simulate(self, info_length, interleaver='random', trials=100, SNR_min=0, SNR_max=5, SNR_step=1,
                 decode_iterations=None):
        """
        """
        import matplotlib.pyplot as plt
        self.set_interleaver(interleaver, info_length)
        SNR_range = np.arange(SNR_min, SNR_max + 1, SNR_step)
        error = np.zeros((trials, len(SNR_range)))
        nn = 10 * np.log10(self.spectral_efficiency)

        for trial in range(trials):
            for i, SNR in enumerate(SNR_range):
                u = np.random.randint(2, size=info_length)
                x = self.transmit(u)
                sigma_N = 10 ** (- SNR / 10)
                w = np.random.normal(0, np.sqrt(sigma_N), size=len(x))
                Lch = 4 / sigma_N
                Lin = Lch * (x + w)
                u_hat = self.receive(Lin, decode_iterations)
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


#######################################################################
if __name__ == "__main__":
    pass


