import numpy as np
from TurboCodes.turbo import Turbo


class Parallel(Turbo):
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
        to-do: add Rc property to all codes
        """
        return 1 / (1 / self.code1.Rc + 1 / self.code2.Rc - 1) * self.Rp

    @property
    def spectral_efficiency(self):
        """
        spectral efficiency
        """
        return self.Rc * self.m

    def encode(self, u):
        """

        """
        u = np.asarray(u).flatten()
        ui = self.interleave('turbo', u)
        c1 = self.code1.encode(u)
        c2 = self.code2.encode(ui)
        return c1, c2

    def transmit(self, u):
        """

        """
        self._interleaver_check()
        c1, c2 = self.encode(u)
        c = self.merge(c1, c2)
        cp = self.puncture(c)
        x = 1 - 2 * cp
        return x

    def turbo_decode(self, Lin1, Lin2, iterations, code1_algorithm=None, code2_algorithm=None):
        """
        to-do: rename to turbo_decode in all codes
        """
        for _ in range(iterations):
            try:
                La1 = self.deinterleave('turbo', Le2)
                u_hat, Lu1, Le1 = self.code1.soft_decode(Lin1, La1, code1_algorithm, mode="Lu")
            except NameError:
                u_hat, Lu1, Le1 = self.code1.soft_decode(Lin1, None, code1_algorithm, mode="Lu")
            La2 = self.interleave('turbo', Le1)
            _, Lu2, Le2 = self.code2.soft_decode(Lin2, La2, code2_algorithm, mode="Lu")
        return u_hat

    def receive(self, Lin, iterations=10, algorithm_code1=None, algorithm_code2=None):
        """
        """
        self._interleaver_check()
        Lin_ = np.real(Lin)
        Linp = self.depuncture(Lin_)
        Lin1, Lin2 = self.demerge(Linp)
        u_hat = self.turbo_decode(Lin1, Lin2, iterations, algorithm_code1, algorithm_code2)
        return u_hat

    def simulate(self, info_length, interleaver='random', trials=500, SNR_min=0, SNR_max=5, SNR_step=1,
                 decode_iterations=10, algorithm_code1=None, algorithm_code2=None):
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
                var_N = 10 ** (- SNR / 10)
                w = np.random.normal(0, np.sqrt(var_N), size=len(x))
                Lch = 4 / var_N
                Lin = Lch * (x + w)
                u_hat = self.receive(Lin, decode_iterations, algorithm_code1, algorithm_code2)
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


if __name__ == "__main__":
    from BlockCodes.spc import SPC
    from ConvolutionCodes.convcode import ConvCode
    PP = Parallel(SPC(3), ConvCode([[1, 1, 1], [1, 0, 1]], 1))
    PP.simulate(300)
