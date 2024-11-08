import numpy as np
from TurboCodes.turbo import Turbo


class BICM(Turbo):
    def __init__(self, code, modulation):
        super().__init__()
        self.code, self.mod = code, modulation

    def __str__(self) -> str:
        pass

    def __repr__(self) -> str:
        pass

    @property
    def Rc(self):
        return self.code.Rc * self.Rp

    @property
    def spectral_efficiency(self):
        """
        spectral efficiency
        """
        return self.Rc * self.mod.m

    def encode(self, u):
        """
        """
        return self.code.encode(u)

    def modulate(self, ci):
        """
        """
        return self.mod.modulate(ci)

    def transmit(self, u):
        """
        """
        self._interleaver_check()
        c = self.encode(u)
        ci = self.interleave('turbo', c)
        cp = self.puncture(ci)
        x = self.modulate(cp)
        return x

    def turbo_demap(self, y, SNR_dB, iterations, mod_algorithm=None, code_algorithm=None):
        """

        """
        for _ in range(iterations):
            try:
                La = self.interleave('turbo', Le)
                _, Lc = self.mod.softdemodulation(y, SNR_dB, La, mod_algorithm)
            except NameError:
                _, Lc = self.mod.softdemodulation(y, SNR_dB, None, mod_algorithm)
            Lin = self.deinterleave('turbo', Lc)
            u_hat, _, Le = self.code.soft_decode(Lin, None, code_algorithm, mode="Lc")
        return u_hat

    def receive(self, y, SNR_dB, iterations=10, mod_algorithm=None, code_algorithm=None):
        """
        """
        yp = self.depuncture(y)
        u_hat = self.turbo_demap(yp, SNR_dB, iterations, mod_algorithm, code_algorithm)
        return u_hat

    def simulate(self, info_length, interleaver='random', trials=500, SNR_min=0, SNR_max=5, SNR_step=1,
                 decode_iterations=10, mod_algorithm=None, code_algorithm=None):
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
                w = np.random.normal(0, 10 ** (-SNR / 20), size=len(x))
                y = x + w
                u_hat = self.receive(y, SNR, decode_iterations, mod_algorithm, code_algorithm)
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
