import numpy as np
from Mapping.mapping import Mapping
from HelperFuncs import de2bi
from HelperFuncs import bi2de


class Modulation(Mapping):
    def __init__(self, m=1, coding_type='gray', modulation_type='PSK'):
        """
        Parameters
        ----------
        m : number of bits per symbol
        coding_type : 'gray', 'natural', 'antigray'
        modulation_type : 'ASK', 'PSK', 'QAM'
        """
        if modulation_type not in ['PSK', 'ASK', 'QAM']:
            raise ValueError("modulation_type needs to be 'PSK', 'ASK' or 'QAM'")

        self.modulation_type = modulation_type
        self.modDICT = {'PSK': self.PSK,
                        'ASK': self.ASK,
                        'QAM': self.QAM,
                        'new': 'self.new'}

        if self.modulation_type == 'QAM':
            if coding_type == 'gray':
                coding_type = 'QAM_gray'
            elif coding_type == 'antigray':
                coding_type = 'QAM_antigray'

        super().__init__(m, coding_type)

        # idea is that the coding array is a reordering of the modulation array (but with args)
        self._MOD_ = self.modDICT[self.modulation_type]()
        self.E = np.sqrt(np.sum(np.abs(self._MOD_) ** 2) / self.M)  # energy normalisation factor
        self._MOD = self._MOD_[self._sort] / self.E  # normalised
        self._X = self.modulate(self._Cbin)

        self.Lmax = 1000
        self._funct = {
            'exact': lambda input: np.log(np.sum(np.nan_to_num(np.exp(input), nan=self.Lmax))),
            'approximation': lambda input: np.max(input)
                        }
        self.default = 'approximation'
        self._funct['default'] = self._funct[self.default]

        self.type = f"{self.M} {self.modulation_type} {self.coding_type}"

    def __str__(self):
        return f"{self.modulation_type} modulation, \n {self.coding_type} coding, \n {self.m} bits per symbol, " \
               f"\n {self.M} symbols"

    def __repr__(self):
        return f"coding dictionary: \n {self._Cdict}, modulation dictionary: \n {self._X}"

    @property
    def spectral_efficiency(self):
        return self.m

    @property
    def constellation(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        constellation = self._X * self.E
        plt.scatter(constellation.real, constellation.imag, marker="o", label="Constellation Points")
        ax.grid(True, linewidth=0.25, linestyle="--", color="gray")

        for i, c in enumerate(self._Cbin):
            ax.annotate(str(c), (constellation.real[i], constellation.imag[i]), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, color='black')

        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"{self.M} {self.modulation_type} Constellation Diagram")
        # ax.legend()
        plt.show()

        return self.__str__()

    def modulate(self, data):
        """
        """
        data = self._zero_pad(data)
        data = np.reshape(data, (-1, self.m))
        dec = bi2de(data)
        x = np.array([self._MOD[d] for d in dec])
        return x

    def demodulate(self, data, return_Xhat=False):
        """
        hard demodulation - distance comparison
        """
        x = np.zeros_like(data, dtype=int)
        x_hat = np.zeros_like(data)
        for i, d in enumerate(data):
            distance = np.inf
            for j, z in enumerate(self._MOD):
                temp_d = np.abs(z - d)**2
                if temp_d < distance:
                    distance = temp_d
                    x[i] = j
                    x_hat[i] = self._MOD[j]
        u_hat = de2bi(x, self.m).ravel()

        # can be done with wrappers in future
        if self.padded_bits > 0:
            u_hat = u_hat[:-self.padded_bits]

        if return_Xhat:
            return u_hat, x_hat
        else:
            return u_hat

    def softdemodulate(self, y, sigma2_N, La=None, algorithm='approximation'):
        """
        """
        try:
            funct = self._funct[algorithm]
        except KeyError:
            print("using default algorithm")
            funct = self._funct['default']

        if La is None:
            La = np.zeros((len(y), self.m))
        else:
            try:
                La = np.reshape(La, (len(y), self.m))
            except Exception:
                raise ValueError("Apriori La has wrong length: check if its due to padding")

        Lc = np.zeros((len(y), self.m))
        for i, (y_, La_) in enumerate(zip(y, La)):
            for u in range(self.m):
                Lc_ = [[], []]  # temporary list to store sum terms
                for x, c in zip(self._X, self._Cbin):
                    Lc_[c[u]].append(- (np.abs(y_ - x) ** 2) / sigma2_N - np.dot(La_, c))
                Lc[i, u] = funct(Lc_[0]) - funct(Lc_[1])
        Lc = Lc.ravel()

        # can be done with wrappers in future
        if self.padded_bits > 0:
            Lc = Lc[:-self.padded_bits]

        return Lc

    def ASK(self):
        """
        """
        return (1 - self.M) / 2 + np.arange(self.M)

    def PSK(self):
        """
        """
        return np.array([np.exp((2 * np.pi * 1j / self.M) * n) for n in range(self.M)])

    def QAM(self):
        """
        """
        Re, Im = np.meshgrid(np.arange(1, self.m + 1), np.arange(self.m, 0, -1))
        Re = Re - np.mean(Re)
        Im = Im - np.mean(Im)
        return np.array((Re + 1j * Im).ravel())

    def simulate(self, info_length, trials=100, SNRdB_min=-1, SNRdB_max=30, SNRdB_step=1, algorithm="approximation"):
        import matplotlib.pyplot as plt
        SNRdB_range = np.arange(SNRdB_min, SNRdB_max + 1, SNRdB_step)
        sigma2_N_range = 10 ** (-SNRdB_range / 10)
        sigma_N_range = np.sqrt(sigma2_N_range)
        error_soft = np.zeros((trials, len(SNRdB_range)))
        error_hard = np.zeros((trials, len(SNRdB_range)))
        nn = 10 * np.log10(self.spectral_efficiency)

        for trial in range(trials):
            for i, (sigma2_N, sigma_N) in enumerate(zip(sigma2_N_range, sigma_N_range)):
                c = np.random.randint(2, size=info_length)
                x = self.modulate(c)
                w = np.random.normal(0, sigma_N/2, size=len(x)) + 1j * np.random.normal(0, sigma_N/2, size=len(x))
                y = x + w
                Lc = self.softdemodulate(y, sigma2_N, None, algorithm)
                c_soft = Lc <= 0
                c_hard = self.demodulate(y)
                error_soft[trial, i] = np.count_nonzero(c_soft - c)
                error_hard[trial, i] = np.count_nonzero(c_hard - c)

        error_rate_soft = np.mean(error_soft, axis=0) / info_length
        error_rate_hard = np.mean(error_hard, axis=0) / info_length

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.semilogy(SNRdB_range, error_rate_soft, 'b-', label='soft')
        ax1.semilogy(SNRdB_range, error_rate_hard, 'r--', label='hard')
        ax1.set_xlabel('$E_s/N_0(dB)$')
        ax1.set_ylabel('Symbol Error Rate ($P_s$)')
        ax1.set_title('Probability of Symbol Error over AWGN channel')
        ax1.set_xlim(SNRdB_range[0] - 1, SNRdB_range[-1] + 1)
        ax1.legend()
        ax1.grid(True)
        ax2.semilogy(SNRdB_range - nn, error_rate_soft, 'b-', label='soft')
        ax2.semilogy(SNRdB_range - nn, error_rate_hard, 'r--', label='hard')
        ax2.set_xlabel('$E_b/N_0(dB)$')
        ax2.set_ylabel('Bit Error Rate ($P_b$)')
        ax2.set_title('Probability of Bit Error over AWGN channel')
        ax2.set_xlim(SNRdB_range[0] - 1 - nn, SNRdB_range[-1] + 1 - nn)
        ax2.legend()
        ax2.grid(True)
        plt.show()


##################################################################################################
if __name__ == "__main__":
    M = Modulation(4, 'gray', 'QAM')
    M.simulate(100, 1)
