import numpy as np
from Modulation.mapping import Mapping
from Tools.helperFuncs import de2bi
from Tools.helperFuncs import bi2de


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
        modDICT = {'PSK': self.PSK,
                   'ASK': self.ASK,
                   'QAM': self.QAM,
                   'new': 'self.new'}

        if self.modulation_type == 'QAM':
            if coding_type == 'gray':
                coding_type = 'QAM_gray'
            elif coding_type == 'antigray':
                coding_type = 'QAM_antigray'

        super().__init__(m, coding_type)

        self.constellation = modDICT[self.modulation_type]()  # [self._sort]
        self.E = np.mean(np.abs(self.constellation) ** 2)  # energy normalisation factor
        self.constellationNorm = self.constellation / np.sqrt(self.E)  # normalised
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
    def plot_constellation(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        constellation = self.constellation
        plt.scatter(constellation.real, constellation.imag, marker="o", label="Constellation Points")
        ax.grid(True, linewidth=0.25, linestyle="--", color="gray")

        if self.modulation_type == 'ASK':
            for i, c in enumerate(self._Cbin):
                ax.annotate(str(c), (constellation.real[i], constellation.imag[i]), textcoords="offset points",
                            xytext=(0, 10 * (-1) ** i), ha='center', fontsize=8, color='black')
        else:
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
        x = np.array([self.constellationNorm[d] for d in dec])
        return x

    def demodulate(self, data, return_Xhat=False):
        """
        hard demodulation - distance comparison
        """
        x = np.zeros_like(data, dtype=int)
        x_hat = np.zeros_like(data)
        for i, d in enumerate(data):
            distance = np.inf
            for j, z in enumerate(self.constellationNorm):
                temp_d = np.abs(z - d) ** 2
                if temp_d < distance:
                    distance = temp_d
                    x[i] = j
                    x_hat[i] = self.constellationNorm[j]
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
        return (1 - self.M) + np.arange(self.M) * 2

    def PSK(self):
        """
        """
        return np.array([np.exp((2 * np.pi * 1j / self.M) * n) for n in range(self.M)])

    def QAM(self):
        """
        """
        sqrtM = np.sqrt(self.M)
        Re, Im = np.meshgrid(np.arange(1 - sqrtM, sqrtM, 2), np.arange(1 - sqrtM, sqrtM, 2))
        return np.array((Re + 1j * Im).ravel())

    def impulse_shaping(self, data, filter_type, filter_length, symbol_duration, sampling_rate, rolloff=0.25, f3dB=0.3):
        """
        Performs convolution of symbol sequence with filter impulse response

        Parameters
        ----------
        data : vector of data symbols (1-D ndarray of floats)
        filter : string can be 'rc', 'rrc', 'rect','tri'
        filter_length : Length of the filter in samples.
        sumbol_duration : Symbol period in seconds.
        sampling_rate : Sampling Rate in Hz.
        rolloff : Rolloff factor in casse of filter=='rc' or filter='rrc' (valid values are [0, 1]).

        Returns
        -------
        x : 1-D ndarray (float)
            Impulse response of the raised cosine filter.

        time_idx : 1-D ndarray (float)
            Array containing the time indices, in seconds, for the impulse response.
        """

        if filter_type == 'rrc':
            _, g = self.generate_g_rrc(filter_length, rolloff, symbol_duration, sampling_rate)
        elif filter_type == 'rc':
            _, g = self.generate_g_rc(filter_length, rolloff, symbol_duration, sampling_rate)
        elif filter_type == 'rect':
            _, g = self.generate_g_rect(filter_length, symbol_duration, sampling_rate)
        elif filter_type == 'tri':
            _, g = self.generate_g_tri(filter_length, symbol_duration, sampling_rate)
        elif filter_type == 'gauss':
            _, g = self.generate_g_gauss(filter_length, sampling_rate, f3dB)
        else:
            raise ValueError("filter_type needs to be 'rc', 'rrc', 'rect', 'tri' or 'gauss'")

        # number of symbols
        N_symbols = data.shape[0]

        # upsampling factor
        w = int(sampling_rate * symbol_duration)

        # oversampling data symbols by inserting w zeros
        data_up = np.append(data.reshape((-1, 1)), np.zeros((N_symbols, w - 1)), axis=1).flatten()

        x = np.convolve(data_up, g) * symbol_duration

        # time axis
        time = (np.arange(N_symbols * w + filter_length - 1) - filter_length / 2) / sampling_rate

        return time, x

    def generate_g_rc(self, N, r, Ts, Fs):
        """
        Generates a raised cosine (RC) filter (FIR) impulse response.

        Parameters
        ----------
        N : Length of the filter in samples.
        r : Roll off factor (Valid values are [0, 1]).
        Ts : Symbol period in seconds.
        Fs : Sampling Rate in Hz.

        Returns
        -------
        gh_rc : 1-D ndarray (float)
            Impulse response of the raised cosine filter.

        time_idx : 1-D ndarray (float)
            Array containing the time indices, in seconds, for the impulse response.
        """

        T_delta = 1 / float(Fs)
        time_idx = ((np.arange(N) - N / 2)) * T_delta
        sample_num = np.arange(N)
        g_rc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x - N / 2) * T_delta
            if t == 0.0:
                g_rc[x] = 1.0
            elif r != 0 and t == Ts / (2 * r):
                g_rc[x] = (np.pi / 4) * (np.sin(np.pi * t / Ts) / (np.pi * t / Ts))
            elif r != 0 and t == -Ts / (2 * r):
                g_rc[x] = (np.pi / 4) * (np.sin(np.pi * t / Ts) / (np.pi * t / Ts))
            else:
                g_rc[x] = (np.sin(np.pi * t / Ts) / (np.pi * t / Ts)) * (np.cos(np.pi * r * t / Ts) / (1 - (((2 * r * t) / Ts) * ((2 * r * t) / Ts))))

        return time_idx, g_rc / Ts

    def generate_g_rrc(self, N, r, Ts, Fs):
        """
        Generates a root raised cosine (RRC) filter (FIR) impulse response.

        Parameters
        ----------
        N : Length of the filter in samples.
        r : Roll off factor (Valid values are [0, 1]).
        Ts : Symbol period in seconds.
        Fs : Sampling Rate in Hz.

        Returns
        ---------
        g_rrc : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.

        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for the impulse response.
        """

        T_delta = 1 / float(Fs)
        time_idx = ((np.arange(N) - N / 2)) * T_delta
        sample_num = np.arange(N)
        g_rrc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x - N / 2) * T_delta
            if t == 0.0:
                g_rrc[x] = 1.0 - r + (4 * r / np.pi)
            elif r != 0 and t == Ts / (4 * r):
                g_rrc[x] = (r / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * r)))) + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * r)))))
            elif r != 0 and t == -Ts / (4 * r):
                g_rrc[x] = (r / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * r)))) + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * r)))))
            else:
                g_rrc[x] = (np.sin(np.pi * t * (1 - r) / Ts) + 4 * r * (t / Ts) * np.cos(np.pi * t * (1 + r) / Ts)) / (np.pi * t * (1 - (4 * r * t / Ts) * (4 * r * t / Ts)) / Ts)

        return time_idx, g_rrc / Ts

    def generate_g_rect(self, N, Ts, Fs):
        """
        Generates a rectangular filter impulse response.

        Parameters
        ----------
        N : Length of the filter in samples.
        Ts : Duration of rectangular pulse period in seconds.
        Fs : Sampling Rate in Hz.

        Returns
        ---------
        g_rect : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.

        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for the impulse response.
        """

        T_delta = 1 / float(Fs)
        time_idx = ((np.arange(N) - N / 2)) * T_delta
        g_rect = np.zeros(N, dtype=float)

        g_rect[time_idx>-Ts/2] = 1.0
        g_rect[time_idx > Ts / 2] = 0.0

        return time_idx, g_rect / Ts

    def generate_g_tri(self, N, Ts, Fs):
        """
        Generates a triangular filter impulse response.

        Parameters
        ----------
        N : Length of the filter in samples.
        Ts : Duration of triangular pulse in seconds.
        Fs : Sampling Rate in Hz.

        Returns
        ---------
        g_rect : 1-D ndarray of floats
            Impulse response of the root raised cosine filter.

        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for the impulse response.
        """

        T_delta = 1 / float(Fs)
        time_idx = ((np.arange(N) - N / 2)) * T_delta
        g_tri = np.zeros(N, dtype=float)
        slope = 2 / Ts

        g_tri[time_idx>-Ts/2] = (time_idx[time_idx>-Ts/2]+Ts/2) * slope
        g_tri[time_idx>0] = 1 - time_idx[time_idx>0] * slope
        g_tri[time_idx > Ts / 2] = 0.0

        return time_idx, g_tri / Ts

    def generate_g_gauss(self, N, Fs, f3dB):
        """
        Generates a Gaussian filter impulse response.

        Parameters
        ----------
        N : Length of the filter in samples.
        Ts : Duration of triangular pulse in seconds.
        Fs : Sampling Rate in Hz.
        f3dBT : Frequency with 3dB attenuation of transfer function normalized to symbol duration T

        Returns
        ---------
        g_gauss : 1-D ndarray of floats
            Impulse response of the Gaussian filter.

        time_idx : 1-D ndarray of floats
            Array containing the time indices, in seconds, for the impulse response.
        """

        T_delta = 1.0 / Fs
        time = ((np.arange(N) - N / 2)) * T_delta

        omega3dB = 2 * np.pi * f3dB

        g_gauss = omega3dB / np.sqrt(2*np.pi*np.log(2)) * np.exp(-(omega3dB * time)**2 / 2 / np.log(2))

        return time, g_gauss

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
                w = np.random.normal(0, sigma_N / 2, size=len(x)) + 1j * np.random.normal(0, sigma_N / 2, size=len(x))
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
