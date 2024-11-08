import numpy as np
# from numba import jit
from ConvolutionCodes.trellis import Trellis


class ConvCode(Trellis):
    def __init__(self,
                 generator_poly=None,
                 rsc_poly: int = 0,
                 terminated: bool = False):
        """
        class for convolutional codes

        Parameters
        ----------
            generator_poly : array or list ( default = [[1,1,1], [1,0,1]] )
                each row represents the binary coefficients of the generator polynomials.

            rsc_poly : int ( default = 0 )
                when chosen to be an integer other than zero, it determines which polynomial
                in the generator_poly array will be used to create RSC.
                If 0 is chosen, the code is created as NSC.

            terminated : boolean ( default = False )
                when chosen to be True the trellis is always terminated at state 0
        """
        if generator_poly is None:
            generator_poly = [[1, 1, 1], [1, 0, 1]]

        super().__init__(generator_poly, rsc_poly)

        self.ctail = None
        self.utail = None
        self.info_Length = None
        self.terminated = terminated
        # maximum LLR for soft-output decoding
        self.Lmax = 1000.0

        if self.systematic:
            self.type = "RSC convolutional code"
        else:
            self.type = "NSC convolutional code"

        # allows seamless compatibility in child classes without dealing with how the soft decode is implemented in
        # each class
        self._softdecode = {
            'maxlogmap': self.maxlogmap,
            'bcjr': self.bcjr,
            'logmap': self.logmap,
            'approximation': self.maxlogmap,
            'exact': self.bcjr
        }
        self.default = 'approximation'
        self._softdecode['default'] = self._softdecode[self.default]

        # suggestion for future: create soft decode subclass

    def __str__(self) -> str:
        return f''

    def __repr__(self) -> str:
        return f''

    @property
    def Rc(self):
        """
        returns code rate
        """
        rc = 1 / self.n
        if self.terminated and self.info_Length is not None:
            rc = self.info_Length / (self.info_Length * self.n + self.M)
        return rc

    def encode(self, u):
        """
        performs soft encoding for convolutional code

        Parameters
        ----------
            u: list or ndarray
                contains the information word as list of 0 and 1

        Return
        -------
            c: ndarray
                encoded version of u (list of 0 and 1)
        """
        u = np.array(u, dtype=int)
        self.info_Length = np.size(u)

        state = 0
        code_u = []
        code_t = []
        tail = []

        for bit in u:
            code_u = np.append(code_u, self.trellis['out'][state, bit, :])
            state = self.state_nexts[state][bit]

        if self.terminated:
            for _ in range(self.M):
                tail = np.append(tail, np.argmin(self.de_trellis['next'][state, :])).astype(int)
                code_t = np.append(code_t, self.trellis['out'][state, tail[-1], :])
                state = np.min(self.de_trellis['next'][state, :])
        self.utail = len(tail)
        self.ctail = len(code_t)
        c = np.append(code_u, code_t).astype(int)
        return c

    def softdecode(self, Lin, La=None, algorithm='maxlogmap'):
        """
        Parameters
        ----------

        Lin: list or ndarray
            intrinsic LLR

        La: list or ndarray
            apriori LLR

        algorithm: str or None
            maxlogmap (default), logmap, bcjr, exact, approximation, None, default

        Return
        -------
        Lu: list or ndarray
            information bit LLR

        Lc: list or ndarray
            code bit LLR
        """
        try:
            return self._softdecode[algorithm](Lin, La)
        except KeyError:
            print('using default decoder')
            return self._softdecode['default'](Lin, La)

    def _viterbi_metrics(self, y, window, initial_state, window_size):
        """
        not intended to be called by user
        """
        path_metric = np.full((window_size, 2 ** self.M, 2 ** self.M), np.inf)
        sum_metric = np.full((2 ** self.M, window_size + 1), np.inf)
        prev_state = np.full((2 ** self.M, window_size + 1), np.nan)
        sum_metric[initial_state, 0] = 0

        for i, l in enumerate(window):
            for s in self.states:
                for s_ in self.state_nexts[s]:
                    path_metric[i, s, s_] = np.sum((1 - 2 * self.state_outs[s, s_] - y[l, :]) ** 2)
                    tmp = sum_metric[s_, i + 1]
                    sum_metric[s_, i + 1] = np.min([tmp, sum_metric[s, i] + path_metric[i, s, s_]])
                    if sum_metric[s_, i + 1] < tmp:
                        prev_state[s_, i + 1] = s

        return path_metric, sum_metric, prev_state

    def viterbi_decode(self, y, decision_depth=None):
        """
        problem: branches don't match at edges with windows
        performs soft-in decoding for convolutional code via the Viterbi algorithm

        Parameters
        -----------
            y : list or ndarray
                the (real valued) output of the channel

            decision_depth: int or 'optimal' or None (default = None)
                implements the sliding window Viterbi algorithm with preferred window size.
                Default uses window with size equal to the length of the input y.

        Return
        -------
            u_hat : ndarray
                decoded version of y
        """
        y = np.reshape(y, (-1, self.n))
        L = len(y)
        initial_state = 0
        best_path = []
        u_hat = []

        if decision_depth is None:
            window_size = L
        elif decision_depth == 'optimal':
            window_size = np.min([5 * self.L_c, L])
        elif isinstance(decision_depth, int):
            window_size = np.min([decision_depth, L])
        else:
            raise Exception("choose a valid decision depth")

        rem = L % window_size
        window_range = L - rem

        windows = np.reshape(np.arange(window_range, dtype=int), (-1, window_size))
        # windows = np.arange(window_range, dtype=int)   # VK

        if rem != 0:
            last_window = np.arange(window_range, L, dtype=int)
        else:
            last_window = windows[-1]
            windows = windows[:-1]
            # last_window_size = 1
        last_window_size = len(last_window)

        for window in windows:
            _, sum_metric, prev_state = self._viterbi_metrics(y, window, initial_state, window_size)
            initial_state = np.argmin(sum_metric[:, -1])
            best_branch = [initial_state]
            for j, _ in enumerate(window):
                best_branch.append(int(prev_state[best_branch[-1], -j - 1]))
            best_path += best_branch
        else:  # for last window
            # if last_window_size>0:
            _, sum_metric, prev_state = self._viterbi_metrics(y, last_window, initial_state, last_window_size)
            if self.terminated:
                best_branch = [0]
            else:
                best_branch = [np.argmin(sum_metric[:, -1])]
            for j, _ in enumerate(last_window):
                best_branch.append(int(prev_state[best_branch[-1], -j - 1]))
            best_path += best_branch
        best_path = np.flip(best_path)

        for s0, s1 in zip(best_path[:-1], best_path[1:]):
            try:
                u_hat = np.append(u_hat, self.state_ins[s0, s1])
            except Exception:
                u_hat = np.append(u_hat, 0.5)  # decision for misalignment

        if self.terminated:
            u_hat = u_hat[:-self.utail]

        return u_hat

    # @jit
    def maxlogmap(self, Lin, La=None):
        """
        performs the soft in, soft out decoding using maxlogmap algorithm

        Parameters
        ----------

            Lin: list or ndarray
                intrinsic LLR

            La: list or ndarray
                apriori LLR

        Return
        -------
            Lu: list or ndarray
                information bit LLR

            Lc: list or ndarray
                code bit LLR

        references:
        ----------
            [1] https://paginas.fe.up.pt/~sam/textos/From%20BCJR%20to%20turbo.pdf
        """
        Lin = np.reshape(Lin, (-1, self.n))
        info_length = len(Lin)
        Lc = np.zeros((info_length, self.n))
        Lu = np.zeros(info_length)

        if La is None:
            La = np.zeros(info_length)
        else:
            La = np.array(La)
            if len(La) != info_length:
                raise Exception(f"length of La need to be {info_length} but has length {len(La)}")

        # ----------initialisation------------

        alpha = np.full((2 ** self.M, info_length), -np.inf)
        beta = np.full((2 ** self.M, info_length), -np.inf)
        gamma = np.zeros((info_length, 2 ** self.M, 2 ** self.M))
        alpha[0, 0] = 0

        if self.terminated:
            beta[0, info_length - 1] = 0
        else:
            beta[:, info_length - 1] = 0

        # -------------------gamma---------------------

        for l in np.arange(info_length):
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    temp = np.dot(Lin[l], 1 - 2 * self.state_outs[s, s_])
                    gamma[l, s, s_] = ((1 - 2 * u) * La[l] + temp)/2

        # --------------alpha and beta-----------------

        for l in np.arange(1, info_length):
            for s in self.states:
                for s_prev in self.state_prevs[s]:
                    alpha[s, l] = np.max([alpha[s, l], gamma[l - 1, s_prev, s] + alpha[s_prev, l - 1]])
                for s_next in self.state_nexts[s]:
                    beta[s, info_length - l - 1] = np.max([beta[s, info_length - l - 1], gamma[info_length - l, s, s_next] + beta[s_next, info_length - l]])

        # ---------------------Lc----------------------

        for l in np.arange(info_length):
            Lc_l = np.zeros((2, self.n))
            Lu_l = np.zeros(2)
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    temp = alpha[s, l] + gamma[l, s, s_] + beta[s_, l]
                    Lu_l[u] = np.max([Lu_l[u], temp])
                    for i, c in enumerate(self.state_outs[s, s_]):
                        Lc_l[c, i] = np.max([Lc_l[c, i], temp])
            Lu[l] = Lu_l[0] - Lu_l[1]
            Lc[l] = Lc_l[0] - Lc_l[1]
        Lc = Lc.ravel()

        if self.terminated:
            Lu = Lu[:-self.utail]
            Lc = Lc[:-self.ctail]

        return Lu, Lc

    #@jit
    def bcjr(self, Lin, La=None):
        """
        performs the soft in, soft out decoding using bcjr algorithm

        Parameters
        ----------

            Lin: list or ndarray
                intrinsic LLR

            La: list or ndarray
                apriori LLR

        Return
        -------
            Lu: list or ndarray
                information bit LLR

            Lc: list or ndarray
                code bit LLR

        references:
        -----------
            [1] https://paginas.fe.up.pt/~sam/textos/From%20BCJR%20to%20turbo.pdf
        """
        Lin = np.reshape(Lin, (-1, self.n))
        info_length = len(Lin)
        Lc = np.zeros((info_length, self.n))
        Lu = np.zeros(info_length)

        if La is None:
            La = np.zeros(info_length)
        else:
            La = np.array(La)
            if len(La) != info_length:
                raise Exception(f"length of La need to be {info_length} but has length {len(La)}")

        # ----------initialisation------------

        alpha = np.zeros((2 ** self.M, info_length))
        beta = np.zeros((2 ** self.M, info_length))
        gamma = np.zeros((info_length, 2 ** self.M, 2 ** self.M))
        alpha[0, 0] = 1

        if self.terminated:
            beta[0, info_length- 1] = 1
        else:
            beta[:, info_length- 1] = 1

        # -------------------gamma---------------------

        for l in np.arange(info_length):
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    temp = np.dot(Lin[l], 1 - 2 * self.state_outs[s, s_])
                    gamma[l, s, s_] = np.exp(((1 - 2 * u) * La[l] + temp)/2)

        # --------------alpha and beta-----------------

        for l in np.arange(1, info_length):
            for s in self.states:
                for s_prev in self.state_prevs[s]:
                    alpha[s, l] += gamma[l - 1, s_prev, s] * alpha[s_prev, l - 1]
                for s_next in self.state_nexts[s]:
                    beta[s, info_length- l - 1] += gamma[info_length- l, s, s_next] * beta[s_next, info_length- l]
            alpha[:, l] = alpha[:, l] / np.sum(alpha[:, l])  # normalisation
            beta[:, info_length- l - 1] = beta[:, info_length- l - 1] / np.sum(beta[:, info_length- l - 1])  # normalisation

        # ---------------------Lc----------------------

        for l in np.arange(info_length):
            Lc_l = np.zeros((2, self.n))
            Lu_l = np.zeros(2)
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    Prob_l = alpha[s, l] * gamma[l, s, s_] * beta[s_, l]
                    Lu_l[u] += Prob_l
                    for i, c in enumerate(self.state_outs[s, s_]):
                        Lc_l[c, i] += Prob_l
            Lu[l] = np.log(Lu_l[0] / Lu_l[1])
            Lc[l] = np.log(Lc_l[0] / Lc_l[1])
        Lc = Lc.ravel()

        if self.terminated:
            Lu = Lu[:-self.utail]
            Lc = Lc[:-self.ctail]

        return Lu, Lc

    #@jit
    def logmap(self, Lin, La=None):
        """
        performs the softin, soft out decoding using logmap algorithm

        Parameters
        ----------

            Lin: list or ndarray
                intrinsic LLR

            La: list or ndarray
                apriori LLR

        Return
        -------
            Lu: list or ndarray
                information bit LLR

            Lc: list or ndarray
                code bit LLR

        references:
        ----------
            [1] https://paginas.fe.up.pt/~sam/textos/From%20BCJR%20to%20turbo.pdf
        """
        Lin = np.reshape(Lin, (-1, self.n))
        info_length= len(Lin)
        Lc = np.zeros((info_length, self.n))
        Lu = np.zeros(info_length)

        if La is None:
            La = np.zeros(info_length)
        else:
            La = np.array(La)
            if len(La) != info_length:
                raise Exception(f"length of La need to be {info_length} but has length {len(La)}")

        max_star = lambda A: np.max(A) + np.log(1 + np.exp(-np.abs(A[0] - A[1])))

        # ----------initialisation------------

        alpha = np.full((2 ** self.M, info_length), -np.inf)
        beta = np.full((2 ** self.M, info_length), -np.inf)
        gamma = np.zeros((info_length, 2 ** self.M, 2 ** self.M))
        alpha[0, 0] = 0

        if self.terminated:
            beta[0, info_length - 1] = 0
        else:
            beta[:, info_length - 1] = 0

        # -------------------gamma---------------------

        for l in np.arange(info_length):
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    temp = np.dot(Lin[l], 1 - 2 * self.state_outs[s, s_])
                    gamma[l, s, s_] = ((1 - 2 * u) * La[l] + temp)/2

        # --------------alpha and beta-----------------

        for l in np.arange(1, info_length):
            for s in self.states:
                for s_prev in self.state_prevs[s]:
                    alpha[s, l] = max_star([alpha[s, l], gamma[l - 1, s_prev, s] + alpha[s_prev, l - 1]])
                for s_next in self.state_nexts[s]:
                    beta[s, info_length - l - 1] = max_star([beta[s, info_length - l - 1], gamma[info_length - l, s, s_next] + beta[s_next, info_length - l]])

        # ---------------------Lc----------------------

        for l in np.arange(info_length):
            Lc_l = np.zeros((2, self.n))
            Lu_l = np.zeros(2)
            for s in self.states:
                for u, s_ in enumerate(self.state_nexts[s]):
                    temp = alpha[s, l] + gamma[l, s, s_] + beta[s_, l]
                    Lu_l[u] = max_star([Lu_l[u], temp])
                    for i, c in enumerate(self.state_outs[s, s_]):
                        Lc_l[c, i] = max_star([Lc_l[c, i], temp])
            Lu[l] = Lu_l[0] - Lu_l[1]
            Lc[l] = Lc_l[0] - Lc_l[1]
        Lc = Lc.ravel()

        if self.terminated:
            Lu = Lu[:-self.utail]
            Lc = Lc[:-self.ctail]

        return Lu, Lc

    def simulate(self, info_length=100, trials=100, SNR_min=0, SNR_max=10, SNR_step=1, algorithm="maxlogmap"):
        """
        simulates the performance of soft output decoding with AWGN channel

        performs the softin, soft out decoding using maxlogmap algorithm

        Parameters
        ----------
            info_length : int = 100
                length of randomly generated information bits

            trials : int = 500
                granularity of the simulation - higher leads to smoother plot but more processing time

            SNR_min : int = 0
                the starting SNR

            SNR_max : int = 10
                the ending SNR

            SNR_step : int = 1
                the steps of SNR

            algorithm: str or None
                maxlogmap (default), logmap, bcjr, exact, approximation, None, default

        Return
        -------
            (2, 1) plot
                Probability of Symbol Error over AWGN channel
                Probability of Bit Error over AWGN channel
        """
        import matplotlib.pyplot as plt
        SNRdB_range = np.arange(SNR_min, SNR_max + 1, SNR_step)
        sigma2_N_range = 0.5 * 10 ** (- SNRdB_range / 10)
        sigma_N_range = np.sqrt(sigma2_N_range)
        error = np.zeros((trials, len(SNRdB_range)))
        nn = 10 * np.log10(self.Rc)

        for trial in range(trials):
            for i, (sigma2_N, sigma_N) in enumerate(zip(sigma2_N_range, sigma_N_range)):
                u = np.random.randint(2, size=info_length)
                c = self.encode(u)
                x = 1 - 2 * c
                w = np.random.normal(0, sigma_N, size=len(x))
                Lch = 2 / sigma2_N
                Lin = Lch * (x + w)
                Lu, _ = self.softdecode(Lin, None, algorithm)
                u_hat = Lu <= 0
                error[trial, i] = np.count_nonzero(u_hat - u)

        error_rate = np.mean(error, axis=0) / info_length

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


if __name__ == "__main__":
    ConvCode([[1, 1, 1], [1, 0, 1]], 1).simulate(algorithm="maxlogmap")
