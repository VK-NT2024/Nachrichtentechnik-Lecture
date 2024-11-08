import numpy as np
import matplotlib.pyplot as plt
from EXIT.info_theory import mi_dmc, mi_awgn, calc_MI


class EXIT:
    def __init__(self,
                 outer_block,
                 inner_block=None,
                 mode='code',
                 info_algorithm='sklearn',
                 steps=0.1,
                 repetitions=100):
        """
        provides functionality for calculating mutual information and plotting EXIT charts

        Parameters:
        -----------
            outer_block : Class: BlockCode, ConvCode
                outer block (not necessarily systematic)

            inner_block : Class: BlockCode, ConvCode, Modulation
                systematic inner block

            mode : str
                "code", "parallel", "serial", "bicm"

            algorithm : str (default = "histogram")
                "reliability", "histogram"

            precision : float (default = 0.1)
                determines the steps in sigma2_a

            reliability: int (default = 100)
                number of repetitions to achieve better performance. Higher number increases processing time
        """
        if mode not in ["code", "parallel", "serial", "bicm"] and inner_block is None:
            mode = 'code'

        if mode not in ["code", "parallel", "serial", "bicm", "ldpc"]:
            raise Exception("mode needs to be: 'code', 'parallel', 'serial', 'bicm' or 'ldpc'")

        if mode in ["parallel", "serial", "bicm"] and inner_block is None:
            raise Exception("inner_block code needs to be defined")

        self.mode = mode
        self.block1 = outer_block  # outer block (not necessarily systematic)
        self.block2 = inner_block  # systematic inner block
        self.algorithm = info_algorithm
        self.steps = steps
        self.reps = repetitions

        self.block1.flatten = True
        self.block2.flatten = True

        self.sigma2_a_range = np.arange(start=0, stop=40, step=steps)
        self.sigma_a_range = np.sqrt(self.sigma2_a_range)
        self.L = self.sigma2_a_range.size

        self.calc_MI = {
            'histogram': lambda u, Lu, bins: self.hist_MI(u, Lu, bins),
            'reliability': lambda u, Lu, bins: self.rel_MI(Lu),
            'sklearn': lambda u, Lu, bins: self.sk_MI(u, Lu)
        }

    @staticmethod
    def sk_MI(binary, continuous):
        """
        """
        from sklearn.feature_selection import mutual_info_classif
        X = binary
        y = np.reshape(continuous, (-1, 1))
        return mutual_info_classif(y, X)[0]

    @staticmethod
    def rel_MI(X):
        """
        """
        L = len(X)

        pX_0 = 1 / (1 + np.exp(np.abs(X)))
        pX_1 = 1 / (1 + np.exp(-np.abs(X)))
        MI = np.sum(pX_0 * np.log2(pX_0) + pX_1 * np.log2(pX_1)) / L
        return MI

    def hist_MI(self, X, Y, bins):
        """
        histogram based mutual information calculation
        """
        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = self.entropy(c_X)
        H_Y = self.entropy(c_Y)
        H_XY = self.entropy(c_XY)

        MI = H_X + H_Y - H_XY
        return MI

    @staticmethod
    def entropy(c):
        """
        """
        c_normalized = c / (np.sum(c) + np.finfo(np.float64).eps)
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = - np.sum(c_normalized * np.log2(c_normalized))
        return H

    def EXIT_ldpc_serial(self, iterations=100, ldpc_algorithm='default'):
        I_a = np.zeros(self.L)
        I_e = np.zeros(self.L)
        L_a = int(self.block1.k * 1 / self.block1.Rc)

        for _ in range(self.reps):
            u = np.random.randint(2, size=self.block1.k)
            c = self.block1.encode(u)
            c_ = 1 - 2 * c

            for i, (sigma2_a, sigma_a) in enumerate(zip(self.sigma2_a_range, self.sigma_a_range)):
                n_a = np.random.normal(0, sigma_a, size=L_a)
                La = 0.5 * sigma2_a * c_ + n_a
                _, Lc = self.block1.message_passing(La, iterations, ldpc_algorithm)
                I_a[i] = I_a[i] + self.calc_MI[self.algorithm](c, La, bins=L_a)
                I_e[i] = I_e[i] + self.calc_MI[self.algorithm](c, Lc[-1], bins=L_a)

        I_a = I_a / self.reps
        I_e = I_e / self.reps

        return I_a, I_e

    def EXIT_ldpc(self, SNR_dB, iterations=1000, ldpc_algorithm='default'):
        I_a = np.zeros(iterations + 1)
        I_e = np.zeros(iterations + 1)
        sigma2_N = 10 ** (- SNR_dB / 10)
        sigma_N = np.sqrt(sigma2_N)

        for _ in range(self.reps):
            u = np.random.randint(2, size=self.block1.k)
            c = self.block1.encode(u)
            x = 1 - 2 * c
            w = np.random.normal(0, sigma_N, size=len(x))
            y = x + w
            Lin = 4 * sigma2_N * y
            I_a[0] = I_a[0] + self.calc_MI[self.algorithm](c, Lin)
            Lc = self.block1.message_passing(Lin, iterations, ldpc_algorithm)

            for i, Lc_i in enumerate(Lc):
                I_a[i + 1] = I_a[i + 1] + self.calc_MI[self.algorithm](c, Lc_i)
                I_e[i + 1] = I_e[i + 1] + self.calc_MI[self.algorithm](c, Lc_i - Lin)

        I_a = I_a / self.reps
        I_e = I_e / self.reps

        return I_a, I_e

    def EXIT_map(self, info_length, SNR_dB, map_algorithm='default'):
        """
        """
        I_a = np.zeros(self.L)
        I_e = np.zeros(self.L)
        sigma2_N = 10 ** (- SNR_dB / 10)
        sigma_N = np.sqrt(sigma2_N)

        for _ in range(self.reps):
            c = np.random.randint(2, size=info_length)
            x = self.block2.modulate(c)
            c_ = 1 - 2 * self.block2._zero_pad(c)
            w = np.random.normal(0, sigma_N, size=len(x))
            y = x + w

            for i, (sigma2_a, sigma_a) in enumerate(zip(self.sigma2_a_range, self.sigma_a_range)):
                n_a = np.random.normal(0, sigma_a, size=len(c))
                La = 0.5 * sigma2_a * c_ + n_a
                Lc = self.block2.softdemodulate(y, sigma2_N, La, map_algorithm)
                I_a[i] = I_a[i] + self.calc_MI[self.algorithm](c, La, bins=info_length)
                I_e[i] = I_e[i] + self.calc_MI[self.algorithm](c, Lc, bins=info_length)

        I_a = I_a / self.reps
        I_e = I_e / self.reps

        return I_a, I_e

    def EXIT_nonsys(self, info_length, code_algorithm='default'):
        """
        uses no apriori LLR
        """
        I_a = np.zeros(self.L)
        I_e = np.zeros(self.L)
        L_a = int(info_length * 1 / self.block1.Rc)

        for _ in range(self.reps):
            u = np.random.randint(2, size=info_length)
            c = self.block1.encode(u)
            c_ = 1 - 2 * c

            for i, (sigma2_a, sigma_a) in enumerate(zip(self.sigma2_a_range, self.sigma_a_range)):
                n_a = np.random.normal(0, sigma_a, size=L_a)
                La = 0.5 * sigma2_a * c_ + n_a
                _, Lc = self.block1.softdecode(La, None, code_algorithm)
                I_a[i] = I_a[i] + self.calc_MI[self.algorithm](c, La, bins=L_a)
                I_e[i] = I_e[i] + self.calc_MI[self.algorithm](c, Lc, bins=L_a)

        I_a = I_a / self.reps
        I_e = I_e / self.reps

        return I_a, I_e

    def EXIT_sys(self, info_length, SNR_dB, algorithm='default', block=2):
        """
        uses apriori LLR
        """
        if block == 1:
            if self.mode in ['bicm', 'serial']:
                raise TypeError("outer block should not take apriori LLR")
            else:
                code = self.block1
        elif block == 2:
            code = self.block2
        else:
            raise ValueError("block needs to be specified as 1 or 2")

        I_a = np.zeros(self.L)
        I_e = np.zeros(self.L)
        sigma2_N = 10 ** (- SNR_dB / 10)
        sigma_N = np.sqrt(sigma2_N)
        Lch = 4 / sigma2_N

        for _ in range(self.reps):
            u = np.random.randint(2, size=info_length)
            c = code.encode(u)
            x = 1 - 2 * c
            u_ = 1 - 2 * u
            w = np.random.normal(0, sigma_N, size=len(x))
            Lin = Lch * (x + w)
            Lin_sys = np.reshape(Lin, (-1, code.n))[:, :code.k].ravel()

            for i, (sigma2_a, sigma_a) in enumerate(zip(self.sigma2_a_range, self.sigma_a_range)):
                n_a = np.random.normal(0, sigma_a, size=info_length)
                La = 0.5 * sigma2_a * u_ + n_a
                Lu, _ = code.softdecode(Lin, La, algorithm)
                Le = Lu - Lin_sys - La
                I_a[i] = I_a[i] + self.calc_MI[self.algorithm](u, La, bins=info_length)
                I_e[i] = I_e[i] + self.calc_MI[self.algorithm](u, Le, bins=info_length)

        I_a = I_a / self.reps
        I_e = I_e / self.reps

        return I_a, I_e

    def chart(self, SNR_dB, info_length, info_length2=None, algorithm1='default', algorithm2='default',
              iterations=1000):
        """
        """
        if info_length2 is None:
            info_length2 = info_length

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.set_title(f"EXIT chart @ {SNR_dB} dB for {self.mode} mode")

        if self.mode == 'code':
            I_a, I_e = self.EXIT_sys(info_length, SNR_dB, algorithm1, block=1)
            ax1.plot(I_a, I_e, label=f'{self.block1.type}')
            ax1.set_xlabel("I_a")
            ax1.set_ylabel("I_e")
            ax1.legend()
            ax1.grid(True)
            plt.show()
            return

        elif self.mode == 'ldpc' and self.block2 is None:
            I_a, I_e = self.EXIT_ldpc(SNR_dB, iterations, algorithm1)
            ax1.plot(I_a, I_e, label=f'{self.block1.type}')
            ax1.set_xlabel("I_a")
            ax1.set_ylabel("I_e")
            ax1.legend()
            ax1.grid(True)
            plt.show()
            return

        elif self.mode == 'ldpc' and self.block2 is not None:
            I_a2, I_e2 = self.EXIT_sys(info_length2, SNR_dB, algorithm2, block=2)
            I_a1, I_e1 = self.EXIT_ldpc_serial(iterations, algorithm1)

        elif self.mode == 'parallel':
            I_a2, I_e2 = self.EXIT_sys(info_length2, SNR_dB, algorithm1, block=2)
            I_a1, I_e1 = self.EXIT_sys(info_length, SNR_dB, algorithm2, block=1)

        elif self.mode == 'serial':
            I_a2, I_e2 = self.EXIT_sys(info_length2, SNR_dB, algorithm2, block=2)
            I_a1, I_e1 = self.EXIT_nonsys(info_length, algorithm1)

        elif self.mode == 'bicm':
            I_a2, I_e2 = self.EXIT_map(info_length2, SNR_dB, algorithm2)
            I_a1, I_e1 = self.EXIT_nonsys(info_length, algorithm1)

        else:
            raise Exception("mode needs to be: 'code', 'parallel', 'serial', 'bicm', 'ldpc' or 'ldpc-serial'")

        try:
            ax1.plot(I_a1, I_e1, label=f'{self.block1.type}')
            ax1.plot(I_e2, I_a2, label=f'{self.block2.type}')
            ax1.set_xlabel("I_a1 = I_e2")
            ax1.set_ylabel("I_e1 = I_a2")
            ax1.legend()
            ax1.grid(True)
            plt.show()

        except Exception:
            raise Exception("error occured")


################################################################################################
if __name__ == "__main__":
    from ConvolutionCodes.convcode import ConvCode
    from BlockCodes.spc import SPC

    EX = EXIT('parallel', ConvCode([[1, 1, 1], [1, 0, 1]], 1), SPC(3))
    EX.chart(SNR_dB=-5, info_length=3000)
