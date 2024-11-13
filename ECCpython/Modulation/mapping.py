import numpy as np
from Tools.helperFuncs import de2bi
from Tools.helperFuncs import bi2de


class Mapping:
    def __init__(self, m=1, coding_type='gray'):
        """
        mapping class
        m
        coding_type
        """
        if coding_type not in ['gray', 'antigray', 'natural', 'random', 'QAM_gray', 'QAM_antigray']:
            raise ValueError("coding_type needs to be 'gray', 'antigray', 'random' or 'natural'")

        self.m = m
        self.coding_type = coding_type
        self.M = 2 ** m  # number of symbols eg: if m = 3 then M = 8

        mapDICT = {'gray': self.gray,
                   'natural': self.natural,
                   'antigray': self.antigray,
                   'random': self.random,
                   'QAM_gray': self.QAM_gray,
                   'QAM_antigray': self.QAM_antigray}

        self._C = mapDICT[self.coding_type]()
        self._D = {c: i for i, c in enumerate(self._C)}
        self._Cbin = de2bi(self._C, self.m)
        self._Cdict = {c_dec: c_bin for c_dec, c_bin in zip(self._C, self._Cbin)}
        self._sort = np.argsort(self._C)

        self.padded_bits = 0

    def _zero_pad(self, data):
        """
        """
        data_padded = data
        if np.size(data) % self.m != 0:
            self.padded_bits = self.m - np.size(data) % self.m
            zero_padding = np.zeros(self.padded_bits, dtype=int)
            data_padded = np.append(data, zero_padding)

        return data_padded

    def mapping(self, raw_data):
        """

        """
        raw_data = self._zero_pad(raw_data)
        raw_data = np.reshape(raw_data, (-1, self.m))
        dec = bi2de(raw_data)
        return np.array([self._Cbin[d] for d in dec])

    def demapping(self, coded_data):
        """
        """
        x = bi2de(np.reshape(coded_data, (-1, self.m)))
        u = [self._D[d] for d in x]

        if self.padded_bits > 0:
            u = u[:-self.padded_bits]

        return u

    def gray(self):
        """
        """
        gray_mapper = [[0], [1]]
        if self.m > 1:
            for _ in range(self.m - 1):
                flipped = np.flip(gray_mapper, axis=0)
                x1 = [np.append(z, 0) for z in gray_mapper]
                x2 = [np.append(z, 1) for z in flipped]
                gray_mapper = np.append(x1, x2, axis=0)
        return bi2de(gray_mapper, leftMSB=False)

    def natural(self):
        """

        """
        return np.arange(self.M)

    def antigray(self):
        """
        """
        def _antigray(len):
            if len == 2:
                antigray_bins = np.array([0, 1])
            else:
                antigray_bins = _antigray(len // 2)
                tmp1 = np.ones((antigray_bins.shape[0], 1), dtype=int)
                tmp0 = np.zeros((antigray_bins.shape[0], 1), dtype=int)
                tmp = np.concatenate((tmp0, tmp1), axis=1).reshape((-1, 1))
                antigray_bins = np.vstack((antigray_bins, 1 - antigray_bins)).reshape((tmp.shape[0], -1))
                antigray_bins = np.hstack((tmp, antigray_bins))
            return antigray_bins

        return bi2de(np.reshape(_antigray(self.M), (-1, self.m))).ravel()

    def random(self):
        perm = np.arange(self.M)
        np.random.shuffle(perm)
        return perm

    def QAM_gray(self):
        """
        """
        perm = np.array([[0, 2], [1, 3]])
        m_tmp = 2

        while m_tmp < self.m:
            upper_left = perm + 0 * 2 ** m_tmp
            upper_right = np.fliplr(perm) + 2 * 2 ** m_tmp
            lower_left = np.flipud(perm) + 1 * 2 ** m_tmp
            lower_right = np.flipud(np.fliplr(perm)) + 3 * 2 ** m_tmp
            upper = np.hstack((upper_left, upper_right))
            lower = np.hstack((lower_left, lower_right))
            perm = np.vstack((upper, lower))
            m_tmp += 2

        return perm.ravel()

    def QAM_antigray(self):
        """
        """
        QAM = {2: [3, 2, 0, 1],

               4: [7, 2, 12, 9, 5, 0, 14, 11, 10, 15, 1, 4, 8, 13, 3, 6],

               6: [37, 30, 2, 28, 46, 21, 39, 12, 18, 44, 40, 60, 53, 5, 49, 55, 15, 50, 6, 26, 59, 8, 43, 17, 11, 35,
                   22, 33, 31, 62, 56, 24, 63, 48, 0, 54, 32, 3, 34, 19, 16, 42, 9, 51, 25, 14, 58, 23, 61, 57, 13, 45,
                   52, 41, 36, 10, 20, 47, 4, 38, 27, 1, 29, 7]}

        try:
            return np.array(QAM[self.m])
        except KeyError:
            raise Exception(f'only m = 2, 4, or 6 are supported but {self.m} was entered')


if __name__ == "__main__":
    a = [1, 2, 3]
    print(a[:None])
