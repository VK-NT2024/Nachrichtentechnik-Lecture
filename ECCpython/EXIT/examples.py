from EXIT.exit import EXIT


def parallel_SPC(k, SNR_dB):
    from BlockCodes.spc import SPC
    spc = SPC(k)
    info_length = spc.k * 100
    EXIT(spc, spc, 'parallel').chart(SNR_dB, info_length)


def serial_SPC(k_outer, SNR_dB):
    from BlockCodes.spc import SPC
    spc_1 = SPC(k_outer)
    spc_2 = SPC(k_outer + 1)
    info_length = spc_1.k * 100
    info_length2 = spc_2.k * 100
    EXIT(spc_1, spc_2, 'serial').chart(SNR_dB, info_length, info_length2)


def parallel_RSC_SPC(G, k, SNR_dB):
    from ConvolutionCodes.convcode import ConvCode
    from BlockCodes.spc import SPC
    rsc = ConvCode(G, 1)
    spc = SPC(k)
    info_length = spc.k * 100
    EXIT(rsc, spc, 'parallel').chart(SNR_dB, info_length)


def serial_REP_RSC(n, G, SNR_dB):
    from BlockCodes.repetition import Repetition
    from ConvolutionCodes.convcode import ConvCode
    rep = Repetition(n)
    rsc = ConvCode(G, 1)
    info_length = 300
    info_length2 = 300 * n
    EXIT(rep, rsc, 'serial').chart(SNR_dB, info_length, info_length2)


def QAM_RSC(G, m, coding, SNR_dB):
    from Mapping.modulation import Modulation
    from ConvolutionCodes.convcode import ConvCode
    mod = Modulation(m, coding, 'QAM')
    rsc = ConvCode(G, 1)
    info_length = 1000
    EXIT(rsc, mod, 'bicm').chart(SNR_dB, info_length)


def QAM_SPC(k, m, coding, SNR_dB):
    from Mapping.modulation import Modulation
    from BlockCodes.spc import SPC
    mod = Modulation(m, coding, 'QAM')
    spc = SPC(k)
    info_length = spc.k * 100
    EXIT(spc, mod, 'bicm').chart(SNR_dB, info_length)


