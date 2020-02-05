import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

PT = 23
N = -90
D0 = 1
K = -10 - PT

def cal_pr(d, _pt, _k, _eta, _d0):
    return _pt + _k - _eta * 10 * math.log10(d/_d0)

def cal_pr_d_in_dB(d, _pt, _k, _eta, _d0):
    return _pt + _k - _eta * (d - 10 * math.log10(_d0))

def cal_SNR(_pr, _n):
    return _pr - _n

def cal_goodput(_snr):
    if _snr < 2.5:
        return 0
    elif _snr < 6:
        return 6
    elif _snr < 8:
        return 11
    elif _snr < 9:
        return 16
    elif _snr < 16:
        return 20
    elif _snr < 21:
        return 28
    elif _snr < 22:
        return 34
    else:
        return 37


def f2():
    etas = [2, 4]
    x = list(np.linspace(1, 1000, 1000))
    x_log = [10 * math.log10(d) for d in x]
    pr = [None] * len(etas)
    SNR = pr[:]
    goodput = pr[:]

    plt.figure(figsize=[15, 8])
    sub1 = plt.subplot(3, 1, 1)
    sub2 = plt.subplot(3, 1, 2)
    sub3 = plt.subplot(3, 1, 3)
    for i in range(len(etas)):
        pr[i] = [cal_pr(d, PT, K, etas[i], D0) for d in x]
        sub1.plot(x, pr[i], label="$\eta$={}".format(etas[i]))

        SNR[i] = [cal_SNR(p, N) for p in pr[i]]
        sub2.plot(x, SNR[i], label="$\eta$={}".format(etas[i]))

        goodput[i] = [cal_goodput(snr) for snr in SNR[i]]
        sub3.plot(x, goodput[i], label="$\eta$={}".format(etas[i]))
    pass
    sub1.ylim = [-130, 0]
    sub1.legend()
    sub2.ylim = [-30, 80]
    sub2.legend()
    sub3.legend()
    plt.savefig("./Q2_result1.jpg")

    plt.figure(figsize=[15, 8])
    sub1 = plt.subplot(3, 1, 1)
    sub2 = plt.subplot(3, 1, 2)
    sub3 = plt.subplot(3, 1, 3)
    for i in range(len(etas)):
        pr[i] = [cal_pr_d_in_dB(d, PT, K, etas[i], D0) for d in x_log]
        sub1.plot(x_log, pr[i], label="$\eta$={}".format(etas[i]))

        SNR[i] = [cal_SNR(p, N) for p in pr[i]]
        sub2.plot(x_log, SNR[i], label="$\eta$={}".format(etas[i]))

        goodput[i] = [cal_goodput(snr) for snr in SNR[i]]
        sub3.plot(x_log, goodput[i], label="$\eta$={}".format(etas[i]))
    pass
    sub1.ylim = [-130, 0]
    sub1.legend()
    sub2.ylim = [-30, 80]
    sub2.legend()
    sub3.legend()
    plt.savefig("./Q2_result2.jpg")




if __name__ == "__main__":
    f2()
    pass