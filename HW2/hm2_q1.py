import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


# in dBm or dB:
PT = 30
ETA = 2.5
SIGMA = 3 
NOISE = -100
D0 = 1
SNR = 10

Pout = SNR + NOISE
K = -30 + ETA *10 * math.log10(1/D0)

def Q_function(x):
    return 1 - norm.cdf(x)


def Poutage(_sigma, _p_out, _p_r):
    return 1 - Q_function((_p_out - _p_r)/_sigma)

def Pr(d, _p_t, _k, _eta, _d0):
    return _p_t + _k - _eta * 10 * math.log10(d/_d0)

def Pr_d_dB(d, _p_t, _k, _eta, _d0):
    return _p_t + _k - _eta * d

def f1():
    x_range = [1, 10000]
    x = list(np.linspace(x_range[0], x_range[1], 10000))
    x_log = [10 * math.log10(item) for item in x]
    y = [Poutage(SIGMA, Pout, Pr(item, PT, K, ETA, D0)) for item in x]

    sigmas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    y1 = [[]]*len(sigmas)
    etas = [2, 2.5, 3, 3.5, 4, 4.5, 5]
    y2 = [[]]*len(etas)
    y0 = [[]]*len(etas)

    # log
    plt.figure(figsize=[15, 5])
    plt.xlim(0, 40)
    for i in range(len(etas)):
        y0[i] = [Pr_d_dB(item, PT, K, etas[i], D0) for item in x_log]
        plt.plot(x_log, y0[i], label="$\eta = {}$".format(etas[i]))
    plt.hlines(-90, 0, x_range[1], linestyles="dashed")
    plt.xlabel("d/dB")
    plt.ylabel("$P_r$/dBm")
    plt.legend()
    plt.savefig("./Q1_result3.jpg")

    plt.figure(figsize=[15, 8])
    plt.subplot(2, 1, 1)
    plt.ylim(0, 1)
    plt.xlim(0, 40)
    for i in range(len(sigmas)):
        y1[i] = [Poutage(sigmas[i], Pout, Pr_d_dB(item, PT, K, ETA, D0)) for item in x_log]
        plt.plot(x_log, y1[i], label="$\sigma = {}$".format(sigmas[i]))
    plt.xlabel("d/dB")
    plt.ylabel("$P_{outage}$/dBm")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    plt.xlim(0, 40)
    for i in range(len(etas)):
        y2[i] = [Poutage(SIGMA, Pout, Pr_d_dB(item, PT, K, etas[i], D0)) for item in x_log]
        plt.plot(x_log, y2[i], label="$\eta = {}$".format(etas[i]))
    plt.legend()
    plt.xlabel("d/dB")
    plt.ylabel("$P_{outage}$/dBm")
    plt.savefig("./Q1_result4.jpg")


    plt.figure(figsize=[15, 5])
    for i in range(len(etas)):
        y0[i] = [Pr(item, PT, K, etas[i], D0) for item in x]
        plt.plot(x, y0[i], label="$\eta = {}$".format(etas[i]))
    plt.hlines(-90, 0, x_range[1], linestyles="dashed")
    plt.legend()
    plt.xlabel("d/m")
    plt.ylabel("$P_r$/dBm")
    plt.savefig("./Q1_result1.jpg")

    plt.figure(figsize=[15, 8])
    plt.subplot(2, 1, 1)
    plt.ylim(0, 1)
    plt.xlim(0, x_range[1])

    for i in range(len(sigmas)):
        y1[i] = [Poutage(sigmas[i], Pout, Pr(item, PT, K, ETA, D0)) for item in x]
        plt.plot(x, y1[i], label="$\sigma = {}$".format(sigmas[i]))
    plt.legend()
    plt.xlabel("d/m")
    plt.ylabel("$P_{outage}$/dBm")


    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    plt.xlim(0, x_range[1])

    for i in range(len(etas)):
        y2[i] = [Poutage(SIGMA, Pout, Pr(item, PT, K, etas[i], D0)) for item in x]
        plt.plot(x, y2[i], label="$\eta = {}$".format(etas[i]))
    plt.legend()
    plt.xlabel("d/m")
    plt.ylabel("$P_{outage}$/dBm")
    plt.savefig("./Q1_result2.jpg")

    pass

def f2():
    
    pass

if __name__ == "__main__":
    f1()
    pass