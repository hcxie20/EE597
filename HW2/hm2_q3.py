import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv


def cal_mc1(ls):
    # p00 p01 p10 p11
    p = [0] * 4 
    nums = [0, 0]
    for i in range(len(ls) - 1):
        if ls[i] == 0:
            nums[0] += 1
            if ls[i+1] == 0:
                p[0] += 1
            else:
                p[1] += 1
        else:
            nums[1] += 1
            if ls[i+1] == 0:
                p[2] += 1
            else:
                p[3] += 1
    p[0] /= nums[0]
    p[1] /= nums[0]
    p[2] /= nums[1]
    p[3] /= nums[1]
    return p

def f3():
    speeds = [0, 5, 10, 15, 20, 25]
    powers = [[]] * len(speeds)
    states = powers[:]
    ps = states[:]
    x = list(np.linspace(1, 2000, 2000))

    plt.figure()
    for i in range(len(speeds)):
        with open("./Q3_{}.csv".format(speeds[i])) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                powers[i] = row
        f.close()
        states[i] = [sep(float(x)) for x in powers[i]]
        ps[i] = cal_mc1(states[i])

        plt.subplot(len(speeds), 1, i+1)
        plt.plot(x, states[i], label="speed = {}".format(speeds[i]))
        plt.legend()
        plt.xlim(1, 50)
    plt.savefig("./Q3_result.jpg")
    with open("./Q3_result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Speed", "P00", "P01", "P10", "P11"])
        for i in range(len(speeds)):
            writer.writerow([str(speeds[i])] + [ str(ps[i][x]) for x in range(4)])
    f.close()
    with open("./Q3_result_table.csv", "w") as f:
        writer = csv.writer(f)
        for i in range(len(speeds)):
            writer.writerow(["|{0}|{1:.4f}|{2:.4f}|{3:.4f}|{4:.4f}|".format(speeds[i], ps[i][0], ps[i][1], ps[i][2], ps[i][3])])
    f.close()
    pass

def sep(x):
    if x >= 3.23:
        return True
    else:
        return False

if __name__ == "__main__":
    f3()