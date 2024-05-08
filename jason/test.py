from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

# <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
# 0       3       -1              0       2       0       3       0       0       1       1               0       8cf07265        ae46a29d        c81688bb        f922efad        25c83c98        13718bbd ad9fa255        0b153874        a73ee510        5282c137        e5d8af57        66a76a26        f06c53ac        1adce6ef        8ff4b403        01adbab4        1e88c74f        26b3c7a721c9516a         32c7478e        b34f3128


def load_data(filename):
    n = 0
    counters = [Counter() for _ in range(26)]

    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            n += 1
            # label = int(line[0])
            # print(line)
            # features = [int(x) for x in line[1:14]]
            categories = line[14:]
            for i, c in enumerate(categories):
                counters[i][c] += 1

            if n % 1000000 == 0:
                # 45000000 [1460, 583, 9961921, 2170414, 305, 24, 12511, 633, 3, 92732, 5678, 8208561, 3192, 27, 14962, 5372601, 10, 5642, 2173, 4, 6927017, 18, 15, 283431, 105, 141625]
                print(n, [len(c) for c in counters])
                for i, c in enumerate(counters):
                    plot_counter_distribution(c, "cat_{}".format(i))


def plot_counter_distribution(counter, name):
    v = np.array(sorted(list(counter.values()), key=lambda x: -x))
    plt.plot(v)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.savefig("fig/{}.png".format(name), dpi=300, bbox_inches="tight")
    plt.xscale("log")
    plt.savefig("fig/{}_log.png".format(name), dpi=300, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    load_data("/disk/train.txt")
