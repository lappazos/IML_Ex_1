import numpy as np
import matplotlib.pyplot as plt

data = np.random.binomial(1, 0.25, (100000, 1000))

# part_a
plot = data[:5, :]
plot = np.cumsum(plot, axis=1)
indexes = np.arange(1, 1001)
plot = plot / indexes
plt.figure()
for i in range(5):
    plt.plot(plot[i], alpha=0.75)
plt.xlabel('m')
plt.ylabel('P')

# part_b
epsilon = np.array([0.5, 0.25, 0.1, 0.01, 0.001])
# part_c
bias = np.cumsum(data, axis=1)
bias = bias / indexes
bias = np.abs(bias - 0.25)
# part b+c
for eps in epsilon:
    plt.figure()
    plt.title("epsilon is " + str(eps))
    chev = 1 / (4 * indexes * np.square(eps))
    chev[chev > 1] = 1
    hoff = 2 * np.exp(-2 * indexes * np.square(eps))
    hoff[hoff > 1] = 1
    plt.plot(chev, label='Chev')
    plt.plot(hoff, label='Hoff')
    plt.xlabel('m')
    plt.ylabel('P')
    # part_c
    satisfy = np.where(bias > eps, 1, 0)
    satisfy_avg = satisfy.mean(axis=0)
    plt.plot(satisfy_avg, label="satisfy percentage")
plt.show()

