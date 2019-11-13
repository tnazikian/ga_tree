import ga_tree
import numpy as np
import matplotlib.pyplot as plt
from ttictoc import TicToc

import warnings
import numpy as np
from numpy import *
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

TEST = False  # Toggle if you want real numbers for coeffs
COEFF_MAX = 10
import copy


num_cycles = 5
num_individuals = 3000
data_in = {}
data_in['x'] = np.linspace(0, 2 * np.pi * 30, 1000)
data_in['y'] = np.linspace(0, 2 * np.pi * 30, 1000)
data_out = np.sin(np.sin(data_in['x']) + np.cos(data_in['y'])) + np.random.normal(0, 1, size=1000)
normalized_data_out = (data_out - min(data_out)) / (max(data_out) - min(data_out))

# Define operators
unary_operands = ['sin', 'cos']
binary_operands = ['+', '-', '*', '/']
terminal_operands = ["c"] + list(data_in.keys())

# test to initialize Population
delta = 0.04
c = [0.1, 0.3]
c.append(1 - sum(c))
individuals = []
for i in range(num_individuals):
    a = ga_tree.Bin_tree(delta=delta, term=terminal_operands, unary=unary_operands, binary=binary_operands)
    a.generate_tree(c)
    print('depth=%d'%a.depth)
    individuals.append(a)
pop = ga_tree.Population(individuals, data_in, data_out)
# if normalize dataset
# pop = ga_tree.Population(individuals, data_in, normalized_data_out)


best_funcs = []
max_scores = []
# Training
t = TicToc() ## TicToc("name")
t.tic()
for i in range(num_cycles):
    pop.cycle('mse')
    best = pop.get_best_func()
    best_funcs.append(best)
    max_scores.append(best.fitness)
t.toc()
print(t.elapsed)
time_elapsed = t.elapsed
#
# plt.plot([i for i in range(num_cycles)], max_scores)
# plt.xlabel("cycles")
# plt.ylabel("max score")
# plt.title("learning curve")
#
#
#
# plt.plot(data_in['x'][:100], data_out[:100])
# plt.xlabel("x")
# plt.ylabel("out")
# plt.title("data")
# plt.show()
#

# Creates two subplots and unpacks the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([i for i in range(num_cycles)], max_scores)
ax1.set_title('learning curve vs cycles')
ax2.plot(data_in['x'][:100], data_out[:100])
ax2.plot(data_in['x'][:100], eval(best.traverse(), globals(), data_in)[:100])
ax2.set_title("truth (with noise) vs estimate")
ax2.legend(["{}".format("sin(sin(x)+cos(y))"), "{}".format(best.traverse())])

plt.show()

print(best)
