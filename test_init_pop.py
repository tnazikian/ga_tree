from numpy import *

import ga_tree
from ga_tree.helper import *

TEST = False  # Toggle if you want real numbers for coeffs
COEFF_MAX = 10

num_cycles = 1
num_individuals = 30
data_in = {}
data_in['x'] = np.linspace(0, 2 * np.pi * 30, 1000)
data_in['y'] = np.linspace(0, 2 * np.pi * 30, 1000)
data_out = np.sin(np.sin(data_in['x']) + np.cos(data_in['y'])) + np.random.normal(0, 1, size=1000)
# normalized_data_out = (data_out - min(data_out)) / (max(data_out) - min(data_out))

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

# fill until no invalid functions remain in population
std_x = {}
std_y = np.std(data_out)
mean_x = {}
mean_y = np.mean(data_out)
standardized_data = {}
standardized_y = (data_out - mean_y) / std_y
for col_name, col_values in data_in.items():
    std_x[col_name] = np.std(col_values)
    mean_x[col_name] = np.mean(col_values)
    standardized_data[col_name] = (col_values - mean_x[col_name]) / std_x[col_name]
indiv_scores = np.array([i.mse_fitness((np.asarray(i.predict(standardized_data))*std_y + mean_y), data_out) for i in individuals])
null_indices = np.where(indiv_scores==0)[0]
while len(null_indices) > 0:
    for i in null_indices:
        individuals[i] = ga_tree.Bin_tree(delta=delta, term=terminal_operands, unary=unary_operands, binary=binary_operands)
        individuals[i].generate_tree(c)
    indiv_scores = np.array([individuals[i].mse_fitness((np.asarray(individuals[i].predict(standardized_data))*std_y + mean_y), data_out) for i in null_indices])
    null_indices = [null_indices[i] for i in np.where(indiv_scores == 0)[0]]
indiv_scores = np.array([i.mse_fitness((np.asarray(i.predict(standardized_data))*std_y + mean_y), data_out) for i in individuals])

pop = ga_tree.Population(individuals, data_in, data_out)

# plot_tree(individuals[5])


# for each tree in initial population, create 10 copies and run them through a GA to determine optimal coefficients.
# #
pop.tweak_coeffs()
new_scores = [i.mse_fitness((np.asarray(i.predict(standardized_data))*std_y + mean_y), data_out) for i in pop.individuals]
delta = []
for i in range(len(new_scores)):
    try:
        delta.append(new_scores[i] - indiv_scores[i])
    except:
        delta.append(1)
x = delta ** 2


