import ga_tree
import numpy as np

num_cycles = 10
num_individuals = 1000
data_in = {}
data_in['x'] = np.linspace(0, 2 * np.pi * 10, 100)
data_in['y'] = np.linspace(0, 2 * np.pi * 10, 100)
data_out = np.sin(data_in['x']) + np.sin(data_in['y'])
normalized_data_out = (data_out - min(data_out)) / (max(data_out) - min(data_out))


# Define operators
unary_operands = ['sin', 'cos']
binary_operands = ['+', '-', '*', '/']
terminal_operands = ["c"] + list(data_in.keys())

# test to initialize Population
delta = 0.24
c = [0.1, 0.4]
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
# Training
for i in range(num_cycles):
    pop.cycle('mse')
    best = pop.get_best_func()
    best_funcs.append(best)

print(best)
