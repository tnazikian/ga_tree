import ga_tree
import numpy as np

num_cycles = 10
num_individuals = 10
data_in = {}
data_in['x'] = np.linspace(0, 2 * np.pi * 10, 1000)
data_in['y'] = np.linspace(0, 2 * np.pi * 10, 1000)
data_out = np.sin(data_in['x']) + np.sin(data_in['y'])

# Define operators
unary_operands = ['sin', 'cos']
binary_operands = ['+', '-', '*', '/']
terminal_operands = ["c"] + list(data_in.keys())

# test to initialize Population
delta = 0.12
c = [0.1, 0.2]
c.append(1 - sum(c))
individuals = []
for i in range(num_individuals):
    a = ga_tree.Bin_tree(delta=delta, term=terminal_operands, unary=unary_operands, binary=binary_operands)
    a.generate_tree(c)
    print('depth=%d'%a.depth)
    individuals.append(a)
pop = ga_tree.Population(individuals, data_in, data_out)

best_funcs = []
# Training
for i in range(num_cycles):
    pop.cycle('other')
    best = pop.get_best_func()
    best_funcs.append(i)

print(best)
