import pandas as pd
import argparse, pickle
import ga_tree
from tqdm import tqdm
from pprint import pprint
import sys
import numpy as np

print(sys.argv)

parser = argparse.ArgumentParser()
parser.add_argument('n_individuals', type=int)
parser.add_argument('n_cycles', type=int)
parser.add_argument('datapath')
# parser.add_argument("-cmin", type=int, default=-10)
# parser.add_argument("-cmax", type=int,default=10)
# parser.add_argument("-emin", type=int,default=-1)
# parser.add_argument("-emax", type=int,default=5)
# parser.add_argument("-operators", '--list', type=str, default="+,-,*,/")
args = parser.parse_args()

pprint(args)

num_cycles = args.n_cycles
num_individuals = args.n_individuals

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
    individuals.append(a)
pop = ga_tree.Population(individuals, data_in, data_out)

best_funcs = []
# Training
for i in range(num_cycles):
    pop.cycle('other')
    best = pop.get_best_func()
    best_funcs.append(i)

print(best)
