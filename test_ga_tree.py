import pandas as pd
import argparse, pickle
from ga_tree import Bin_tree, Population
from tqdm import tqdm
from pprint import pprint
import sys

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

#     f = open(args.datapath, 'rb')
#     data = pickle.load(f)

data = pd.read_pickle(args.datapath)
num_cycles = args.n_cycles
num_individuals = args.n_individuals
x = data.loc[:, data.columns[:-1]]
y = data.loc[:, data.columns[-1]]
#     f.close()

# operators = [i for i in args.list.split(',')]

# Define operators
unary_operands = ['sin', 'cos']
binary_operands = ['+', '-', '*', '/']
terminal_operands = ["c"]
for i in data.columns[:-1]:
    terminal_operands.append(i)

#test to initialize Population
c=[0.1, 0.2]
c.append(1-sum(c))
individuals=[]
for i in range(num_individuals):
    a=Bin_tree(delta=0.12, term=terminal_operands, unary=unary_operands, binary=binary_operands)
    a.generate_tree(a.get_root(), c)
    individuals.append(a)
pop = Population(individuals, x, y)

fitnesses = []      # track best fitness score out of population
best_funcs = []
# Training
for i in tqdm(range(num_cycles)):
    pop.cycle()
    best = pop.get_best_func()
    fitnesses.append(best.mse_fitness(x, y))
    best_funcs.append(i)

print(best)
