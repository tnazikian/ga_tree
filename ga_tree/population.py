#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/28/19

Takes in a list of individuals and dataset and generates 
a new population of at least 100 candidate functions for 
each evolutionary cycle. Each cycle consists of weighted random
resampling with replacement based on fitness scores, and a 
chromosome swapping and mutation phase.

"""
import copy, random
import numpy as np
from .helper import *
from .btree import Bin_tree

MATE = 0.6  # proportion of total population chosen to mate
MUTATE = 0.3  # proportion of offspring that mutate
INDIVIDUALS = 100  # minimum number of individuals in a population


class Population:
    def __init__(self, individuals, data, y):
        self.individuals = individuals
        if len(individuals) < INDIVIDUALS:
            self.pop_size = INDIVIDUALS
        else:
            self.pop_size = len(individuals)
        self.data = data
        self.y = y
        # Calculate fitness scores of initial population
        self.scores = [individual.cor_fitness(self.data, self.y) for individual in individuals]
        self.best_ind = np.where(self.scores == np.max(self.scores))[0][0]
        # used to compare best score of previous population to new population
        self.last_best_func = self.individuals[self.best_ind]

    def get_best_func(self):
        # Return current best candidate function
        return self.individuals[self.best_ind]

    def new_pop(self, scores):
        # Random weighted resampling of current population based on fitness
        sum1 = np.sum(scores)
        prob = scores / sum1
        new_pop = np.random.choice(self.individuals, self.pop_size, p=prob, replace=True)
        return new_pop

    def cycle(self, fitness_func="mse"):
        new_pop = self.new_pop(self.scores)
        # creates deep copies of chosen individuals to make new population
        self.individuals = [copy.deepcopy(individual) for individual in new_pop]
        # If mut_offspring is set to mutate only on offspring
        mut_offspring = False
        if mut_offspring:
            mutate_ind = self.mate()
        else:
            # mutate whole population
            self.mate()
            mutate_ind = range(self.pop_size)
        if len(mutate_ind) > 0:
            self.mutate(mutate_ind)
        if fitness_func == "mse":
            self.scores = [individual.mse_fitness(self.data, self.y) for individual in self.individuals]
        else:
            self.scores = [individual.cor_fitness(self.data, self.y) for individual in self.individuals]
        current_best_ind = np.where(self.scores == np.max(self.scores))[0][0]
        worst_score_ind = np.where(self.scores == np.min(self.scores))[0][0]
        # If best score of current pop < previous pop, replace worst performing
        # candidate with previous best candidate
        if fitness_func == "mse":
            last_score = self.last_best_func.mse_fitness(self.data, self.y)
        else:
            last_score = self.last_best_func.cor_fitness(self.data, self.y)
        if last_score > self.scores[current_best_ind]:
            self.scores[worst_score_ind] = last_score
            self.individuals[worst_score_ind] = self.last_best_func
            self.best_ind = worst_score_ind
        else:
            self.best_ind = current_best_ind
        self.last_best_func = self.individuals[self.best_ind]

    def fill(self, delta, terminal_operands, unary_operands, binary_operands, probs):
        """
        :param delta: Amount probs increases/decreases by per level of tree

        :param terminal_operands: constant, or variables

        :param unary_operands:

        :param binary_operands:

        :param probs: original probability [p(leaf), p(unary node), p(binary node)]

        If the population contains invalid functions (fitness score == 0) then they are replaced with valid functions
        """
        indiv_scores = np.array([i.mse_fitness(self.data, self.y) for i in self.individuals])
        null_indices = np.where(indiv_scores == 0)[0]       # Contains number of individuals that have invalid functions
        # Replace trees until null_indices is empty
        while len(null_indices) > 0:
            for i in null_indices:
                self.individuals[i] = Bin_tree(delta=delta, term=terminal_operands, unary=unary_operands,
                                                  binary=binary_operands)
                self.individuals[i].generate_tree(probs)
            indiv_scores = np.array([i.mse_fitness(self.data, self.y) for i in self.individuals])
            null_indices = np.where(indiv_scores == 0)[0]

    def tweak_coeffs(self):
        """
        Genetic algorithm tweaks the coefficients of each tree in the population.
        A subpopulation of num_copies_per_cycle is created for each tree in population for the GA to run on. Each tree
        in the subpop has randomly generated coefficients between -10 and 10. The subpop is run thru a GA that tweaks
        the coefficients and resamples the trees num_cycles number of times.
        If, at the end of num_cycles, the MSE of one of the modified trees in the subpop exceeds the original,
        it replaces the original tree in the population.
        """

        alpha = 1                       # Step size by which coefficients can be increased/decreased
        coeffs_tweaked = 3              # Number of coefficients per tree to be randomly selected for tweaking
        num_cycles= 100                 # Number of times each tree in pop is run through a genetic algorithm
        num_copies_per_cycle = 10       # Number of copies per subpopulation.
        for i, tree in enumerate(self.individuals):
            # make num_copies_per_cycle copies and run them through a GA
            copies = [tree.get_copy() for _ in range(num_copies_per_cycle)]
            orig_tree_fitness = tree.mse_fitness(self.data, self.y)
            copy_scores = [orig_tree_fitness]*num_copies_per_cycle

            # for each copy, assign random coefficients between -10 and 10
            for copy in copies[:-1]:
                for node in copy.coeff_list:
                    node.coeff = np.random.uniform(-10, 10)

            # resample copies num_cycles times
            for l in range(num_cycles):
                # Save the last best candidate in case all new trees are worse performing
                last_best_ind = np.where(copy_scores == np.max(copy_scores))[0][0]
                last_best = copies[last_best_ind].get_copy()
                last_best_score = last_best.fitness
                for copy in copies:
                    # randomly choose coeffs_tweaked number of coeffs per tree to modify
                    coeffs_to_tweak = np.random.choice(copy.coeff_list, coeffs_tweaked)
                    # add/subtract noise*alpha from chosen node coeffs
                    for c in coeffs_to_tweak:
                        c.coeff += np.random.choice([-1,1])*alpha

                # Generate new subpopulation with random weighted selection using MSE fitness score
                copy_scores = [t.mse_fitness(self.data, self.y) for t in copies]
                copies = np.append(copies, [last_best])
                copy_scores.append(last_best_score)
                copies = np.random.choice(copies, len(copies)-1, p=copy_scores / np.sum(copy_scores), replace=True)
                copy_scores = [i.fitness for i in copies]

                # If last best function has lower MSE than current best function, replace the worst performing candidate
                if last_best_score > max(copy_scores):
                    worst_ind = np.where(copy_scores == np.min(copy_scores))[0][0]
                    copies[worst_ind] = last_best
                    copy_scores[worst_ind] = last_best_score
            if orig_tree_fitness < copies[np.where(copy_scores == np.max(copy_scores))[0][0]].fitness:
                self.individuals[i] = copies[np.where(copy_scores == np.max(copy_scores))[0][0]]

    def mate(self):
        num_to_mate = int(len(self.individuals) * MATE)
        if num_to_mate >= 2:
            if num_to_mate % 2 == 1:
                num_to_mate += 1  # Makes sure there is even number of parents
            mating_ind = np.random.choice(len(self.individuals), num_to_mate, replace=False)
            for i in range(len(mating_ind) // 2):
                swap_subtrees(self.individuals[2 * i], self.individuals[2 * i + 1])
            return mating_ind
        return None

    def mutate(self, mutate_ind):
        num_to_mutate = int(len(mutate_ind) * MUTATE)
        mutate_ind = np.random.choice(mutate_ind, num_to_mutate, replace=False)
        if len(mutate_ind) > 0:
            for i in mutate_ind:
                mut_choice = np.random.choice(self.individuals[i].node_list)  # randomly choose a node to mutate
                self.individuals[i].mutate_node(mut_choice)

    # @staticmethod
    # def generate_individuals(n, num_vars, coeff_min, coeff_max, exp_min, exp_max, operators):
    #     """
    #     Static method for generating initial population of size n from
    #     a dataset with num_vars variables. Coefficient and exponential values of
    #     chromosomes are randomly generated within a user-specified range. operators argument
    #     contains list of strings that represent operators that can be used e.g. ['+', '-', '*'].
    #     """
    #     indiv = []
    #     if num_vars > 1 and len(operators) == 0:
    #         raise ValueError("no operators")
    #     for i in range(n):
    #         l = [np.random.uniform(low=coeff_min, high=coeff_max) for _ in range(num_vars)]
    #         a = [np.random.randint(low=exp_min, high=exp_max) for _ in range(num_vars)]
    #         m = np.random.choice(operators, len(l)-1)
    #         indiv.append(Individual(l, a, m))
    #     return indiv


__all__ = ['Population']
