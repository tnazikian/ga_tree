#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/28/19

Constructs a binary tree representing an analytic function. 
Representation consists of an array of connected binary nodes. 
"""
import warnings
from .node import Node
import numpy as np
from numpy import *
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse

TEST = False  # Toggle if you want real numbers for coeffs
COEFF_MAX = 10
import copy


class Bin_tree:
    def __init__(self, delta, term, unary, binary, COEFF_MAX=10):
        self.root = Node("root", parent=None)
        self.delta = delta
        self.num_nodes = 0
        self.node_list = []
        self.coeff_list = []
        self.terminal_operands = term
        self.unary_operands = unary
        self.binary_operands = binary
        self.fitness = None


    def get_copy(self):
        return copy.deepcopy(self)

    def generate_tree(self, c, node=None):
        """
        generates tree recursively from a starter node

        :param c: array containing probabilities: [P(0 children), P(Unary), P(Binary)]

        :param node: node object (if None use the tree root)
        
        :return out: String containing either None or "constant" - used to check if subtree is a root node with a constant
        """
        if node is None:
            node = self.get_root()
        node.probs = c

        out = None  # variable that tells whether node was constant or variable
        self.num_nodes += 1
        node.index = self.num_nodes
        self.node_list.append(node)
        r = np.random.rand()
        # decrease p(double branch) while
        # increase p(no branch)
        c = copy.deepcopy(c)
        c[0] += self.delta
        c[1] -= self.delta / 2
        c[2] -= self.delta / 2
        # no operands
        if r < c[0]:
            node.num_children = 0
            val = np.random.choice(self.terminal_operands)
            # If a constant is chosen as a leaf node, then the c {index of node} is created
            if val == "c":
                node.value = None
                if TEST:
                    node.coeff = val + str(node.index)
                else:
                    # node.coeff = np.random.uniform(COEFF_MAX)
                    node.coeff = 1
                out = "constant"
            else:
                node.value = val
                node.coeff = 1
        # unary operation
        elif r > c[0] and (r - c[0]) < c[1]:
            node.init_left()
            node.num_children = 1
            node.op = np.random.choice(self.unary_operands)
            self.generate_tree(c, node.left)
            node.coeff = 1
        # binary
        else:
            node.init_left()
            node.init_right()
            node.num_children = 2
            node.op = np.random.choice(self.binary_operands)
            left = self.generate_tree(c, node.left)
            right = self.generate_tree(c, node.right)
            # If the left and right children are constants, then convert current node into leaf node
            if left == "constant" and right == "constant":
                # erase children and assign evaluated value
                node.value = None
                if TEST:
                    node.coeff = "c" + str(node.index)
                else:
                    # node.coeff = eval(str(node.left.coeff) + node.op + str(node.right.coeff))
                    node.coeff = 1
                    # node.coeff = np.random.uniform(COEFF_MAX)
                out = "constant"
                self.node_list.remove(node.left)
                self.node_list.remove(node.right)
                node.left = None
                node.right = None
                node.num_children = 0

            # if one branch is constant and operator is '*', store constant
            # as attribute of other branch, and replace current node with
            # the child node.
            elif (left == "constant" or right == "constant") and (node.op == "*" or node.op == "/"):
                if left == "constant":
                    if TEST:
                        node.right.coeff = 'c' + str(node.index)
                    else:
                        # node.right.coeff = np.random.uniform(COEFF_MAX)
                        node.right.coeff = 1
                    node.right.parent = node.parent
                    # parent node points to child of current node
                    # so that child can replace current node
                    # if node.name != "root":
                    if node.name == "left":
                        node.parent.left = node.right
                    elif node.name == "right":
                        node.parent.right = node.right
                    node.right.depth -= 1
                    node.right.name = node.name
                    node.right.index = node.index
                    self.node_list.remove(node.left)
                    if self.get_root() == node:
                        self.root = node.left
                elif right == "constant":
                    if TEST:
                        node.left.coeff = 'c' + str(node.index)
                    else:
                        # node.left.coeff = np.random.uniform(COEFF_MAX)
                        node.left.coeff = 1
                    node.left.parent = node.parent
                    if node.name == "left":
                        node.parent.left = node.left
                    elif node.name == "right":
                        node.parent.right = node.left
                    node.left.depth -= 1
                    node.left.name = node.name
                    node.left.index = node.index
                    self.node_list.remove(node.right)
                    if self.get_root() == node:
                        self.root = node.right
                self.node_list.remove(node)


            # If operator is * or /, then combine the coefficients of both branches
            elif (left != "constant" and right != "constant") and (node.op == "*" or node.op == "/"):
                if node.left.coeff is not None or node.right.coeff is not None:
                    # node.coeff = "c" + str(node.index)
                    node.coeff = 1
                    # node.coeff = np.random.uniform(COEFF_MAX)
                    node.left.coeff = None
                    node.right.coeff = None
            if node.coeff is None:
                node.coeff = 1

        self.reorder_whole_tree()
        return out

    def get_root(self):
        return self.root

    def __repr__(self):
        return self.traverse(self.get_root())

    def print_tree(self):
        pass

    def del_subtree(self, node):
        if node.name == "left":
            node.parent.left = None
        elif node.name == "right":
            node.parent.right = None
        self.reorder_whole_tree()

    def mutate_node(self, n):
        """takes in either a node or node index, and constructs
        a new tree from that point."""
        if isinstance(n, Node):
            self.generate_tree(n.probs, n)
        elif isinstance(n, int):
            self.generate_tree(self.node_list[n].probs, self.node_list[n])
        self.reorder_whole_tree()

    def traverse(self, node=None):
        if node is None:
            node = self.get_root()
        return node.traverse()

    def index_tree(self, node, ind):
        """depth wise travels down tree and indexes each node. returns list of nodes in same order"""
        self.node_list.append(node)
        node.depth = ind
        node.index = self.num_nodes
        self.num_nodes += 1
        if node.coeff is not None:
            self.coeff_list.append(node)
        for ind, i in enumerate([node.left, node.right]):
            if i is not None:
                self.index_tree(i, ind+1)
                i.parent = node
                if ind==0:
                    node.name = "left"
                else:
                    node.name = "right"

    def reorder_whole_tree(self):
        self.num_nodes = 0
        self.node_list = []
        self.coeff_list = []
        self.index_tree(self.get_root(), 0)
        self.get_root().name = "root"

    def get_tree_cost(self):
        # Get cost of tree based on complexity
        root = self.get_root()
        return root.get_cost()

    def predict(self, data):
        """takes in data and returns y predicted"""
        function = self.traverse()
        # for k in range(100):
        #     data['c%d' % k] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                func_output = eval(function, globals(), data)
            except ZeroDivisionError:
                return False
        return func_output

    def mse_fitness(self, func_output, y):
        """Uses mean square error to generate a fitness score
        Use for tuning coeffs after functional form determined"""

        if func_output is False:
            self.fitness = 0
            return self.fitness
        # func_output = self.predict(data)
        # function = self.traverse()
        # for k in range(100):
        #     data['c%d' % k] = 1.0
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     try:
        #         func_output = eval(function, globals(), data)
        #     except ZeroDivisionError:
        #         self.fitness = 0 # If zero in denom. ignore tree in next cycle
        #         return self.fitness

        if isinstance(func_output, (np.float64, float64, float, int, np.int32)):
            func_output = y * 0.0 + func_output
        func_output_fixed = np.nan_to_num(func_output)
        # normalized_func_output = (func_output_fixed - min(func_output_fixed)) / (
        #             max(func_output_fixed) - min(func_output_fixed))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                p = mse(func_output_fixed, y)
                #if normalize
                # p = mse(normalized_func_output, y)
        except Exception as _excp:
            self.fitness = 0  # If zero in denom. ignore tree in next cycle
            return self.fitness
        if np.isnan(p):
            self.fitness = 0  # If zero in denom. ignore tree in next cycle
            return self.fitness
        self.fitness = 1000 * (1 / (1 + p))
        return self.fitness

    def cor_fitness(self, data, y):
        """
        Uses F-test to generate fitness score
        Use for determining functional form
        """
        function = self.traverse()
        for k in range(100):
            data['c%d' % k] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                func_output = eval(function, globals(), data)
            except ZeroDivisionError:
                self.fitness = 0 # If zero in denom. ignore tree in next cycle
                return self.fitness
        if isinstance(func_output, (np.float64, float64, float, int, np.int32)):
            func_output = y * 0.0 + func_output
        func_output_fixed = np.nan_to_num(func_output)
        # normalized_func_output = (func_output_fixed - min(func_output_fixed)) / (max(func_output_fixed) - min(func_output_fixed))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                p = pearsonr(func_output_fixed, y)[0]
                # if normalize
                # p = pearsonr(normalized_func_output, y)[0]
        except Exception as _excp:
            p = 0.0
        if np.isnan(p):
            p = 0.0
        self.fitness = np.abs(p)

        asdf = self.get_tree_cost()
        return self.fitness/self.get_tree_cost()

    @property
    def depth(self):
        return max([node.depth for node in self.node_list])

__all__ = ['Bin_tree']
