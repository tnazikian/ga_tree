#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/28/19

Constructs a binary tree representing an analytic function. 
Representation consists of an array of connected binary nodes. 
"""
from .node import Node
import numpy as np
from numpy import *
from scipy.stats.stats import pearsonr

TEST = False  # Toggle if you want real numbers for coeffs
COEFF_MAX = 3
import copy


class Bin_tree:
    def __init__(self, delta, term, unary, binary, COEFF_MAX=3):
        self.root = Node("root")
        self.delta = delta
        self.num_nodes = 0
        self.node_list = []
        self.coeff_list = []
        self.terminal_operands = term
        self.unary_operands = unary
        self.binary_operands = binary

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
            #             node.coeff = np.random.uniform(coeff_max)
            val = np.random.choice(self.terminal_operands)
            # If a constant is chosen as a leaf node, then the c {index of node} is created
            if val == "c":
                node.value = None
                if TEST:
                    node.coeff = val + str(node.index)
                else:
                    node.coeff = np.random.uniform(COEFF_MAX)
                out = "constant"
            else:
                node.value = val
        # unary operation
        elif r > c[0] and (r - c[0]) < c[1]:
            node.init_left()
            node.num_children = 1
            node.op = np.random.choice(self.unary_operands)
            #             node.coeff = np.random.uniform(coeff_max)
            self.generate_tree(c, node.left)
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
                    node.coeff = eval(str(node.left.coeff) + node.op + str(node.right.coeff))
                out = "constant"
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
                        node.right.coeff = np.random.uniform(COEFF_MAX)
                    node.right.parent = node.parent
                    # parent node points to child of current node
                    # so that child can replace current node
                    if node.name != "root":
                        if node.name == "left":
                            node.parent.left = node.right
                        else:
                            node.parent.right = node.right
                    node.right.name = node.name
                    node.right.index = node.index
                else:
                    if TEST:
                        node.left.coeff = 'c' + str(node.index)
                    else:
                        node.left.coeff = np.random.uniform(COEFF_MAX)
                    node.left.parent = node.parent
                    if node.name != "root":
                        if node.name == "left":
                            node.parent.left = node.left
                        else:
                            node.parent.right = node.left
                    node.left.name = node.name
                    node.left.index = node.index

            # If operator is * or /, then combine the coefficients of both branches
            elif (left != "constant" and right != "constant") and (node.op == "*" or node.op == "/"):
                node.coeff = "c" + str(node.index)
                node.left.coeff = None
                node.right.coeff = None

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

    def traverse(self, node=None, string=""):
        if node is None:
            node = self.get_root()
        if node.num_children == 1:
            string += node.op
            if node.coeff == None:
                string += '(' + self.traverse(node.left) + ')'
            else:
                string += '(' + str(node.coeff) + '*' + self.traverse(node.left) + ')'
        # if binary
        elif node.num_children == 2:
            string += '(' + self.traverse(node.left)
            string += node.op
            string += self.traverse(node.right) + ')'

        # if leaf node
        else:
            if node.coeff == None:
                string += str(node.value)
            # if constant leaf node, will only have coefficient
            elif node.value == None:
                string += str(node.coeff)
            else:
                string += '(' + str(node.coeff) + '*' + str(node.value) + ')'
        return string

    def index_tree(self, node):
        """depth wise travels down tree and indexes each node. returns list of nodes in same order"""
        self.node_list.append(node)
        node.index = self.num_nodes
        self.num_nodes += 1
        for i in [node.left, node.right]:
            if i is not None:
                self.index_tree(i)

    def reorder_whole_tree(self):
        self.num_nodes = 0
        self.node_list = []
        self.index_tree(self.get_root())

    def mse_fitness(self, data, y):
        """Uses mean square error to generate a fitness score
        Use for tuning coeffs after functional form determined"""
        f = 0
        # find all columns that are in tree
        datacols = data.columns
        # Get subset of data with only these

        text = self.traverse()
        # Create a sorted list of variables that are present in function
        x = [(text.find(i), i) for i in datacols]
        # sort vars by order of appearance from left to right
        #         x.sort(key=lambda x: x[0])
        vars_in_equation = [i[1] for i in x]
        #         #create data subset
        #         subset = data[vars_in_equation]
        ind = 0
        for i, row in data.iterrows():
            text_c = text
            for var in vars_in_equation:
                text_c = text_c.replace(var, str(data[var][i]))
            try:
                y_pred = eval(text_c)
                (y[i] - y_pred) ** 2
            except:
                print(self.traverse())
                print(self.node_list)
                print(self.get_root().num_children)
                raise
            err = (y[i] - y_pred) ** 2
            f += err
        mean_error = f / len(data)
        self.fitness = 1000 * (1 / (1 + mean_error))
        return self.fitness

    def cor_fitness(self, data, y):
        """
        Uses F-test to generate fitness score
        Use for determining functional form
        """
        function = self.traverse()
        print(function)
        for k in range(100):
            data['c%d' % k] = 1.0
        func_output = eval(function, globals(), data)
        if isinstance(func_output, (np.float64, float64, float)):
            func_output = y * 0.0 + func_output
        func_output_fixed = np.nan_to_num(func_output)
        try:
            p = pearsonr(func_output_fixed, y)[0]
        except Exception:
            p = 0.0
        if np.isnan(p):
            p = 0.0
        self.fitness = np.abs(p)
        return self.fitness

    # def __deepcopy__(self, memo={}):


__all__ = ['Bin_tree']
