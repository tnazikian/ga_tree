#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/28/19

Helper methods for the GA project
"""
from graphviz import Digraph
import numpy as np
import copy


def swap_subtrees(tree1, tree2, debug=False):
    """randomly selects two subtrees and swaps them"""
    choice1 = np.random.choice(tree1.node_list)
    choice2 = np.random.choice(tree2.node_list)
    if debug:
        print("index of node from tree1: {}, repr: {}\n".format(choice1.index, choice1))
        print("index of node from tree2: {}, repr: {}\n".format(choice2.index, choice2))
    swap_parents(choice1, choice2, tree1, tree2)
    # re-index each node in the tree
    tree1.reorder_whole_tree()
    tree2.reorder_whole_tree()


def swap_parents(node1, node2, tree1, tree2):
    """remaps the parents of two randomly selected nodes"""
    parent_node1, left_node1, right_node1, val_node1, coeff_node1, op_node1, num_children_node1, name_node1, depth_n1 = get_relatives(
        node1
    )
    parent_node2, left_node2, right_node2, val_node2, coeff_node2, op_node2, num_children_node2, name_node2, depth_n2 = get_relatives(
        node2
    )
    # make nodes point to new parents
    if node1.name != "root":
        pi1 = tree1.node_list.index(node1.parent)
        p1 = tree1.node_list[pi1]
        node1.parent = p1
    if node2.name != "root":
        pi2 = tree2.node_list.index(node2.parent)
        p2 = tree2.node_list[pi2]
        node2.parent = p2

    node1.coeff = coeff_node2
    node1.value = val_node2
    node1.left = left_node2
    node1.right = right_node2
    node1.op = op_node2
    node1.num_children = num_children_node2
    node1.depth = depth_n2

    node2.coeff = coeff_node1
    node2.value = val_node1
    node2.left = left_node1
    node2.right = right_node1
    node2.op = op_node1
    node2.num_children = num_children_node1
    node2.depth = depth_n1

def get_relatives(node):
    """returns deep copy of node attributes"""
    return copy.deepcopy(
        [node.parent, node.left, node.right, node.value, node.coeff, node.op, node.num_children, node.name, node.depth]
    )


def plot_tree(tree, show_coeff=True, render=False, render_format='pdf', name="test"):
    """takes in tree & creates graphviz (DOT language graph description).

    :param tree: Tree you want to visualize

    :param show_coeff: displays coefficients on nodes that have them

    :param render: exports the graph image if True

    :param render_format: Can choose png, pdf, svg, etc.

    :param name: Name of exported model
    """
    root = tree.get_root()
    dot = Digraph(comment="tree")
    dot.attr(rank='min')
    # Create the nodes
    for n in tree.node_list:
        # if node is not leaf node, only display the operation (and coeff if applicable)
        if n.num_children > 0:
            if show_coeff:
                if n.coeff is not None:
                    dot.node(str(n.index), str(n.coeff) + '\n' + str(n.op))
                else:
                    dot.node(str(n.index), n.op)
            else:
                dot.node(str(n.index), n.op)
        # if leaf node, then display either the coeff, value, or coeff * value.
        # Current implementation of nodes makes so that each node must contain either
        # a constant coefficient or a variable, or both
        else:
            if n.coeff is not None and n.value is None:
                dot.node(str(n.index), str(n.coeff))
            elif n.coeff is not None and n.value is not None:
                dot.node(str(n.index), str(n.coeff) + '*' + str(n.value))
            elif n.coeff is None and n.value is not None:
                dot.node(str(n.index), str(n.value))

    # For each node in tree, draw edges to its children
    for n in tree.node_list:
        if n.num_children > 0:
            if n.left is not None:
                dot.edge(str(n.index), str(n.left.index))
            if n.right is not None:
                dot.edge(str(n.index), str(n.right.index))

    # Export & display tree in separate window
    if render:
        dot.format = render_format
        dot.render('test-output/{}'.format(name), view=True)

    return dot

# def normalize_data(data):
#     """Takes in dataset as a dict and returns normalized dataset """
#     new
#     for col in data:
#         normalize(col)
#
#     return norm_data, std_data