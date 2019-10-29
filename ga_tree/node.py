#!/usr/bin/env python
# coding: utf-8
"""
Toshiki Nazikian 10/28/19

Object represents single node in a binary tree representation 
of an analytic function. 

NOTE: Currently nodes can either have a coeff, value, or value and
coeff. If the value is NULL and coeff is not, then it is a leaf node
containing a constant. If coeff is NULL and value is not, it is a leaf
node with a variable and coeff of 1. If both are not NULL, it is a variable 
times a coeff. e.g. (c1 * x1)
"""

__all__=['Node']

class Node():
    def __init__(self, name, parent=None):
        self.name = name
        self.num_children=0
        self.index=None
        self._left=None
        self._right=None
        self._op=None
        self.coeff=None
        self._value = None
        self.probs = None
        self.parent = parent
        
    def init_left(self):
        left_node = Node("left", parent=self)
        self.left = left_node
        self.num_children += 1
        
    def init_right(self):
        right_node = Node("right", parent=self)
        self.right = right_node
        self.num_children += 1 
    
    @property    
    def left(self):
        return self._left
    
    @property    
    def right(self):
        return self._right
    
    @property    
    def op(self):
        return self._op
    
    @property    
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
    
    @left.setter
    def left(self, node):
        self._left = node
    
    @right.setter
    def right(self, node):
        self._right = node

    @op.setter
    def op(self, op):
        self._op = op
        
    def __repr__(self):
        if self.num_children > 0:
            return "(left: {}, op: {}, right: {})".format(
                self.left, self.op, self.right)
        return "{}*{}".format(self.coeff, self.value)
