"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class Activations:
    def __init__(self, type):
        self.type = type
        self.a = None
    
    def forward(self, z):
        if (self.type == "ReLU"):
            a = np.maximum(0,z)
            self.a = a
            return a
        elif (self.type == "Sigmoid"):
            a = z * -1
            a = 1 + np.exp(a)
            a = 1/ a
            self.a = a
            return a
        elif (self.type == "Tanh"):
            a = np.tanh(z)
            self.a = a
            return a
        elif (self.type == "Softmax"):
            #using a numerically stable way to compute softmax since e^z may blow up for large values
            maxv = np.max(z)
            a = z - maxv
            a = np.exp(a)
            sumv = np.sum(a)
            a = a / sumv
            self.a = a
            return a

    def backward(self, da):
        if (self.type == "ReLU"):
            d = np.where(self.a > 0, 1, 0)
            return d*da
        elif (self.type == "Sigmoid"):
            d = self.a * (1 - self.a)
            return d*da
        elif (self.type == "Tanh"):
            d = 1 - (self.a * self.a)
            return d*da
        elif (self.type == "Softmax"):
            diag = np.diag(self.a)
            outer = np.outer(self.a,self.a)
            return (diag-outer) @ da