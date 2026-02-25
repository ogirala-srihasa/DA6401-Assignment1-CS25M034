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
        if (self.type == "relu"):
            a = np.maximum(0,z)
            self.a = a
            return a
        elif (self.type == "sigmoid"):
            a = z * -1
            a = 1 + np.exp(a)
            a = 1/ a
            self.a = a
            return a
        elif (self.type == "tanh"):
            a = np.tanh(z)
            self.a = a
            return a
        elif (self.type == "softmax"):
            #using a numerically stable way to compute softmax since e^z may blow up for large values
            maxv = np.max(z,axis=0, keepdims= True)
            a = z - maxv
            a = np.exp(a)
            sumv = np.sum(a, axis= 0, keepdims= True)
            a = a / sumv
            self.a = a
            return a

    def backward(self, da):
        if (self.type == "relu"):
            d = np.where(self.a > 0, 1, 0)
            return d*da
        elif (self.type == "sigmoid"):
            d = self.a * (1 - self.a)
            return d*da
        elif (self.type == "tanh"):
            d = 1 - (self.a * self.a)
            return d*da
        elif (self.type == "softmax"):
            temp = self.a * da
            st = np.sum(temp,axis= 0, keepdims= True)
            return self.a * (da - st)