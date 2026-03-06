"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from ann.activations import Activations

class NeuralLayer:

    def __init__(self,input_size,neurons,initialization,activation):

        self.input_size = input_size
        self.neurons = neurons
        self.initialization = initialization
        self.activation = Activations(activation)
        self.weights = None
        self.bias = None
        self.aprev = None
        self.a = None
        self.grad_W = np.zeros((input_size,neurons))
        self.grad_b = np.zeros(neurons)
        self.v_w = np.zeros((input_size,neurons))
        self.v_b = np.zeros(neurons)
        self.m_w = np.zeros((input_size,neurons))
        self.m_b = np.zeros(neurons)

        if(initialization == "random"):
            #the -0.5 is so that the weights all wont be positive
            self.weights = np.random.rand(input_size,neurons) - 0.5
            self.bias = np.random.rand(neurons) - 0.5

        elif(initialization == "xavier"):
            self.weights = np.random.normal(0.0,np.sqrt(2.0)/np.sqrt(input_size + neurons),(input_size,neurons))
            self.bias = np.zeros(neurons)

        elif(initialization == "zeros"):
            self.weights = np.zeros((input_size,neurons))
            self.bias = np.zeros(neurons)

    def forward(self,aprev):
        self.aprev = aprev
        z = np.dot(aprev,self.weights) + self.bias
        a = (self.activation.forward(z))
        self.a = a
        return a
    
    def backward(self,da):
        batch_size = da.shape[0]
        dz = self.activation.backward(da)
        self.grad_W = np.dot(self.aprev.T,dz)/batch_size
        self.grad_b = np.sum(dz,axis=0)/batch_size
        daprev = np.dot(dz,self.weights.T)
        return daprev
    



    