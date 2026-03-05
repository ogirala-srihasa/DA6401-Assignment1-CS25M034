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
        self.grad_W = np.zeros((neurons,input_size))
        self.grad_b = np.zeros(neurons)
        self.v_w = np.zeros((neurons,input_size))
        self.v_b = np.zeros(neurons)
        self.m_w = np.zeros((neurons,input_size))
        self.m_b = np.zeros(neurons)

        if(initialization == "random"):
            #the -0.5 is so that the weights all wont be positive
            self.weights = np.random.rand(neurons,input_size) - 0.5
            self.bias = np.random.rand(neurons) - 0.5

        elif(initialization == "xavier"):
            self.weights = np.random.normal(0.0,np.sqrt(2.0)/np.sqrt(input_size + neurons),(neurons,input_size))
            self.bias = np.zeros(neurons)

        elif(initialization == "zeros"):
            self.weights = np.zeros((neurons,input_size))
            self.bias = np.zeros(neurons)

    def forward(self,aprev):
        self.aprev = aprev.T
        z = np.dot(self.weights,aprev.T) + self.bias.reshape(-1,1)
        a = (self.activation.forward(z)).T
        return a
    
    def backward(self,da1):
        da = da1.T
        batch_size = da.shape[1]
        dz = self.activation.backward(da)
        self.grad_W = np.dot(dz,self.aprev.T)/batch_size
        self.grad_b = np.sum(dz,axis=1)/batch_size
        daprev = np.dot(self.weights.T, dz)
        return daprev.T
    
    def reset_gradients(self):
        self.grad_W = np.zeros((self.neurons,self.input_size))
        self.grad_b = np.zeros(self.neurons)


    