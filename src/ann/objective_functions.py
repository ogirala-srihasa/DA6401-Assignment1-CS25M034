"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from ann.activations import Activations
class Loss_functions:

    def __init__(self, type):
        self.type = type
        self.d = None

    def loss_computation(self,y,yhat):
        #ybar is the one-hot encoded vector
        batch_size = y.shape[0]
        ybar = np.zeros((10,batch_size))
        ybar[y,np.arange(batch_size)] = 1
        ybar = ybar.T
        if (self.type == 'mean_squared_error'):
            l = np.square(ybar - yhat)
            self.d = (yhat - ybar)
            return np.mean(l)
        elif (self.type == 'cross_entropy'):
            sma = Activations('softmax')
            prb  =  (sma.forward(yhat.T)).T
            #adding 1e-9 for a small chance that yhat may be very very small number
            l = -(np.sum(ybar * np.log((prb+(1e-9)))))/batch_size
            self.d = prb - ybar
            return l
    
    def backwards(self):
        #calculated and stored the derivative during the forward pass
        return self.d