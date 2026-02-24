"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class Loss_functions:

    def __init__(self, type):
        self.type = type
        self.d = None

    def loss_computation(self,y,yhat):
        #ybar is the one-hot encoded vector
        ybar = np.zeros(10)
        ybar[y] = 1
        if (self.type == 'mean_squared_error'):
            l = np.square(ybar - yhat)
            self.d = (yhat - ybar) / 5
            return np.mean(l)
        elif (self.type == 'cross_entropy'):
            #adding 1e-9 for a small chance that yhat may be very very small number
            l = -ybar * np.log((yhat+(1e-9)))
            self.d = np.zeros(10)
            self.d[y] = -1 / (yhat[y]+(1e-9))
            return np.sum(l)
    
    def backwards(self):
        #calculated and stored the derivative during the forward pass
        return self.d