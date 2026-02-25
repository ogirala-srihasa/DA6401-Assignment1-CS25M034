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
        batch_size = y.shape[0]
        ybar = np.zeros((10,batch_size))
        ybar[y,np.arange(batch_size)] = 1
        if (self.type == 'mean_squared_error'):
            l = np.square(ybar - yhat)
            self.d = (yhat - ybar) / 5
            return np.mean(l)
        elif (self.type == 'cross_entropy'):
            #adding 1e-9 for a small chance that yhat may be very very small number
            l = -(np.sum(ybar * np.log((yhat+(1e-9)))))/batch_size
            self.d = -ybar / (yhat+(1e-9))
            return l
    
    def backwards(self):
        #calculated and stored the derivative during the forward pass
        return self.d