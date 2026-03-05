"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
class optimizer:

    def __init__(self, type):
        self.t = 0
        self.type = type

    def update_weights(self,layers,learning_rate,weight_decay):

        if(self.type == 'sgd'):
            for layer in layers:
                W = layer.weights
                b = layer.bias
                grad_W = layer.grad_W
                grad_b = layer.grad_b
                layer.weights = W - (learning_rate * grad_W)
                layer.bias = b - (learning_rate * grad_b)

        elif (self.type == 'momentum'):
            #Assuming gamma = 0.9 since we arent taking gamma as cli 
            for layer in layers:
                W = layer.weights
                b = layer.bias
                grad_W = layer.grad_W
                grad_b = layer.grad_b
                v_wold = layer.v_w
                v_bold = layer.v_b
                v_w = (0.9 * v_wold) + (learning_rate* grad_W)
                v_b = (0.9 * v_bold) + (learning_rate* grad_b)
                layer.v_w = v_w
                layer.v_b = v_b
                layer.weights = W - v_w
                layer.bias = b - v_b

        elif (self.type == 'nag'):
            #Assuming gamma = 0.9 since we arent taking gamma as cli 
            for layer in layers:
                W = layer.weights
                b = layer.bias
                grad_W = layer.grad_W
                grad_b = layer.grad_b
                v_wold = layer.v_w
                v_bold = layer.v_b
                v_w = (0.9 * v_wold) + (learning_rate* grad_W)
                v_b = (0.9 * v_bold) + (learning_rate* grad_b)
                layer.v_w = v_w
                layer.v_b = v_b
                layer.weights = W - (0.9 * v_w + (learning_rate * grad_W))
                layer.bias = b - (0.9 * v_b + (learning_rate* grad_b))

        elif (self.type == 'rmsprop'):
            #Assuming beta = 0.9
            for layer in layers:
                W = layer.weights
                b = layer.bias
                grad_W = layer.grad_W
                grad_b = layer.grad_b
                m_wold = layer.m_w
                m_bold = layer.m_b
                m_w = (0.9 * m_wold) + (0.1 * (grad_W**2))
                m_b = (0.9 * m_bold) + (0.1 * (grad_b**2))
                layer.m_w = m_w
                layer.m_b = m_b
                layer.weights = W - ((learning_rate * grad_W)/(1e-9 + np.sqrt(m_w)))
                layer.bias = b - ((learning_rate * grad_b)/(1e-9 + np.sqrt(m_b)))