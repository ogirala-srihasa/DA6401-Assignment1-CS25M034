"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from neural_layer import NeuralLayer
from objective_functions import Loss_functions
from optimizers import optimizer
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.input_size = 784
        self.output_size = 10
        self.layers = []
        nhiddenlayers = cli_args.num_layers
        self.activation = cli_args.activation
        self.weight_init = cli_args.weight_init
        self.layersizes = cli_args.hidden_size
        self.optimizer = optimizer(cli_args.optimizer)
        self.loss_function = Loss_functions(cli_args.loss)
        self.learning_rate  = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        while(nhiddenlayers < len(self.layersizes)):
            self.layersizes.append(self.layersizes[-1])

        self.layers.append(NeuralLayer(784,self.layersizes[0],self.weight_init,self.activation))
        for i in range(1,nhiddenlayers):
            self.layers.append(NeuralLayer(self.layersizes[i-1],self.layersizes[i],self.weight_init,self.activation))
        self.layers.append(NeuralLayer(self.layersizes[-1],10,self.weight_init,'softmax'))


    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """
        prev = X
        curr = None
        for i in self.layers:
            curr = i.forward(prev)
            prev = curr
        return curr
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """
        loss = self.loss_function.loss_computation(y_true,y_pred)
        prev = self.loss_function.backwards()
        for i in self.layers[::-1]:
            prev = i.backward(prev)
        for layer in self.layers:
            layer.grad_W += self.weight_decay * layer.weights
            loss += (self.weight_decay/2) * np.sum(np.square(layer.weights))
        return loss
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update_weights(self.layers,self.learning_rate,self.weight_decay)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        training_samples = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(training_samples)
            count = 0
            for sample in permutation:
                yhat = self.forward(X_train[sample])
                self.backward(y_train[sample],yhat)
                count += 1
                if(count%batch_size == 0):
                    self.update_weights()
                    for layer in self.layers:
                        layer.reset_gradients()
                    count = 0
                
            if(count != 0):
                self.update_weights()
                for layer in self.layers:
                    layer.reset_gradients()
            
                
                
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        test_samples = X.shape[0]
        correct_predicted = 0
        total_loss = 0
        for sample in range(test_samples):
            yhat = self.forward(X[sample])
            ypred = np.argmax(yhat)
            if(ypred == y[sample]):
                correct_predicted += 1
            total_loss += self.loss_function.loss_computation(y[sample],yhat)
        
        return (correct_predicted/test_samples,total_loss/test_samples)