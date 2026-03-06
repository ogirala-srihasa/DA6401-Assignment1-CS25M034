"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import Loss_functions
from ann.optimizers import optimizer
import wandb
from sklearn.model_selection import train_test_split
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args = None):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        print(vars(cli_args))
        self.input_size = 784
        self.output_size = 10
        self.layers = []
        self.loss_function = Loss_functions('cross_entropy')
        if cli_args:
            if hasattr(cli_args, "num_layers"):
                nhiddenlayers = cli_args.num_layers
            else:
                nhiddenlayers = cli_args.num_hidden_layers
            if hasattr(cli_args, "hidden_size"):
                self.layersizes = cli_args.hidden_size
            else:
                self.layersizes = cli_args.hidden_layer_sizes
            self.activation = cli_args.activation
            self.weight_init = cli_args.weight_init
            
            self.optimizer = optimizer(cli_args.optimizer)
            self.loss_function = Loss_functions(cli_args.loss)
            self.learning_rate  = cli_args.learning_rate
            self.weight_decay = cli_args.weight_decay
            while(nhiddenlayers > len(self.layersizes)):
                self.layersizes.append(self.layersizes[-1])

            self.layers.append(NeuralLayer(784,self.layersizes[0],self.weight_init,self.activation))
            for i in range(1,nhiddenlayers):
                self.layers.append(NeuralLayer(self.layersizes[i-1],self.layersizes[i],self.weight_init,self.activation))
            self.layers.append(NeuralLayer(self.layersizes[-1],10,self.weight_init,'linear'))


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
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        self.loss_function.loss_computation(y_true,y_pred)
        prev = self.loss_function.backwards()
        for layer in self.layers[::-1]:
            prev = layer.backward(prev)
        for layer in self.layers:
            layer.grad_W += self.weight_decay * layer.weights
        for layer in self.layers[::-1]:
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        self.optimizer.update_weights(self.layers,self.learning_rate,self.weight_decay)
    
    def train(self, X_train, y_train, epochs = 1, batch_size = 32):
        """
        Train the network for specified epochs.
        """
        X_train,x_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.1,random_state= 6)
        training_samples = X_train.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(training_samples)
            for i in range(0,training_samples,batch_size):
                indices = permutation[i : i + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]
                yhat = self.forward(batch_X)
                self.backward(batch_y, yhat)
                self.update_weights()
            l2_loss = 0
            for layer in self.layers:
                l2_loss += 0.5 * self.weight_decay * np.sum(np.square(layer.weights))
            
            train_epoch_accuracy, train_epoch_loss = self.evaluate(X_train,y_train)
            train_epoch_loss += l2_loss
            val_epoch_accuracy, val_epoch_loss = self.evaluate(x_val,y_val)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_epoch_loss:.4f} - Val Acc: {val_epoch_accuracy:.4f}")
            log_dict = {
                "epoch": epoch+1,
                "train_loss": train_epoch_loss,
                "train_accuracy": train_epoch_accuracy, 
                "val_loss": val_epoch_loss,
                "val_accuracy": val_epoch_accuracy,
                "first_layer_grad_norm": np.linalg.norm(self.layers[0].grad_W) # for question 2.4
            }
            for i in range(len(self.layers)-1):
                layer = self.layers[i]
                dead_neuron_percentage = 0
                if(layer.activation.type == 'relu'):
                    dead_neuron_percentage = np.mean((layer.activation.a) <= 1e-9) * 100
                elif(layer.activation.type == 'tanh'):
                    dead_neuron_percentage = np.mean(np.abs(layer.activation.a) + 1e-9 >= 1) * 100
                elif(layer.activation.type == 'sigmoid'):
                    dead_neuron_percentage = np.mean(np.abs((2 * layer.activation.a)-1) + 1e-9 >= 1) * 100
                log_dict[f'layer:{i}_dead_neuron_pct'] = dead_neuron_percentage


            wandb.log(log_dict)
            
                
                
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        test_samples = X.shape[0]
        correct_predicted = 0
        total_loss = 0
        batch_size = 128
        for i in range(0,test_samples,128):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yhat = self.forward(X_batch)
            predictions = np.argmax(yhat, axis=1)
            correct_predicted += np.sum(predictions == y_batch)
            total_loss += self.loss_function.loss_computation(y_batch,yhat)* len(y_batch)
        
        return (correct_predicted/test_samples,total_loss/test_samples)
    
    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.weights.copy()
            d[f"b{i}"] = layer.bias.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.weights = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.bias = weight_dict[b_key].copy()