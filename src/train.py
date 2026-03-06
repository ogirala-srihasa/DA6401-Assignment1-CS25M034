"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
from utils.data_loader import load_fashion_mnist
from utils.data_loader import load_mnist
from sklearn.model_selection import train_test_split
from ann.neural_network import NeuralNetwork
import wandb
import numpy as np


def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', help="choose between 'mnist' or 'fashion_mnist'", choices=['mnist','fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type = int ,default= 128, help='Mini-batch size')
    parser.add_argument('-lr', '--learning_rate', type= float, default= 0.001, help= 'Learning rate for optimizer')
    parser.add_argument('-o', '--optimizer', type = str, default= 'sgd', choices=['sgd','momentum','nag','rmsprop'], help = 'choose the optimizer')
    parser.add_argument('-nhl', '--num_layers', type= int, default= 3, help= 'number of hidden layers')
    parser.add_argument('-sz','--hidden_size', type= int,nargs='+', default= [128,64,32], help='list of sizes of hiddenlayers')
    parser.add_argument('-a','--activation', type=str, default= 'relu', choices= ['relu','sigmoid','tanh'])
    parser.add_argument('-l' , '--loss', type= str, default='cross_entropy', choices=['cross_entropy','mean_squared_error'])
    parser.add_argument('-w_i', '--weight_init',type = str, default='xavier',choices=['random','zeros','xavier'])
    parser.add_argument('-wd', '--weight_decay', type = float, default= 0, help='weight decay for L2 regularization')
    parser.add_argument('-w_p','--wandb_project', type = str, help= 'Project name used to track experiments in Weights & Biases dashboard', default='DA6401-Assignment-1')
    parser.add_argument('-p','--model_save_path',type=str, default='best_model.npy')


    
    return parser.parse_args()


def main():
    """
    Main training function.
    """
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=args)
    if (args.dataset == 'mnist'):
        x_train,y_train,x_test,y_test = load_mnist()
    else:
        x_train,y_train,x_test,y_test = load_fashion_mnist()
    
    

    network = NeuralNetwork(args)
    network.train(x_train,y_train,args.epochs,args.batch_size)
    test_accuracy, test_loss = network.evaluate(x_test,y_test)
    print("Training complete!")
    wandb.finish()
    best_weights = network.get_weights()
    np.save("best_model.npy", best_weights)



if __name__ == '__main__':
    main()