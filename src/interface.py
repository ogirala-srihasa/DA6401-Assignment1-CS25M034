"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from ann.neural_network import NeuralNetwork
import numpy as np
from ann.neural_layer import NeuralLayer
from sklearn.metrics import confusion_matrix
from ann.objective_functions import Loss_functions
from utils.data_loader import load_fashion_mnist,load_mnist

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('-p','--path',type=str, default='../models/best_model.npy')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', help="choose between 'mnist' or 'fashion_mnist'", choices=['mnist','fashion_mnist'])
    parser.add_argument('-b', '--batch_size', type = int ,default= 128, help='Mini-batch size')
    parser.add_argument('-nhl', '--num_layers', type= int, default= 3, help= 'number of hidden layers')
    parser.add_argument('-sz','--hidden_size', type= int,nargs='+', default= [128,64,32], help='list of sizes of hiddenlayers')
    parser.add_argument('-a','--activation', type=str, default= 'relu', choices= ['relu','sigmoid','tanh'])
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.

    """
    network = NeuralNetwork()
    parameters = np.load(model_path, allow_pickle= True)
    for layer_param in parameters:
        W = layer_param['weights']
        b = layer_param['bias']
        a = layer_param['activation']
        neurons,input_size = W.shape
        layer = NeuralLayer(input_size,neurons,'random',a)
        layer.weights = W
        layer.bias = b
        network.layers.append(layer)

    return network


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    #assuming crossentropy loss since it is not being taken from cli
    loss_function = Loss_functions('cross_entropy')
    logits = []
    samples = X_test.shape[0]
    loss = 0
    for i in range(0,samples,128):
        X_batch = X_test[i:i+128].T
        y_batch = y_test[i:i+128]
        logits_batch = model.forward(X_batch)
        logits.append(logits_batch.T)
        loss += loss_function.loss_computation(y = y_batch,yhat = logits_batch) * len(y_batch)
    logits = np.vstack(logits)
    y_pred = np.argmax(logits, axis = 1)
    cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
    TP = np.diag(cm) 
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    # Calculate Metrics (add epsilon to avoid division by zero)
    epsilon = 1e-9
    # Accuracy: Total correct / Total samples
    accuracy = np.sum(TP) / np.sum(cm)
    # Precision: TP / (TP + FP)
    precision_per_class = TP / (TP + FP + epsilon)
    precision = np.mean(precision_per_class)
    # Recall: TP / (TP + FN)
    recall_per_class = TP / (TP + FN + epsilon)
    recall = np.mean(recall_per_class)
    # F1 Score: harmonic mean of Precision and Recall
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)
    f1 = np.mean(f1_per_class)

    return {
        "logits": logits,
        "loss": loss/samples,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    if args.dataset == 'mnist':
        _, _, X_test, y_test = load_mnist()
    else:
        _, _, X_test, y_test = load_fashion_mnist()    
    # Load the model
    print(f"Loading model from {args.path}...")
    model = load_model(args.path)
    # Evaluate
    print("Running evaluation (this may take a moment)...")
    metrics = evaluate_model(model, X_test, y_test)
    #Print Results
    print("\n" + "="*30)
    print(f"RESULTS FOR {args.dataset.upper()}")
    print("="*30)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print("="*30)
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()