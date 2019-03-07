
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        self.weights = initialize_weights()

    
    def initialize_weights(self):
        weights = []
        
        for i in range(0, self.num_hidden_layers):
            
            vector_weight = np.zeros(1, dtype = float)
            
            if i == 0: #we consider this to be the edge case where the weights connecting the input layer to the hidden layers
                
                vector_weight = np.random(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.input_layer_size)
            
            elif i == num_hidden_layers: # we treat this as the special case where the inner layers connect to the output layer 
                
                vector_weight = np.random(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.output_layer_size)
            
            else:

                vector_weight = np.random.uniform(low = -0.1, high = 0.1, size = self.hidden_layer_size + 1) #we add 1 to account for the bias unit in each layer
            
            weights.append(vector_weight)

        return weights


    def classify(self, feature_vector):
         pass
         
    def run_activation_pass(self, dataset):
        activation_vals = 0

        return activation_vals

    def run_backpropagation(self, activation_pass_vals):
        error = 0

        return error