
import numpy as np
import math



class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size):
       
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        self.bias_units = np.random.unform(low = -0.1, high = 0.1, size = num_hidden_layers)

    def sigmoid(x):
        return 1.0/(1+ np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1.0 - x)
    
    def initialize_weights(self):
        self.weights = []
        
        for i in range(0, self.num_hidden_layers): 
             
            weights_vector = np.zeros(1, dtype = float) 
            
            if i == 0: #we consider this to be the edge case where the weights connect the input layer to the hidden layers
                
                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.input_layer_size)
            
            elif i == num_hidden_layers: # we treat this as the special case where the inner layers connect to the output layer 
                
                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.output_layer_size)
            
            else:

                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = (self.hidden_layer_size * self.hidden_layer_size) + 1) #we add 1 to account for the bias unit in each layer
            
            self.weights.append(weights_vector)


    def classify(self, feature_vector):
         pass


    def train_on_input(self, dataset):
        for i in range(0, len(dataset)):
            print('Going through sample: ', i, '\n')
            run_activation_pass(dataset[i])
            run_backpropagation(dataset[i])


    def run_activation_pass(self, feature_vector):
        self.activation_vals = []
        
        for layer_index in range(0 , self.num_hidden_layers + 2): #adds 2 to take into consideration the weights corresponding to the output and input layers
            
            if layer_index == 0:
                activation_output = sigmoid(np.dot(feature_vector, self.weights[layer_index]))
       
            else:
                activation_output = sigmoid(np.dot(self.activation_vals[layer_index - 1], self.weights[layer_index]))
            
            self.activation_vals.append(activation_output)
            
        

    def run_backpropagation(self, feature_vector):
        self.error_vals = []
        
        return error