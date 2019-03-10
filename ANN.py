
import numpy as np
import math



class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, learning_rate):
       
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        

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
            self.bias_units = np.random.unform(low = -0.1, high = 0.1, size = num_hidden_layers)


    def classify(self, feature_vector):
         pass


    def train_on_input(self, dataset):
        
        for i in range(0, len(dataset)):    
            
            print('Going through sample: ', i, '\n\n')
            label = dataset[len(dataset[i])]
            feature_vector = np.delete(dataset[i], len(dataset[i]))
            run_activation_pass(feature_vector)
            run_backpropagation(feature_vector, label)


    def run_activation_pass(self, feature_vector):
     
        self.activation_vals = []
        
        for layer_index in range(0 , self.num_hidden_layers + 2): #adds 2 to take into consideration the weights that correspond to the output and input layers
            
            if layer_index == 0:
                activation_output = sigmoid(np.dot(feature_vector, self.weights[layer_index]) + self.bias_units[layer_index])
       
            else:
                activation_output = sigmoid(np.dot(self.activation_vals[layer_index - 1], self.weights[layer_index]) + self.bias_units[layer_index])
            
            self.activation_vals.append(activation_output)
            
        

    def run_backpropagation(self, feature_vector, label):
        
        self.weight_gradients = []
        vector_label = np.zeros(self.output_layer_size, dtype = int)
        vector_label[label] = 1

        for layer_index in range((self.num_hidden_layers + 2) - 1, -1, -1):

            #the error starts propagating from the top layer (the one which is the closest to the output layer)
            if layer_index == self.num_hidden_layers + 2:
                pass 
            else:
                pass

        