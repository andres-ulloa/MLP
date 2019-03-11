
import numpy as np
import math



class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, learning_rate, num_layers):
       
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        self.num_layers = num_layers


    def sigmoid(x):
        return 1.0/(1+ np.exp(-x))


    def sigmoid_derivative(x):
        return x * (1.0 - x)
    

    def initialize_weights(self):
        self.weights = []
        
        for i in range(0, self.num_hidden_layers): 
             
            weights_vector = None
            
            if i == 0: #we consider this to be the edge case where the weights connect the input layer to the hidden layers
                
                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.input_layer_size)
            
            elif i == num_hidden_layers: # we treat this as the special case where the inner layers connect to the output layer 
                
                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = self.hidden_layer_size * self.output_layer_size)
            
            else:

                weights_vector = np.random.uniform(low = -0.1, high = 0.1, size = (self.hidden_layer_size * self.hidden_layer_size) + 1) #we add 1 to account for the bias unit in each hidden layer
            
            self.weights.append(weights_vector)
            self.bias_units = np.ones(num_layers) 


    def load_weights_from_memory(self, csv_path):
       params = np.loadtxt(csv_path, delimiter = ',')
       self.bias_units = params[len(params)]
       self.weights = np.delete(len(params))


    def save_weights(self, csv_path):
        params = self.weights.copy()
        params.append(self.bias_units)   
        np.savetxt(csv_path, params, delimiter = ',')


    def classify(self, feature_vector):
        run_activation_pass(feature_vector)
        return self.activation_vals[len(self.activation_vals)]


    def train_on_input(self, dataset):
        
        for i in range(0, len(dataset)):    

            print('Going through sample: ', i, '\n\n')
            label = dataset[len(dataset[i])]
            feature_vector = np.delete(dataset[i], len(dataset[i]))
            run_activation_pass(feature_vector)
            
            if num_hidden_layers > 1:
                run_multilayer_backpropagation(feature_vector, label)
            else:
                run_shallow_backpropagation(feature_vector, label)   


    def run_activation_pass(self, feature_vector):
     
        self.activation_vals = []
        
        for layer_index in range(0 , self.num_layers): 
            
            if layer_index == 0:
                activation_output = sigmoid(np.dot(feature_vector, self.weights[layer_index]) + self.bias_units[layer_index])
       
            else:
                activation_output = sigmoid(np.dot(self.activation_vals[layer_index - 1], self.weights[layer_index]) + self.bias_units[layer_index])
            
            self.activation_vals.append(activation_output)
            
    
    def run_shallow_backpropagation(self, feature_vector,label):
        
        vector_label = np.zeros(self.output_layer_size, dtype = int)
        vector_label[label] = 1

        last_layer_output =  self.activation_vals[len(self.activation_vals)]
        global_error_gradient = (2 * (vector_label - last_layer_output) * sigmoid_derivative(last_layer_output))

        layer_2_gradient = np.dot(last_layer_output, global_error_gradient)
        layer_1_gradient = np.dot(feature_vector,  (np.dot(global_error_gradient, self.weights[1]) * sigmoid_derivative(self.activation_vals[0])))

        self.weights[0] += layer_1_gradient * self.learning_rate
        self.bias_units[0] += layer_1_gradient * self.learning_rate

        self.weights[1] += layer_2_gradient * self.learning_rate
        self.bias_units[1] += layer_2_gradient * self.learning_rate


    """ incompleto """
    def run_multilayer_backpropagation(self, feature_vector, label):
        
        weight_gradients = []
        vector_label = np.zeros(self.output_layer_size, dtype = int)
        vector_label[label] = 1
        
        last_layer_output =  self.activation_vals[len(self.activation_vals)]
        global_error_gradient = (2 * (vector_label - last_layer_output) * sigmoid_derivative(last_layer_output))

        for layer_index in range((self.num_layers) - 1, -1, -1):
               
            #the error starts propagating from the top layer (the one which is the closest to the output layer)
            layer_gradient = np.dot(self.activation_vals[layer_index], np.dot(global_error_gradient, self.weights[layer_index] ) * sigmoid_derivative(self.activation_vals[layer_index]))
            weight_gradients.append(layer_gradient)
        
        for layer_index in range(0, len(self.weights)): 
            self.weights[layer_index] += weight_gradients.pop() * self.learning_rate


        
        
        