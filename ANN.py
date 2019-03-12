
import numpy as np
import math


def sigmoid(x):
    return 1.0/(1+ np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, hidden_layer_size, learning_rate, num_layers, input_, labels):
        
        self.labels = labels
        self.input = input_ 
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers             
        self.hidden_layer_size = hidden_layer_size 
        self.num_layers = num_layers
        self.learning_rate = learning_rate
   
    

    def initialize_weights(self):
        self.weights_layer_1   = np.random.rand(self.input.shape[1],self.hidden_layer_size) 
        self.weights_layer_2 = np.random.rand(self.hidden_layer_size, self.output_layer_size) 
        self.bias_unit = 1



    def load_weights_from_memory(self, csv_path):
       params = np.loadtxt(csv_path, delimiter = ',')
       self.bias_unit = params[len(params) - 1]
       self.weights = np.delete(params, len(params) - 1)
     


    def save_weights(self, csv_path):
        print('\nSaving model parameters...')
        params = [self.weights_layer_1, self.weights_layer_2, self.bias_unit] 
        np.savetxt(csv_path, params, delimiter = ',', fmt='%s')
        print('\nDone.')


    def classify(self, dataset):
        self.input = dataset
        #we get rid of the label at the end of the vector
        run_shallow_activation_pass()
        return self.output



    def run_shallow_activation_pass(self):

        self.activation_layer_1 = sigmoid(np.dot(self.input, self.weights_layer_1))
        self.output = sigmoid(np.dot(self.activation_layer_1, self.weights_layer_2)+ self.bias_unit)

            
    
    def run_shallow_backpropagation(self):
        
        global_error_derivative = (2 * (self.labels - self.output) * sigmoid_derivative(self.output))

        layer_2_gradient = np.dot(self.activation_layer_1.T, global_error_derivative)

        layer_2_error_derivative =  (np.dot(2 * (self.labels - self.output) * sigmoid_derivative(self.output), self.weights_layer_2.T) * sigmoid_derivative(self.activation_layer_1))

        layer_1_gradient = np.dot(self.input.T, layer_2_error_derivative)

        b_gradient = np.sum(global_error_derivative)

        self.weights_layer_1 += layer_1_gradient * self.learning_rate
        self.bias_unit += b_gradient * self.learning_rate
        self.weights_layer_2 += layer_2_gradient * self.learning_rate




    """ incompleto """
    def run_multilayer_backpropagation(self, feature_vector, label):
        
        weight_gradients = []
        vector_label = np.zeros(self.output_layer_size, dtype = int)
        vector_label[label] = 1
        
        last_layer_output =  self.activation_vals[len(self.activation_vals) - 1]
        global_error_gradient = (2 * (vector_label - last_layer_output) * sigmoid_derivative(last_layer_output))

        for layer_index in range((self.num_layers) - 1, -1, -1):
               
            #the error starts propagating from the top layer (the one which is the closest to the output layer)
            layer_gradient = np.dot(self.activation_vals[layer_index], np.dot(global_error_gradient, self.weights[layer_index] ) * sigmoid_derivative(self.activation_vals[layer_index]))
            weight_gradients.append(layer_gradient)
        
        for layer_index in range(0, len(self.weights)): 
            self.weights[layer_index] += weight_gradients.pop() * self.learning_rate


        
        
        