
import numpy as np

class ANN:
    
    def __init__(self, input_layer_size, output_layer_size, num_hidden_layers, activation_func, hidden_layer_size):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.weights = np.zeros((self.hidden_layer_size * self.input_layer_size + self.num_hidden_layers), dtype = float)
        self.activation_func = activation_func
        self.hidden_layer_size = hidden_layer_size 
       
        

    def initialize_weights(self):
        print('Initializing weights...\n')
        weights = np.random.uniform(low = -0.5, high = 0.5, size = (self.hidden_layer_size * self.input_layer_size + self.num_hidden_layers))
        print('W = ', weights)
        self.weights = weights
    
    def compute_hypotheses():
        pass

    def classify(self, feature_vector):
         pass

    def update_weights(self):
        pass

    def run_activation_pass(self, feature_vector):
        activation_val = 0

        return activation_val

    def run_backpropagation(self, activation_pass_vals):
        error = 0

        return error