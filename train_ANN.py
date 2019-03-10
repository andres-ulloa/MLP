
import numpy as np
import pandas as pd
from ANN import ANN
label_position = 28 * 28



def classify(dataset, model):

    global label_position
    labels = list()
    examples_counter = 0
    likelihood = list()
    
    for example in dataset:

        examples_counter +=1 
        likelihood = list()

        #print('Likelihood of belonging to each possible class = \n\n', likelihood)
        likelihood.sort(key = lambda tup: tup[0])
        label = (examples_counter, likelihood[len(likelihood) - 1][1], example[label_position])
        labels.append(label)
    
    return labels


def trim_dataset(dataset, train_set_size, test_set_size):
    
    training_set = list()
    test_set = list()
    
    for i in range(0, train_set_size + test_set_size):
        if i < train_set_size:
            training_set.append(dataset[i])
        if i >= train_set_size:
            test_set.append(dataset[i])

    return training_set, test_set


def compute_confution_matrix(labels):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for label in labels:

        assigned_label = label[1]
        true_label = label[2]

        if assigned_label == true_label and assigned_label == 1:
            true_positives += 1
        
        elif assigned_label == true_label and assigned_label == 0:
            true_negatives += 1

        elif assigned_label !=  true_label and assigned_label == 0:
            false_negatives += 1
        
        elif assigned_label != true_label and assigned_label == 1:
            false_positives += 1
    
    confution_matrix = np.array([[true_positives, false_positives],[false_negatives, true_negatives]])
    return confution_matrix


def train(neural_net, dataset, num_epochs):

    neural_net.initialize_weights()
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------INITIALIZING TRAINING-----------------------------------------------')
    print('------------------------------------------------------------------------------------------------------\n\n')
    print('Epochs = ', num_epochs)
    print('Alpha_rate = ', neural_net.learning_rate)
    print('Hidden layers = ', neural_net.num_hidden_layers)
    print('Input layer size = ', neural_net.input_layer_size)
    print('Hidden layer size = ', neural_net.hidden_layer_size)
    print('Output layer size = ', neural_net.output_layer_size)
    input('Press enter to continue...')
    for i in range(0 , num_epochs):
        print('------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------')
        print('------------------------------------------------------------------------------------------------------')
        print('Epoch num: ', i, '\n')
        neural_net.train_on_input(dataset)
        print('---------------------------------------------------------------------------------------------------------------\n')
        print('---------------------------------------------------------------------------------------------------------------\n')


def main():
    
    dataset = np.genfromtxt('mnist.txt')
    input_layer_size = 28 * 28 #there are going to be as much input neurons as there are pixels in each image
    num_classes = 10
    num_hidden_layers = 2
    hidden_layer_size = 10 #not considering bias units so a + 1 size in each layer should always be taken into account
    num_epocs = 1000
    training_set_size = 900
    test_set_size = 100

    train_test, test_set = trim_dataset(dataset, train_set_size, test_set_size)    
    neural_net = ANN(input_layer_size, num_classes, num_hidden_layers, sigmoid_func, hidden_layer_size)
    train(neural_net, dataset, num_epochs)

    """
    labels = classify(test_set, neural_net)
    print('\nRESULTS = \n', compute_confution_matrix(labels))
    """


if __name__ == '__main__':
    main()