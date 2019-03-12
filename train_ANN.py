
import numpy as np
import pandas as pd
from ANN import ANN
label_position = 28 * 28

def find_highest_scoring_class(vector):

    sorted_list = vector.copy()
    sorted_list.sort()
    target = sorted_list[len(sorted_list) - 1]
    best_score_index = 0

    for i in range(0, len(vector)):
        if vector[i] == target:
            best_score_index = i
            break

    return best_score_index

def classify(dataset, true_labels ,model):

    examples_counter = 0
    predictions = model.classify(dataset)
    results = []
    for num_example in range(0 , len(predictions)):
        label = (num_example, predictions[i], true_labels[i])
        results.append(label)

    return results


def trim_dataset(dataset, train_set_size, test_set_size):
    
    training_set = list()
    test_set = list()
    
    for i in range(0, train_set_size + test_set_size):
        if i < train_set_size:
            training_set.append(dataset[i])
        if i >= train_set_size:
            test_set.append(dataset[i])

    return training_set, test_set


def compute_confution_matrix(labels, num_classes):
    

    for i in range(0 , num_classes):
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

  

    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------INITIALIZING TRAINING-----------------------------------------------')
    print('------------------------------------------------------------------------------------------------------\n\n')
    print('Epochs = ', num_epochs)
    print('Alpha_rate = ', neural_net.learning_rate)
    print('Hidden layers = ', neural_net.num_hidden_layers)
    print('Input layer size = ', neural_net.input_layer_size)
    print('Hidden layer size = ', neural_net.hidden_layer_size)
    print('Output layer size = ', neural_net.output_layer_size)
    input('\nPress enter to continue...')
    
    neural_net.initialize_weights()

    for i in range(0 , num_epochs):
        
        neural_net.run_shallow_activation_pass()
        neural_net.run_shallow_backpropagation()

    print('\nDone.')


def split_labels(dataset):
    
    labels = []
    new_dataset = []

    for i in range(0, len( dataset)):
        feature_vector = dataset[i]
        label = feature_vector[len(dataset[i]) - 1]
        feature_vector = np.delete(feature_vector, len(dataset[i]) - 1)
        labels.append([int(label)])
        new_dataset.append(feature_vector)
    
    return np.asarray(labels), np.asarray(new_dataset)


def main():
    
    dataset = np.genfromtxt('mnist.txt')
    input_layer_size = 28 * 28 #there are going to be as much input neurons as there are pixels in each image
    num_classes = 10
    num_hidden_layers = 1
    hidden_layer_size = 10 #not considering bias units so a + 1 size in each layer should always be taken into account
    num_epochs = 1000
    training_set_size = 900
    test_set_size = 100
    num_layers = num_hidden_layers + 1
    learning_rate = 0.01

    train_test, test_set = trim_dataset(dataset, training_set_size, test_set_size)    
    labels, dataset = split_labels(dataset)
    neural_net = ANN(input_layer_size, num_classes, num_hidden_layers, hidden_layer_size, learning_rate, num_layers, dataset, labels) 
    
    train(neural_net, dataset, num_epochs)
    neural_net.save_weights('model_params.csv')
    
    """
    labels_test, dataset_test = split_labels(test_set)
    
    print('\nRESULTS = \n', compute_confution_matrix(labels, num_classes))
    """


if __name__ == '__main__':
    main()