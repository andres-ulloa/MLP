
import numpy as np
import pandas as pd

label_position = 28 * 28


def compute_cost_function():
    pass

def compute_activation_function(activatiom_func):
    pass

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


def train():
    
    dataset = np.genfromtxt('mnist.txt')
    input_layer_size = 28 * 28 #there are going to be as much input neurons as there are pixels in each image
    num_classes = 10
    sigmoid_func =  lambda x:  1/(1 + math.exp(- x))
    train_test, test_set = trim_dataset(dataset, 900, 100)
    print(len(train_test), len(test_set))

if __name__ == '__main__':
    train()