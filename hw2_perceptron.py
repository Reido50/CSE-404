# Homework 2: Perceptron

import numpy as np  #functions, vectors, matrices, linear algebra...
from random import choice, random   #randomly select item from list
import matplotlib.pyplot as plt #plots

def train_perceptron(training_data):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    indices = np.arange(X.shape[0])
    model_size = X.shape[1]
    w = np.zeros(model_size)    #np.random.rand(model_size)
    iteration = 1
    while True:
        # compute results according to the hypothesis
        r = np.sign(np.multiply(np.matmul(X, w), y))

        # get incorrect predictions (you can get the indices)
        incorrect_indices = indices[r != 1]

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if len(incorrect_indices) == 0:
            break

        # Pick one misclassified example.
        picked_incorrect = choice(incorrect_indices)
        xstar = X[picked_incorrect]
        ystar = y[picked_incorrect]

        # Update the weight vector with perceptron update rule
        w += xstar * ystar

        iteration += 1

    return w

def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))


if __name__ == '__main__':

    rnd_x = np.array([[0,1,1],\
                      [0.6,0.6,1],\
                      [1,0,1],\
                      [1,1,1],\
                      [0.3,0.4,1],\
                      [0.2,0.3,1],\
                      [0.1,0.4,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    trained_model = train_perceptron(rnd_data)
    print("Model:", trained_model)
    print_prediction(trained_model, rnd_x)



