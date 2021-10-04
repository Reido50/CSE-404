# Homework 2: Perceptron

import numpy as np  #functions, vectors, matrices, linear algebra...
from random import choice, random, randrange   #randomly select item from list
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

    return (w, iteration)

'''
1.4a 20 Random Linearly Separable Data Points
'''
# Label plot
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Random Set of 20 1.4a')
# Generate random set of 20 linearly separated points
rand_x = []
y = []
for i in range(20):
    seed = round(random() * 4 - 2, 1)
    if random() >= 0.5:
        rand_x.append([seed, 0.5*seed -1 - round(random(), 1)])
        y.append(-1)
    else:
        rand_x.append([seed, 0.5*seed + 1 + round(random(), 1)])
        y.append(1)
    plt.scatter(rand_x[i][0], rand_x[i][1])
rand_x = np.array(rand_x)
y = np.array(y)
# Plot the points
x_line = np.linspace(-2,2,10)
y_line = x_line*0.5
plt.plot(x_line, y_line, 'r')
#plt.show()

'''
1.4b Print number of updates perceptron takes to converge
'''
training = [rand_x, y]
result = train_perceptron(training)
g = result[0]
iterations = result[1]
print("1.4b: 20 randoms (linearly separated) points took " + str(iterations) + " iterations.")
plt.plot(g, 'b')
plt.show()

'''
1.4c 20 more Random Linearly Separable Data Points
'''
# Label plot
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Random Set of 20 1.4c')
# Generate random set of 20 linearly separated points
rand_x = []
y = []
for i in range(20):
    seed = round(random() * 4 - 2, 1)
    if random() >= 0.5:
        rand_x.append([seed, 0.5*seed -1 - round(random(), 1)])
        y.append(-1)
    else:
        rand_x.append([seed, 0.5*seed + 1 + round(random(), 1)])
        y.append(1)
    plt.scatter(rand_x[i][0], rand_x[i][1])
rand_x = np.array(rand_x)
y = np.array(y)
# Plot the points
x_line = np.linspace(-2,2,10)
y_line = x_line*0.5
plt.plot(x_line, y_line, 'r')
training = [rand_x, y]
result = train_perceptron(training)
g = result[0]
iterations = result[1]
print("1.4c: 20 more randoms (linearly separated) points took " + str(iterations) + " iterations.")
plt.plot(g, 'b')
plt.show()

'''
1.4d 100 Random Linearly Separable Data Points
'''
# Label plot
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Random Set of 20 1.4d')
# Generate random set of 20 linearly separated points
rand_x = []
y = []
for i in range(100):
    seed = round(random() * 4 - 2, 1)
    if random() >= 0.5:
        rand_x.append([seed, 0.5*seed -1 - round(random(), 1)])
        y.append(-1)
    else:
        rand_x.append([seed, 0.5*seed + 1 + round(random(), 1)])
        y.append(1)
    plt.scatter(rand_x[i][0], rand_x[i][1])
rand_x = np.array(rand_x)
y = np.array(y)
# Plot the points
x_line = np.linspace(-2,2,10)
y_line = x_line*0.5
plt.plot(x_line, y_line, 'r')
training = [rand_x, y]
result = train_perceptron(training)
g = result[0]
iterations = result[1]
print("1.4d: 100 randoms (linearly separated) points took " + str(iterations) + " iterations.")
plt.plot(g, 'b')
plt.show()

'''
1.4e 1000 Random Linearly Separable Data Points
'''
# Label plot
fig = plt.figure()
ax1 = fig.add_subplot()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Random Set of 20 1.4e')
# Generate random set of 20 linearly separated points
rand_x = []
y = []
for i in range(1000):
    seed = round(random() * 4 - 2, 1)
    if random() >= 0.5:
        rand_x.append([seed, 0.5*seed -1 - round(random(), 1)])
        y.append(-1)
    else:
        rand_x.append([seed, 0.5*seed + 1 + round(random(), 1)])
        y.append(1)
    plt.scatter(rand_x[i][0], rand_x[i][1])
rand_x = np.array(rand_x)
y = np.array(y)
# Plot the points
x_line = np.linspace(-2,2,10)
y_line = x_line*0.5
plt.plot(x_line, y_line, 'r')
training = [rand_x, y]
result = train_perceptron(training)
g = result[0]
iterations = result[1]
print("1.4e: 1000 randoms (linearly separated) points took " + str(iterations) + " iterations.")
plt.plot(g, 'b')
plt.show()
