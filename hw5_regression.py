# CSE 404 Intro to Machine Learning
# Homework 5: Linear Regression & Optimization

# imports
import time
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Randomly split the dataset into training & testing sets
# @train_perc :: Percentage (in decimal format) of data to use for training
#                Example: if train_perc == 0.7 --> 70% training, 30% testing
def rand_split_train_test(data, label, train_perc):
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    data_tr = data[:num_train_sample]
    data_te = data[num_train_sample:]

    label_tr = label[:num_train_sample]
    label_te = label[num_train_sample:]

    return data_tr, data_te, label_tr, label_te


# Takes a subsample of the entire dataset
def subsample_data(data, label, subsample_size):
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])
    data, label = shuffle(data, label)
    data = data[:subsample_size]
    label = label[:subsample_size]
    return data, label


# Generates a random dataset with dimensions based on feature_size & sample_size
# @bias :: for Gaussian noise
def generate_rnd_data(feature_size, sample_size, bias=False):
    # Generate X matrix
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)

    # Generate ground truth model
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10

    # Generate labels
    label = np.dot(data, truth_model)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model



# Sine Function :)
def sine_data(sample_size, order_M, plot_data = False, noise_level = 0.1, bias = False):
    if int(order_M) != order_M: 
        raise Exception('order_M should be an integer.')
    if order_M < 0:
        raise Exception('order_M should be at least larger than 0.')
    
    # Generate X matrix
    x = np.random.rand(sample_size,1) * 2 * np.pi        # generate x from 0 to 2pi
    X = np.column_stack([ x**m for m in range(order_M)])

    data = np.concatenate((X, np.ones((sample_size, 1))), axis=1) if bias else X

    # Ground truth model: a sine function
    f = lambda x: np.sin(x)

    # Generate labels
    label = f(x)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)*noise_level

    if plot_data:
        plt.figure()
        xx = np.arange(0, np.pi * 2, 0.001)
        yy = f(xx)
        plt.plot(xx, yy, linestyle = '-', color = 'g', label = 'Objective Value')
        plt.scatter(x, label, color = 'b', marker = 'o', alpha = 0.3)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Sine Data (N = %d) with Noise Level %.4g.".format(sample_size, noise_level))
        plt.show()

    return data, label, f


######################################################################################

def mean_squared_error(true_label, predicted_label):
    """
        Compute the mean square error between the true and predicted labels
        :param true_label: Nx1 vector
        :param predicted_label: Nx1 vector
        :return: scalar MSE value
    """
    mse = (true_label - predicted_label)**2
    mse = mse / len(true_label)
    mse = np.sum(mse)
    mse = np.sqrt(mse)
    return mse



def least_squares(feature, target):
    """
        Compute the model vector obtained after MLE
        w_star = (X^T X)^(-1)X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :return: w_star (d+1)x1 model vector
        """
    w_star = np.dot(feature.T, feature)
    w_star = np.linalg.inv(w_star)
    w_star = np.dot(w_star, np.dot(feature.T, target))
    return w_star



def ridge_regression(feature, target, lam = 1e-17):
    """
        Compute the model vector when we use L2-norm regularization
        w_star = (X^T X + lambda I)^(-1) X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :param lam: the scalar regularization parameter, lambda
        :return: w_star (d+1)x1 model vector
        """
    I = np.eye(feature.shape[1])
    w_star = np.dot(feature.T, feature) + I*lam
    w_star = np.linalg.inv(w_star)
    w_star = np.dot(w_star, np.dot(feature.T, target))
    return w_star


# K-fold cross validation
#def k_fold(current_fold, total_fold, total_sample_size):
#    folds = np.spl
#    return data_train, data_test, label_train, label_test

########################################################################################

def compute_gradient(feature, target, model, lam = 1e-17):
    # Compute the gradient of linear regression objective function with respect to w
    gradient = ridge_regression(feature, target, lam)
    gradient = np.gradient(gradient)
    return gradient



#def compute_objective_value(feature, target, model):
    # Compute MSE 
#    return mse


# Gradient Descent
def gradient_descent(feature, target, step_size, max_iter, lam = 1e-17):
   
    #for i in range(max_iter):
        # Compute gradient

        # Update the model

        # Compute the error (objective value)
        
    return #model, objective value


# Stochastic Gradient Descent
def batch_gradient_descent(feature, target, step_size, max_iter, batch_size, lam = 1e-17):
   
    #for i in range(max_iter):
        
        # Compute gradient

        # Update the model

        # Compute the error (objective value)
        
    return #model, objective value

# Plots/Errors
# def plot_objective_function(objective_value, batch_size=None):
# def print_train_test_error(train_data, test_data, train_label, test_label, model):

##########################################################################################

# TODO: Homework Template
if __name__ == '__main__':
    #plt.interactive(False)
    #np.random.seed(491)


    # Problem 1
    # [X] Complete Least Squares, Ridge Regression, MSE

    # [X] Randomly generate & 30 data points using sine function
    X = np.random.rand(1,30) * 2 * np.pi
    y = np.sin(X) + np.random.normal(0, 0.3, (1, 30))
    # [X] Plot the data points along with the sine function
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Random Set of 30 sine data points with guassian noise')
    plt.scatter(X, y)
    sinX = np.linspace(0, 2*np.pi, 100)
    sinY = np.sin(sinX)
    plt.plot(sinX, sinY, "r")
    plt.show()

    # [X] Randomly split the dataset
    data_tr, data_te, label_tr, label_te = rand_split_train_test(X.T, y.T, 0.7)

	# [X] For each lambda, use Ridge Regression to calculate & plot MSE for training & testing sets
    lambdasSTR = ["1e-10", "1e-5", "1e-2", "1e-1", "1", "10", "100", "1000"]
    lambdas = [10**-10, 10**-5, 10**-2, 10**-1, 1, 10, 100, 1000]
    testPerformance = []
    trainPerformance = []
    for lam in lambdas:
        wStar = ridge_regression(data_tr, label_tr, lam)
        testPerformance.append(mean_squared_error(label_te, wStar))
        trainPerformance.append(mean_squared_error(label_tr, wStar))
    fig = plt.figure()
    ax2 = fig.add_subplot()
    ax2.set_xlabel('lambda values')
    ax2.set_ylabel('MSE')
    ax2.set_title('Performance of Training Data')
    plt.bar(lambdasSTR, trainPerformance)
    plt.show()
    fig = plt.figure()
    ax3 = fig.add_subplot()
    ax3.set_xlabel('lambda values')
    ax3.set_ylabel('MSE')
    ax3.set_title('Performance of Testing Data')
    plt.bar(lambdasSTR, testPerformance)
    plt.show()

	# [] Implement k-fold CV & choose best lambda
    maxLam = -1
    maxPerformance = -1
    for lam in lambdas:
        foldPerformance = k_fold(X, y, lam, 4)
        if foldPerformance > maxPerformance:
            maxLam = lam
            maxPerformance = foldPerformance
    print("The best performing lambda value is " + str(maxLam) + " with a k-fold MSE of " + str(maxPerformance))


    # Problem 2
    # [] Complete Gradient Descent & Stochastic GD

    # [] Implement ridge regression with GD & plot objectives at each iteration

    # [] Implement SGD & plot objectives at each iteration per batch 


	

    