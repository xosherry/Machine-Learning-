import math
import sys
import numpy as np

# This is the gaussion function that computes ||x_i - x_j ||^2 / 2sigma^2
# numerator: squared euclidean distance
#       (p1 - q1)^2 + (p2 - q2)^2 + ... (pn - qn)^2
# denominator: -2sigma^2
def gaussian(sigma, x, y):
    squared_euc_dist = 0.0
    for i in range(len(x)):
        # print 'x[i]', x[i]
        # print 'y[i]', y[i]
        squared_euc_dist += (x[i] - y[i]) ** 2
    return math.exp(squared_euc_dist / (-2 * (sigma ** 2)))

# This function creates the kernel matrix (the kernel version of the gram matrix)
def create_kernel_matrix(sigma, data_x, data_y):
    dimension = len(data_x)
    #print 'dim', dimension
    matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        #prediction = 0.0
        for j in range(dimension):
            #prediction += alphas[j] * y[j] * gaussian(sigma, test_x[i], x[j])
            matrix[i][j] = gaussian(sigma, data_x[i], data_y[j])
    return matrix

# This function makes the decision result
# If it is less than or equal to zero, our data point has been misclassified.
# Utilizes the passed in kernel matrix created by create_kernel_matrix
def decision(i, alpha_array, x, y, kernelmatrix):
    sum = 0.0
    for j in range(len(alpha_array)):
        #sum += alpha_array[j] * y[j] * gaussian(sigma, x[i],x[j])
        sum += alpha_array[j] * y[j] * kernelmatrix[i][j]
    result = y[i] * sum
    return result

# main function that parses data from all four files, creates the kernel matrix, builds
# the perceptron line by checking if further iterations need to be performed (misclassified point),
# once that is done, we move on to the testing data and classify those points based on the line
# created by the training data.
def kernel_perceptron(sigma, pos_train_file, neg_train_file, pos_test_file, neg_test_file):
    labels = []
    test_true_labels = []
    sigma = float(sigma)

    with open(pos_train_file, 'r') as trainfile:
        total_and_dimensions = trainfile.readline().split()
        pos_train_total = int(total_and_dimensions[0])
        pos_dimensions = int(total_and_dimensions[1])
        pos_train_x = [line.split() for line in trainfile]
        labels += [1] * pos_train_total

        for point in pos_train_x:
            point.append(1)
            for d in range(pos_dimensions):
                point[d] = float(point[d])
    trainfile.close()

    with open(neg_train_file, 'r') as trainfile:
        total_and_dimensions = trainfile.readline().split()
        neg_train_total = int(total_and_dimensions[0])
        neg_dimensions = int(total_and_dimensions[1])
        neg_train_x = [line.split() for line in trainfile]
        labels += [-1] * neg_train_total

        for point in neg_train_x:
            point.append(1)
            for d in range(pos_dimensions):
                point[d] = float(point[d])
    trainfile.close()

    with open(pos_test_file, 'r') as testfile:
        pos_test_total = int(testfile.readline().split()[0])
        test_true_labels += [1] * pos_test_total
        pos_test_x = [line.split() for line in testfile.readlines()]

        for point in pos_test_x:
            point.append(1)
            for d in range(pos_dimensions):
                point[d] = float(point[d])
    testfile.close()

    with open(neg_test_file, 'r') as testfile:
        neg_test_total = int(testfile.readline().split()[0])
        test_true_labels += [-1] * neg_test_total
        neg_test_x = [line.split() for line in testfile.readlines()]

        for point in neg_test_x:
            point.append(1)
            for d in range(pos_dimensions):
                point[d] = float(point[d])
    testfile.close()

    # size of training data
    train_size = pos_train_total + neg_train_total
    # initialize alphas to all 0's for each pt
    alphas = [0.0] * train_size
    converged = False
    y = labels      # true labels of training data
    x = pos_train_x + neg_train_x

    train_kernel_matrix = create_kernel_matrix(sigma, x, x)
    #print train_kernel_matrix

    # building the perceptron line using training data only
    while converged is False:
        converged = True
        #print 'train_size', train_size
        for i in range(train_size):
            # print decision(sigma, i, alphas, x, y)
            # if decision(sigma, i, alphas, x, y) <= 0:
            if decision(i, alphas, x, y, train_kernel_matrix) <= 0:
                alphas[i] += 1
                converged = False

    # now that we built the perceptron line, try it on testing data
    test_x = pos_test_x + neg_test_x

    P, N, FP, TP, TN, FN = 0,0,0,0,0,0
    for i in range(len(test_x)):
        prediction = 0.0
        for j in range(train_size):
            prediction += alphas[j] * y[j] * gaussian(sigma, test_x[i], x[j])
        #print 'pred', prediction
        if prediction > 0:
            P += 1
            if test_true_labels[i] is not 1:
                FP += 1
            else:
                TP += 1
        else:
            N += 1
            if test_true_labels[i] is not -1:
                FN += 1
            else:
                TN += 1

    print "Alphas: " + ' '.join(str(int(a)) for a in alphas)
    print "False positives: ", FP
    print "False negatives: ", FN
    print "Error rate: ", str(int(float(FP + FN) / float(P + N) * 100)) + '%'


kernel_perceptron(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])