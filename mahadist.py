import sys
import math
import numpy as np
from numpy.linalg import inv

trainfile = sys.argv[1]
testfile = sys.argv[2]

# Pseudocode:
# 1. Compute covariance matrix = 1/k * scatter matrix
#       - first compute scatter matrix from training_file:
#           - X_z: data matrix minus mean matrix
#           - scatter matrix = X_z dot Transpose[X_z]
#       - divide by k, the # of data points
# 2. For each data point, get data point minus centroid matrix and compute:
#       - (data_point_i_matrix - CENTROID_matrix)
#       - Transpose[(data_point_i_matrix - CENTROID_matrix)]
# 3. Dot the covariance matrix, and the two results from step 3 together
# 4. Take square root and that is the mahal distance for data point i; Repeat for all points

def mahalanobis(training_file, testing_file):
    with open(training_file, 'r') as trainfile:
        total_and_dimensions = trainfile.readline().split()
        train_total = int(total_and_dimensions[0])
        dimensions = int(total_and_dimensions[1])
    #print train_total, dimensions
    #print labels

    # 1. compute epsilon: 1/k * X_z dot transpose(X_z) from training data
        train_matrix_x = [line.split() for line in trainfile]
    trainfile.close()

    with open(testing_file, 'r') as testfile:
        total_and_dimensions = testfile.readline().split()
        test_total = int(total_and_dimensions[0])
        test_matrix_x = [line.split() for line in testfile]
    testfile.close()

    for point in test_matrix_x:
        for d in range(dimensions):
            point[d] = float(point[d])
    test_matrix_x = np.asmatrix(test_matrix_x)

    # get training_file's data into matrix format and extract centroid
    centroid = [0.0] * dimensions
    for point in train_matrix_x:
        for d in range(dimensions):
            point[d] = float(point[d])
            centroid[d] += point[d]
    train_matrix_x = np.asmatrix(train_matrix_x)
    centroid = [value / train_total for value in centroid]

    #print train_matrix_x
    #print centroid
    train_matrix_x_z = np.transpose(train_matrix_x - np.asmatrix([centroid] * train_total))
    #print train_matrix_x_z
    transposed_train_matrix_x_z = np.transpose(train_matrix_x_z)
    #print transposed_train_matrix_x_z
    scatter_matrix = train_matrix_x_z.dot(transposed_train_matrix_x_z)
    #print scatter_matrix
    covariance_matrix = scatter_matrix / train_total
    #print covariance_matrix

    # 1.2 compute inverse of epsilon/covariance matrix
    inverse_covariance_matrix = inv(covariance_matrix)

    # 2. for each pt, find mahal distance
    mahal_results = []
    #print test_matrix_x
    #print(covariance_matrix)
    for i in range(len(test_matrix_x)):
        result = math.sqrt(
            np.transpose(np.transpose(test_matrix_x[i] - centroid)).dot(inverse_covariance_matrix).dot(np.transpose(test_matrix_x[i] - centroid)))
        mahal_results.append(result)

    print "Centroid:"
    print ' '.join("%.2f" % x for x in centroid)
    print "Covariance Matrix:"
    for row in np.array(covariance_matrix):
        print ' '.join("%.2f" % x for x in row)
    print "Distances:"
    test_matrix_x = np.array(test_matrix_x)
    for i in range(len(test_matrix_x)):
        print str(i+1) + ". " \
              + ' '.join("%.2f" % x for x in test_matrix_x[i]) \
              + " -- " + ("%.2f" % mahal_results[i])


mahalanobis(trainfile, testfile)






