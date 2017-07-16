import sys
import numpy as np
import math
from numpy import array

# python boostit.py 5 train.1.pos.txt train.1.neg.txt test.1.pos.txt test.1.neg.txt
#  python boostit.py T train_pos train_neg test_pos test_neg

ensemble_size, train_pos, train_neg, test_pos, test_neg = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
train_pos_total = int(open(train_pos, 'r').readline().split()[0])
train_neg_total = int(open(train_neg, 'r').readline().split()[0])
train_total = train_pos_total + train_neg_total

def read_file(pos_filename, neg_filename):
    """
    Reads a file and returns the data as a numpy array for each class
    & each entry is (point_x, point_y, weight)
    """
    # pos_weight = np.array([1/train_pos_total for i in range(train_pos_total)])
    # neg_weight = [1/train_neg_total for i in range(train_neg_total)]

    pos_data = np.loadtxt(pos_filename, skiprows=1)
    neg_data = np.loadtxt(neg_filename, skiprows=1)

    dim = int(open(train_neg, 'r').readline().split()[1])

    pos_data = np.insert(pos_data, dim, values=1/float(train_total), axis=1)  # insert values before column 3
    neg_data = np.insert(neg_data, dim, values=1/float(train_total), axis=1)  # insert values before column 3

    return pos_data, neg_data


def confusion_matrix(Y_true, Y_pred):
    """
    Outputs a confusion matrix by comparing true labels and predicted labels
    """
    conf_mat = np.zeros((2, 2))

    for y_t, y_p in zip(Y_true, Y_pred):
        conf_mat[int(y_p)][int(y_t)] += 1

    return conf_mat


def classifier_metrics(conf_mat, curr_class):
    """
    Computes true positive rate, false positive rate, accuracy, error rate and precision
    for a given class from a confusion matrix.
    """
    not_curr_class = [c for c in range(conf_mat.shape[0]) if c != curr_class]

    tp = conf_mat[curr_class, curr_class]
    fp = conf_mat[curr_class, not_curr_class].sum()
    fn = conf_mat[not_curr_class, curr_class].sum()
    tn = conf_mat[np.ix_(not_curr_class, not_curr_class)].sum()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = 1 - accuracy

    return fp, fn, error_rate

def weighted_mean(pos_data, neg_data, dim):
    """
    computes centroid by using the weighted mean of the passed in data

    """
    if dim == 2:
        pos_weighted_x = sum([point[0] * point[2] for point in pos_data]) / sum(point[2] for point in pos_data)
        pos_weighted_y = sum([point[1] * point[2] for point in pos_data]) / sum(point[2] for point in pos_data)

        neg_weighted_x = sum([point[0] * point[2] for point in neg_data]) / sum(point[2] for point in neg_data)
        neg_weighted_y = sum([point[1] * point[2] for point in neg_data]) / sum(point[2] for point in neg_data)

        # print 'centroid', [pos_weighted_x, pos_weighted_y], [neg_weighted_x, neg_weighted_y]
        return [pos_weighted_x, pos_weighted_y], [neg_weighted_x, neg_weighted_y]

    elif dim == 3:
        pos_weighted_x = sum([point[0] * point[3] for point in pos_data]) / sum(point[3] for point in pos_data)
        pos_weighted_y = sum([point[1] * point[3] for point in pos_data]) / sum(point[3] for point in pos_data)
        pos_weighted_z = sum([point[2] * point[3] for point in pos_data]) / sum(point[3] for point in pos_data)

        neg_weighted_x = sum([point[0] * point[3] for point in neg_data]) / sum(point[3] for point in neg_data)
        neg_weighted_y = sum([point[1] * point[3] for point in neg_data]) / sum(point[3] for point in neg_data)
        neg_weighted_z = sum([point[2] * point[3] for point in neg_data]) / sum(point[3] for point in neg_data)

        return [pos_weighted_x, pos_weighted_y, pos_weighted_z], [neg_weighted_x, neg_weighted_y, neg_weighted_z]


#########################################      BEGIN BOOSTING ALGORITHM     ###################################################################

# READ TRAINING FILE DATA AND TEST FILE DATA
train_pos_data, train_neg_data = read_file(train_pos, train_neg)
dim = int(open(train_pos, 'r').readline().split()[1])

test_pos_data, test_neg_data = read_file(test_pos, test_neg)
test_pos_total = int(open(test_pos, 'r').readline().split()[0])
test_neg_total = int(open(test_neg, 'r').readline().split()[0])

true_test_labels = [1 for i in range(len(train_pos_data))] + [0 for i in range(len(train_neg_data))]

# alphas - array of values that represent the confidence (alpha) of each classifier
# updated_classifier_result - a list of lists, where each list represents the result of
#       a classifier, and the result is the score (np.inner(point, W) - T) for each test point
alphas = []
updated_classifier_result = []

for i in range(ensemble_size):
    predictions = []
    scores = []

    # get centroids and convert to normal array instead of numpy
    train_pos_exemplar, train_neg_exemplar = weighted_mean(train_pos_data, train_neg_data, dim)
    train_pos_exemplar = array(train_pos_exemplar)
    train_neg_exemplar = array(train_neg_exemplar)

    # compute W and T vectors
    W = (train_pos_exemplar - train_neg_exemplar)
    T = 0.5 * np.inner((train_pos_exemplar + train_neg_exemplar), (train_pos_exemplar - train_neg_exemplar))

    for point in train_pos_data:
        point = point[:-1]
        score = np.inner(point, W) - T
        if score > 0:
            predictions.append(1)
        else:
            predictions.append(0)

    for point in train_neg_data:
        point = point[:-1]
        score = np.inner(point, W) - T
        if score > 0:
            predictions.append(1)
        else:
            predictions.append(0)

    #find misclassified points and sum their weights
    sum_misclassified_weights = 0.0

    for x in range(len(train_pos_data)):
        if predictions[:train_pos_total][x] != true_test_labels[x]:
            sum_misclassified_weights += train_pos_data[x][-1]

    for x in range(len(train_neg_data)):
        if predictions[train_pos_total:][x] != true_test_labels[train_pos_total:][x]:
            sum_misclassified_weights += train_neg_data[x][-1]

    error = sum_misclassified_weights


    if error >= 0.5:
        ensemble_size = i - 1
        break

    # find misclassified points and update their weights
    for x in range(len(train_pos_data)):
        if predictions[:train_pos_total][x] != true_test_labels[x]:
            train_pos_data[x][-1] /= (2 * error)
        else:
            train_pos_data[x][-1] /= 2*(1 - error)

    for x in range(len(train_neg_data)):
        if predictions[train_pos_total:][x] != true_test_labels[train_pos_total:][x]:
            train_neg_data[x][-1] /= (2 * error)
        else:
            train_neg_data[x][-1] /= 2 * (1 - error)

    for point in test_pos_data:
        point = point[:-1]
        scores.append(np.inner(point, W) - T)

    for point in test_neg_data:
        point = point[:-1]
        scores.append(np.inner(point, W) - T)

    # scores = list of positive test pt's predictions, and neg. test pt's predictions
    # since scores is representative of the results for current  classifier, append to updated_classifier_result
    updated_classifier_result.append(scores)

    # append alpha/confidence for current classifier
    alphas.append(0.5 * math.log((1-error) / error))

    print "Iteration", i+1
    print "Error = %.2f" % error
    print "Alpha = %.2f" % alphas[i]
    print "Factor to increase weights = %.4f" % (1 / float(2 * error))
    print "Factor to decrease weights = %.4f" % (1 / float(2 * (1 - error)))
    print ' '

# classify test points by
#       1. compute average result M(x) from ensemble of classifiers
#       2. if M(x) < 0 then negative prediction else positive prediction

test_predictions = []
test_pos_data, test_neg_data = read_file(test_pos, test_neg)

FP, FN = 0, 0

#TODO: check this part [jth classsifer offset?]
for ithPoint in range(len(test_pos_data)):
    sum = 0.0
    for jthClassifier in range(ensemble_size):
        # print 'alpha:', alphas[jthClassifier]
        # print 'score', updated_classifier_result[jthClassifier][ithPoint]
        # print ' '
        x = 1 if updated_classifier_result[jthClassifier][ithPoint] > 0 else -1
        sum += alphas[jthClassifier] * x
    test_predictions.append(sum)

for ithPoint in range(len(test_neg_data)):
    sum = 0.0
    for jthClassifier in range(ensemble_size):
        x = 1 if updated_classifier_result[jthClassifier][test_pos_total:][ithPoint] > 0 else -1
        sum += alphas[jthClassifier] * x
    test_predictions.append(sum)

#print test_predictions

for pred in test_predictions[:test_pos_total]:
    if pred <= 0:
        # predicted neg, so false neg
        FN += 1

for pred in test_predictions[test_pos_total:]:
    if pred > 0:
        # predicted pos, so false pos
        FP += 1

print "Testing:"
print "False positives: ", FP
print "False Negatives: ", FN
print "Error rate: %.2f" % (100 * (FP + FN) / float(test_pos_total + test_neg_total)) + '%'

