import sys
import math
import itertools

# Euc distance: square_root[ (p1-q1)^2 + (p2-q2)^2 + ... (pn - qn)^2 ]
def euclidean_distance(trained_point, test_point):
    sum = 0.0
    for i in range(len(trained_point)):
        sum += (trained_point[i] - test_point[i]) ** 2
    return math.sqrt(sum)

    # euclidean_label = {}
    #
    # # find distance from testing point to each labeled training point
    # # euclidean is a dictionary with:
    # # key= (euc_distance_to_training_pt_1, true_label_of_training_point_1)
    # # value = count
    # for trained_point in trained_list:
    #     for i in range(len(test_point)):
    #         sum += (test_point[i] - trained_point[i]) ** 2
    #     tuple = (math.sqrt(sum), trained_point[len(trained_point)-1])
    #     euclidean_label[tuple] = euclidean_label.get(tuple, 0) + 1
    #
    # euclidean_label = sorted(euclidean_label.iteritems())

    #highest_count = euclidean_label


    # case 1: vote tie is when 2+ training points are the same distance to the test point
    # solution:

    #return euclidean_label



def knn(k, training_file, testing_file):
    with open(training_file, 'r') as trainfile:
        total_and_dimensions = trainfile.readline().split()
        train_total = int(total_and_dimensions[0])
        train_dimensions = int(total_and_dimensions[1])

        training_points = [line.split() for line in trainfile]
    trainfile.close()

    with open(testing_file, 'r') as testfile:
        total_and_dimensions = testfile.readline().split()
        test_total = int(total_and_dimensions[0])
        test_dimensions = int(total_and_dimensions[1])

        testing_points = [line.split() for line in testfile]
    testfile.close()

    #print training_points
    for point in training_points:
        for d in range(train_dimensions):
            point[d] = float(point[d])
        point[train_dimensions] = int(point[train_dimensions])       # convert label to an int instead of float

    for point in testing_points:
        for d in range(test_dimensions):
            point[d] = float(point[d])

    #print training_points
    #print testing_points

    # classes_count is a dictionary where:
    # keys = label
    # value = frequency of label AFTER analyzing training data
    classes_count = {}
    for point_label in training_points:
        label = point_label[len(point_label) - 1]
        classes_count[label] = 0

    for test_point in testing_points:
        # an array of distances from test_point to each trained/labeled data point
        euclideans = []
        for train_point in training_points:
            # strip label
            label = train_point[-1]
            train_point = train_point[:-1]
            euclideans.append([euclidean_distance(train_point, test_point), label])
            #print euclidean_distance(train_point, test_point)

        euclideans = sorted(euclideans)
        #print euclideans

        # grab k nearest neighbors and find the most popular class
        k_nearest = []
        # print 'testing point: ', test_point
        # print euclideans
        k_nearest = euclideans[:k]
        # print 'k nearest'
        # print k_nearest

        most_popular_classes = []
        for neighbor in k_nearest:
            label = neighbor[-1]
            classes_count[label] = classes_count.get(label, 0) + 1
        # print 'class count'
        # print classes_count

        most_popular_classes.append(max(classes_count, key=classes_count.get))
        for key, value in classes_count.items():
            #print 'matching,', classes_count.get(most_popular_classes[0])
            if classes_count.get(key) == classes_count.get(most_popular_classes[0]) and key is not most_popular_classes[0]:
                #print key, classes_count.get(key), 'matches', classes_count.get(most_popular_classes[0])
                most_popular_classes.append(key)
        # print 'most popular classes'
        # print most_popular_classes

        if len(most_popular_classes) == 1:
            # only one winner
            test_point.append(most_popular_classes[0])
        else:
            # choose point that is closest
            # euclideans[0] is closest distance, and euclideans[1] is the label of that trained_point
            # case 1: vote tie:
            euclideans = sorted([x for x in euclideans if x[-1] in most_popular_classes])
            if euclideans[0][0] == euclideans[1][0]:
                #case 2: distance tie, choose smaller label
                test_point.append(euclideans[0][1] if euclideans[0][1] < euclideans[1][1] else euclideans[1][1])
            else:
                # if no distance tie, use label of smallest distance
                test_point.append(euclideans[0][1])

        # reset count for each class
        for key,value in classes_count.items():
            classes_count[key] = 0

    count = 1
    for item in testing_points:
        print str(count) + '. ' + (' ').join("%.1f" % x for x in item[:-1]) + ' -- ' + str(item[-1])
        count += 1


knn(int(sys.argv[1]), sys.argv[2], sys.argv[3])

