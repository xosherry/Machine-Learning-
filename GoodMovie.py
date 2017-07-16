from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier
import sys

training_file = sys.argv[1]
testing_file = sys.argv[2]

def trained_decision_tree():
    train_features = []     # list of lists of features
    train_labels = []       # list of labels
    with open(training_file, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            train_features.append(line.split()[1:-1])
            train_labels.append(line.split()[-1])
    f.close()

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(train_features, train_labels)

    return dt


def predict(test_features,trained_dt):
    result = trained_dt.predict(test_features)

    return result

def analysis():
    with open(testing_file, "r") as f:
        test_features = []
        test_labels = []
        count = 0
        lines = f.readlines()[1:]
        for line in lines:
            test_features.append(line.split()[1:-1])
            test_labels.append(line.split()[-1])
            count += 1
        f.close()

    trained_dt = trained_decision_tree()
    predicted_labels = predict(test_features, trained_dt)

    TP, TN, FN, FP = 0, 0, 0, 0
    for i in range(len(predicted_labels)):
        #print "predicted, actual:", predicted_labels[i], test_labels[i]
        if predicted_labels[i] == test_labels[i]:   # correctly classified
            if test_labels[i] == "1":
                TP += 1
            else:
                TN += 1
        else:       #incorrectly classified
            if test_labels[i] == "1":
                FN += 1
            else:
                FP += 1

    with open("outputs.txt", "a") as out:
        result = "Number of movies: " + str(count) + "\nTrue Positives: " + str(TP) + \
                 "\nTrue Negatives: " + str(TN) + "\nFalse Positives: " + str(FP) + "\nFalse Negatives: " + str(FN) + \
                 "\nError rate: " + str(float(FP + FN) / float(count)) + '\n'

        print(result, file=out)
        out.close()


analysis()