import sys
import math

training_file = sys.argv[1]
testing_file = sys.argv[2]

# step 1) parse training data
training_data = [line.strip() for line in open(training_file)]

dimension = int(training_data[0].split()[0])
A_size = int(training_data[0].split()[1])
A_class = training_data[1:1+A_size]

B_size = int(training_data[0].split()[2])
B_class = training_data[2+A_size:2+A_size+B_size]

C_size = int(training_data[0].split()[3])
C_class = training_data[2+A_size+B_size:-1]


# step 2) calculate centroids for class A, B, C
def centroid(class_data, dimensionality, size):
    # takes an array of data for ONE class
    # class data: feature_1_value, feature_2_value ...
    mean_vector = []
    for col in range(dimensionality):
        mean_vector.append(sum(float(data.split()[col]) for data in class_data)/size)

    return mean_vector

A_centroid = centroid(A_class, dimension, A_size)
B_centroid = centroid(B_class, dimension, B_size)
C_centroid = centroid(C_class, dimension, C_size)

# print(A_centroid, B_centroid, C_centroid)
#

# step 3) find disc. line between A&B, A&C, B&C using centroids
# discriminant line = (P - N) dot (P + N)/2
#   where P is centroid for positive class, N for neg
# dot product: sum([i*j for (i, j) in zip(list1, list2)])

def old_discriminant_line(positive, negative):
    p_minus_n = []
    p_plus_n_over_2 = []
    if positive == "A":
        if negative == "B":
            p_minus_n = [i-j for (i, j) in zip(A_centroid, B_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(A_centroid, B_centroid)]
        else:
            p_minus_n = [i-j for (i, j) in zip(A_centroid, C_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(A_centroid, C_centroid)]
    elif positive == "B":
        if negative == "A":
            p_minus_n = [i-j for (i, j) in zip(B_centroid, A_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(B_centroid, A_centroid)]
        else:
            p_minus_n = [i-j for (i, j) in zip(B_centroid, C_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(B_centroid, C_centroid)]
    elif positive == "C":
        if negative == "A":
            p_minus_n = [i-j for (i, j) in zip(C_centroid, A_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(C_centroid, A_centroid)]
        else:
            p_minus_n = [i-j for (i, j) in zip(C_centroid, B_centroid)]
            p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(C_centroid, B_centroid)]

    return sum([i*j for (i, j) in zip(p_minus_n, p_plus_n_over_2)])

    # if (positive == "A" and negative == "B") or (positive == "B" and negative == "A"):
    #     p_plus_n_over_2 = [(i+j)/2 for (i, j) in zip(A_centroid, B_centroid)]
    # elif (positive == "A" and negative == "C") or (positive == "C" and negative == "A"):
    #     p_plus_n_over_2 = [(i + j) / 2 for (i, j) in zip(A_centroid, C_centroid)]
    # else:
    #     p_plus_n_over_2 = [(i + j) / 2 for (i, j) in zip(B_centroid, C_centroid)]

    # return dot product of p_minus_n and p_plus_n_over_2

def magnitude(data_point):
    return math.sqrt(sum(i ** 2 for i in data_point))

def discriminant_line(pos_centroid, neg_centroid):
    return (magnitude(pos_centroid) ** 2 - magnitude(neg_centroid) ** 2) / 2.0
    # p_minus_n = []
    # p_plus_n_over_2 = []
    # if positive == "A":
    #     if negative == "B":
    #         p_minus_n = [i-j for (i, j) in zip(A_centroid, B_centroid)]
    #         p_plus_n_over_2 = [(i+j)/2.0 for (i, j) in zip(A_centroid, B_centroid)]
    #     else:
    #         p_minus_n = [i-j for (i, j) in zip(A_centroid, C_centroid)]
    #         p_plus_n_over_2 = [(i+j)/2.0 for (i, j) in zip(A_centroid, C_centroid)]
    # elif positive == "B":
    #         p_minus_n = [i-j for (i, j) in zip(B_centroid, C_centroid)]
    #         p_plus_n_over_2 = [(i+j)/2.0 for (i, j) in zip(B_centroid, C_centroid)]

    # return sum([i*j for (i, j) in zip(p_minus_n, p_plus_n_over_2)])

A_B_disc = discriminant_line(A_centroid, B_centroid)
A_C_disc = discriminant_line(A_centroid, C_centroid)
B_C_disc = discriminant_line(B_centroid, C_centroid)

# print('A_B_disc', A_B_disc)
# print('A_C_disc', A_C_disc)

def find_orthogonal_w(pos_centroid, neg_centroid):
    return [i - j for (i, j) in zip(pos_centroid, neg_centroid)]

    # if positive == "A":
    #     if negative == "B":
    #         return [i-j for (i, j) in zip(A_centroid, B_centroid)]
    #     else:
    #         return [i - j for (i, j) in zip(A_centroid, C_centroid)]
    # else: # only B, C left
    #     return [i - j for (i, j) in zip(B_centroid, C_centroid)]



# for data in training_data[1:]:
#     # find data x dot w
#     # w = p - n = positive_centroid - negative_centroid
#     score = sum([i*j for (i, j) in zip(p_minus_n, p_plus_n_over_2)])
#

def classify(data_point, wab, wac, wbc):
    dp = data_point.split()
    for i in range(len(dp)):
        dp[i] = float(dp[i])

    dot_product_A_B = sum([i*j for (i, j) in zip(dp, wab)])
    dot_product_A_C = sum([i*j for (i, j) in zip(dp, wac)])
    dot_product_B_C = sum([i * j for (i, j) in zip(dp, wbc)])

    # decide A or B
    if dot_product_A_B - A_B_disc > 0:
        # between A or B, it's A. So check between A or C
        if dot_product_A_C - A_C_disc > 0:
            return 1
        else:
            return 3
    else:
        # between A or B, it's B. So check between B or C
        if dot_product_B_C - B_C_disc > 0:
            return 2
        else:
            return 3

predictions = {}
def evaluate(test_file):
    # find centroid of positive class and negative class in test file to compute w = p-n
    testing_data = [line.strip() for line in open(test_file)]
    dimension_test = int(testing_data[0].split()[0])
    A_size_test = int(testing_data[0].split()[1])
    B_size_test = int(testing_data[0].split()[2])
    C_size_test = int(testing_data[0].split()[3])
    testing_data = testing_data[1:]

    A_class_test = training_data[:A_size_test-1]
    B_class_test = training_data[A_size_test:A_size_test + B_size_test - 1]
    C_class_test = training_data[A_size_test + B_size_test:]

    A_test_centroid = centroid(A_class_test, dimension_test, A_size_test)
    B_test_centroid = centroid(B_class_test, dimension_test, B_size_test)
    C_test_centroid = centroid(C_class_test, dimension_test, C_size_test)

    w_A_B = find_orthogonal_w(A_centroid, B_centroid)
    w_A_C = find_orthogonal_w(A_centroid, C_centroid)
    w_B_C = find_orthogonal_w(B_centroid, C_centroid)

    # TODO: WRONG HERE, maybe fixed
    # print ("wab", w_A_B_test)
    # print ("wac", w_A_C_test)
    # print ("wbc", w_B_C_test)

    # count no. of correctly predicted positive A's
    # print(A_class_test)
    TP_A, TP_B, TP_C = 0.0, 0.0, 0.0
    FP_A, FP_B, FP_C = 0.0, 0.0, 0.0
    FN_A, FN_B, FN_C = 0.0, 0.0, 0.0
    TN_A, TN_B, TN_C = 0.0, 0.0, 0.0
    P_est_A, P_est_B, P_est_C = 0.0, 0.0, 0.0
    N_est_A, N_est_B, N_est_C = 0.0, 0.0, 0.0


    TPR_A, TPR_B, TPR_C = 0.0, 0.0, 0.0
    FPR_A, FPR_B, FPR_C = 0.0, 0.0, 0.0
    index = 0
    for data_point in testing_data:
        score = classify(data_point, w_A_B, w_A_C, w_B_C)
        # predicted class A
        if score == 1:
            P_est_A += 1
            N_est_B += 1
            N_est_C += 1
            # is actually A
            if index < A_size_test:
                TP_A += 1
                TN_B += 1
                TN_C += 1
            elif index >= A_size_test and index < A_size_test + B_size_test:
                FP_A += 1
                FN_B += 1
                TN_C += 1
            else:
                FP_A += 1
                FN_C += 1
                TN_B += 1

        # predicted B
        elif score == 2:
            N_est_A += 1
            P_est_B += 1
            N_est_C += 1
            if index < A_size_test:
                FP_B += 1
                FN_A += 1
                TN_C += 1
            elif index >= A_size_test and index < A_size_test + B_size_test:
                TP_B += 1
                TN_A += 1
                TN_C += 1
            else:
                FP_B += 1
                FN_C += 1
                TN_A += 1
        # predicted C
        elif score == 3:
            N_est_A += 1
            N_est_B += 1
            P_est_C += 1
            if index < A_size_test:
                FP_C += 1
                FN_A += 1
                TN_B += 1
            elif index >= A_size_test and index < A_size_test + B_size_test:
                FP_C += 1
                FN_B += 1
                TN_A += 1
            else:
                TP_C += 1
                TN_A += 1
                TN_B += 1
        index += 1

    # class A
    TPR_A = TP_A / float(A_size_test)
    FPR_A = FP_A / (float(B_size_test) + float(C_size_test))
    error_A = (FP_A + FN_A) / (P_est_A + N_est_A)
    acc_A = (TP_A + TN_A) / (P_est_A + N_est_A)
    pre_A = TP_A / P_est_A

    # class B
    TPR_B = TP_B / float(B_size_test)
    FPR_B = FP_B / (float(A_size_test) + float(C_size_test))
    error_B = (FP_B + FN_B) / (P_est_B + N_est_B)
    acc_B = (TP_B + TN_B) / (P_est_B + N_est_B)
    pre_B = TP_B / P_est_B

    # class c
    TPR_C = TP_C / float(C_size_test)
    FPR_C = FP_C / (float(A_size_test) + float(B_size_test))
    error_C = (FP_C + FN_C) / (P_est_C + N_est_C)
    acc_C = (TP_C + TN_C) / (P_est_C + N_est_C)
    pre_C = TP_C / P_est_C



    #results
    print 'True positive rate =', (TPR_A + TPR_B + TPR_C) / 3.0
    print 'False positive rate =', (FPR_A + FPR_B + FPR_C) / 3.0
    print "Error rate = ", (error_A + error_B + error_C) / 3.0
    print "Accuracy = ", (acc_A + acc_B + acc_C) / 3.0
    print "Precision = ", (pre_A + pre_B + pre_C) / 3.0


evaluate(testing_file)

