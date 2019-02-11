import pandas as pd
import numpy as np
import sys
from math import sqrt

csv_path = raw_input("Enter path to input CSV file: ")
dataset = pd.read_csv(csv_path, delimiter=' ', header=None)

dataset = dataset.dropna(axis=1, how='all')

#split data into train data and validation data
splitted = np.split(dataset, [int(.8 * len(dataset.index))])
train_data = splitted[0].reset_index()
validation_data = splitted[1].reset_index()

X = dataset.keys()[[1,2,3,4,5,6]]
Y = dataset.keys()[0]
ID = dataset.keys()[7]

from heapq import *

def heappeek(heap):
    largest = heappop(heap)
    heappush(heap, largest)
    return largest[0]

def k_nearest_majority(dist_list, k):
#     print "\n\n\n\n"
#     for dist in dist_list:
#         print dist[0]
#     print "\n"
    max_heap = []

    i = 0
    while i < k:
        heappush( max_heap, (dist_list[i][0], dist_list[i][1]) )
        i += 1

    while i < len(dist_list):
        if dist_list[i][0] > heappeek(max_heap):
            heappop(max_heap)
            heappush(max_heap, (dist_list[i][0], dist_list[i][1]))
        i += 1

    zero_count, one_count = 0, 0
    try:
        while True:
            largest = heappop(max_heap)
#             print largest
            if largest[1] == 0:
                zero_count += 1
            else:
                one_count += 1
    except IndexError:
        pass

    if zero_count > one_count:
        return 0
    else:
        return 1


def euclidean_nearest(inst, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0.0
        for att in X:
            dist += ( row[att] - inst[att] ) ** 2
        dist = sqrt(dist)
        distances.append((-1*dist, row[Y]))

    op = k_nearest_majority(distances, k)

    return op

def minkowski_nearest(inst, p, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0.0
        for att in X:
            dist += ( abs(row[att] - inst[att]) ) ** p
        dist = dist ** (1.0/p)
        #print distances
        distances.append((-1*dist, row[Y]))
        #print distances

    op = k_nearest_majority(distances, k)

    return op

def manhattan_nearest(inst, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0.0
        for att in X:
            dist += abs( row[att] - inst[att] )
        distances.append((-1*dist, row[Y]))

    op = k_nearest_majority(distances, k)

    return op

def chebychev_nearest(inst, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0.0
        for att in X:
            dist = max(dist, abs(row[att] - inst[att]) )

        distances.append((-1*dist, row[Y]))

    op = k_nearest_majority(distances, k)

    return op

def cosine_nearest(inst, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0.0
        mod_row = 0.0
        mod_inst = 0.0
        for att in X:
            dist += row[att] * inst[att]
            mod_row += row[att]**2
            mod_inst += inst[att]**2
        mod_row = sqrt(mod_row)
        mod_inst = sqrt(mod_inst)
        dist = dist / (mod_row * mod_inst)
        dist = 1 - dist
        distances.append((-1*dist, row[Y]))

    op = k_nearest_majority(distances, k)

    return op

def hamming_nearest(inst, k):
    distances = []
    op = 0
    for index, row in train_data.iterrows():
        dist = 0
        for att in X:
            if row[att] != inst[att]:
                dist += 1
        distances.append((-1*dist, row[Y]))

    op = k_nearest_majority(distances, k)

    return op


def KNN_classifier(new_data, measure, k):

    predictions = []

    if measure == 'euclidean':
        for index, row in new_data.iterrows():
            predictions.append(euclidean_nearest(row, k))

    elif measure == 'minkowski':
        for index, row in new_data.iterrows():
            predictions.append(minkowski_nearest(row, 3, k))

    elif measure == 'manhattan':
        for index, row in new_data.iterrows():
            predictions.append(manhattan_nearest(row, k))

    elif measure == 'chebychev':
        for index, row in new_data.iterrows():
            predictions.append(chebychev_nearest(row, k))

    elif measure == 'cosine':
        for index, row in new_data.iterrows():
            predictions.append(cosine_nearest(row, k))

    elif measure == 'hamming':
        for index, row in new_data.iterrows():
            predictions.append(hamming_nearest(row, k))

    TP, TN, FP, FN = 0, 0, 0, 0

    i = 0
    for index, row in new_data.iterrows():
        if predictions[i] == 1:
            if row[Y] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if row[Y] == 0:
                TN += 1
            else:
                FN += 1
        i += 1

    return TP, FP, TN, FN



from matplotlib import pyplot as plt

def plot_wrt_k():

    plt.figure(figsize=(15,10))

    k_arr = []

    accuracy_euclidean = []
    accuracy_minkowski = []
    accuracy_manhattan = []
    accuracy_chebychev = []
    accuracy_cosine = []
    accuracy_hamming = []

    k = 1
    while k < 30:

        TP, FP, TN, FN = KNN_classifier(validation_data, 'euclidean', k)
        accuracy_euclidean.append( float(TP + TN) / (TP + TN + FP + FN) )

        TP, FP, TN, FN = KNN_classifier(validation_data, 'minkowski', k)
        accuracy_minkowski.append( float(TP + TN) / (TP + TN + FP + FN) )

        TP, FP, TN, FN = KNN_classifier(validation_data, 'manhattan', k)
        accuracy_manhattan.append( float(TP + TN) / (TP + TN + FP + FN) )

        TP, FP, TN, FN = KNN_classifier(validation_data, 'chebychev', k)
        accuracy_chebychev.append( float(TP + TN) / (TP + TN + FP + FN) )

        TP, FP, TN, FN = KNN_classifier(validation_data, 'cosine', k)
        accuracy_cosine.append( float(TP + TN) / (TP + TN + FP + FN) )

        TP, FP, TN, FN = KNN_classifier(validation_data, 'hamming', k)
        accuracy_hamming.append( float(TP + TN) / (TP + TN + FP + FN) )

        k_arr.append(k)

        k += 2

    plt.plot(k_arr, accuracy_euclidean, label = "Euclidean Distance", marker='o', linewidth = 3, color='red')
    plt.plot(k_arr, accuracy_manhattan, label = "Manhattan Distance", marker='o', linewidth = 3, linestyle='--')
    plt.plot(k_arr, accuracy_chebychev, label = "Chebychev Distance", marker='o', linewidth = 3, linestyle='-.')
    plt.plot(k_arr, accuracy_cosine, label = "Cosine Distance", marker='o', linewidth = 3, linestyle=':')
    plt.plot(k_arr, accuracy_hamming, label = "Hamming Distance", marker='o', linewidth = 3, linestyle='-.')
    plt.plot(k_arr, accuracy_minkowski, label = "Minkowski Distance", marker='o', linewidth = 3, linestyle=' ')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuracy w.r.t. k in KNN')
    plt.legend()
    plt.show()

plot_wrt_k()
