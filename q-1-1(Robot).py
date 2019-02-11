import pandas as pd
import numpy as np
import sys
from math import sqrt

if len(sys.argv) < 2:
    print("Run as: python/python3 <file_name.py> <absolute_path_of_test_file>")
    sys.exit(0)

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



TP, FP, TN, FN = KNN_classifier(validation_data, 'euclidean', 9)

accuracy = float(TP + TN) / (TP + TN + FP + FN)

if TP + FP == 0:
    precision = 0
else:
    precision = float(TP) / (TP + FP)

if TP + FN == 0:
    recall = 0
else:
    recall = float(TP) / (TP + FN)

F1measure = 2.0 / ( (1/recall) + (1/precision) )

print ("\n\nValidation Results for k=9:\n")
print ("accuracy = " + str(accuracy))
print ("precision = " + str(precision))
print ("recall = " + str(recall))
print ("F1 measure = " + str(F1measure))


from sklearn.neighbors import KNeighborsClassifier
print('\n\nsklearn performance:')
for k in range(1,30,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data[X],train_data[Y])
    print ('at k = '+str(k)+', accuracy = '+str(knn.score(validation_data[X], validation_data[Y])))


def predict_test(test_set):
    print ("\nPredictions:")
    for measure in ['euclidean', 'minkowski', 'manhattan', 'chebychev', 'cosine', 'hamming']:
        for k in range(1,10,2):
            tp,fp,tn,fn = KNN_classifier(test_set, measure, k)
            accuracy = float(tp+tn)/(tp+tn+fp+fn)
            print('measure: '+measure+'\nk: '+str(k)+'\naccuracy = '+str(accuracy))
            print '\n'


csv_path = str(sys.argv[1])
test_set = pd.read_csv(csv_path, delimiter=' ', header=None)
test_set = test_set.dropna(axis=1, how='all')

predict_test(test_set)
