import pandas as pd
import numpy as np
import sys

csv_path = 'LoanDataset/data.csv'#raw_input("Enter path to input CSV file: ")
dataset = pd.read_csv(csv_path, header=None)

dataset = dataset.iloc[1:]
dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

#split data into train data and validation data
splitted = np.split(dataset, [int(.8 * len(dataset.index))])
train_data = splitted[0]
validation_data = splitted[1]


X = dataset.keys()[[0,1,2,3,4,5,6,7,9,10,11,12]]
Y = dataset.keys()[8]


def is_numeric(att):
    if att in dataset.keys()[[0,1,2,4,5,7]]:
        return True
    else:
        return False

mean = {}
sd = {}

for x in X:
#     if is_numeric(x):
    accept = train_data[train_data[Y] == 1][x]
    reject = train_data[train_data[Y] == 0][x]

    mean[x] = [np.mean(reject), np.mean(accept)]
    sd[x] = [np.std(reject), np.std(accept)]


# import scipy.stats
import math

y1_filtered = train_data[train_data[Y] == 1]
y0_filtered = train_data[train_data[Y] == 0]
p_y1 = float(len(y1_filtered)) / len(train_data)
p_y0 = 1 - p_y1

def predict(row):

    p_accept = 1
    p_reject = 1

    for x in X:

        row_val = row[x].values[0]

#         filtered = train_data[train_data[x] == row_val]
#         p_x = float(len(filtered)) / len(train_data)

#         p_XbyY = scipy.stats.norm(mean[x][1], sd[x][1]).pdf(0)
#         p_accept *= p_XbyY * p_y1 / p_x

#         p_XbyY = scipy.stats.norm(mean[x][0], sd[x][0]).pdf(0)
#         p_reject *= p_XbyY * p_y0 / p_x

#         if is_numeric(x):
        exponent = math.exp(-(math.pow(row_val-mean[x][0],2)/(2*math.pow(sd[x][0],2))))
        p_reject *= ( 1 / (math.sqrt(2*math.pi)*sd[x][0]) ) * exponent

        exponent = math.exp(-(math.pow(row_val-mean[x][1],2)/(2*math.pow(sd[x][1],2))))
        p_accept *= ( 1 / (math.sqrt(2*math.pi)*sd[x][1]) ) * exponent

#         else:

#             filtered = y1_filtered[y1_filtered[x] == row_val]
#             required = len(filtered)

#             p_accept *= float(required) / len(y1_filtered)

#             filtered = y0_filtered[y0_filtered[x] == row_val]
#             required = len(filtered)

#             p_reject *= float(required) / len(y0_filtered)

#     p_accept *= p_y1
#     p_reject *= p_y0

#     print p_accept, p_reject

    if p_accept > p_reject:
        return 1
    else:
        return 0


def calculate_performance(validation_data):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(validation_data)):
        row = validation_data.iloc[[i]][Y]
        row = row.tolist()
        row = row[0]
        if predict(validation_data.iloc[[i]]) == 1:
            if row == 1:
                TP += 1
            else:
                FP += 1
        else:
            if row == 0:
                TN += 1
            else:
                FN += 1

    accuracy = float(TP + TN) / (TP + TN + FP + FN)

    if TP + FP == 0:
        precision = 0
    else:
        precision = float(TP) / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = float(TP) / (TP + FN)

    if precision == 0 or recall == 0:
        F1measure = 0
    else:
        F1measure = 2.0 / ( (1/recall) + (1/precision) )

#     print TP, TN, FP, FN
    print ("Validation Results:\n")
    print ("accuracy = " + str(accuracy))
    print ("precision = " + str(precision))
    print ("recall = " + str(recall))
    print ("F1 measure = " + str(F1measure))


calculate_performance(validation_data)
