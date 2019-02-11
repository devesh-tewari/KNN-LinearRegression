import pandas as pd
import numpy as np
import sys

csv_path = 'AdmissionDataset/data.csv'#raw_input("Enter path to input CSV file: ")
dataset = pd.read_csv(csv_path)

dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

#split data into train data and validation data
splitted = np.split(dataset, [int(.8 * len(dataset.index))])
train_data = splitted[0]
validation_data = splitted[1]


Attributes = dataset.keys()[[0,1,2,3,4,5,6]]
Label = dataset.keys()[7]

for att in Attributes:
    mean = np.mean(train_data[att].values)
    std = np.std(train_data[att].values)
    train_data[att] = (train_data[att]-mean)/(std)

for att in Attributes:
    mean = np.mean(validation_data[att].values)
    std = np.std(validation_data[att].values)
    validation_data[att] = (validation_data[att]-mean)/(std)


att_data = train_data[Attributes]
label_data = train_data[Label]

X = att_data.values
Y = label_data.values

X = np.array(X)
extra_col = np.ones([X.shape[0],1])
X = np.concatenate((extra_col,X),axis=1)

theta = np.matmul( np.matmul( np.linalg.inv(np.matmul(X.T, X)), X.T ) , Y )

print ('theta: '+str(theta))


def predict(row):

    x = row.values
    x = x.reshape([1,7])
    extra_col = np.ones([1,1])
    x = np.concatenate((extra_col,x),axis=1)

    y = np.dot(x, theta)

    return float(y)



def calculate_performance():

    print ('\n\n\nPerformance Measures\n\n')

    MSE = 0.0
    MAE = 0.0
    MPE = 0.0
    for i in range(len(validation_data)):
        row = validation_data.iloc[[i]]

        y_predicted = predict(validation_data.iloc[[i]][Attributes])

        y_actual = float(row[Label])

        MSE += (y_predicted-y_actual)**2
        MAE += abs(y_predicted-y_actual)
        MPE += (y_actual-y_predicted)/y_actual

    MSE /= len(validation_data)
    MAE /= len(validation_data)
    MPE /= len(validation_data)
    MPE *= 100

    print ('Mean Squared Error:')
    print (MSE)
    print ('\nMean Absolute Error:')
    print (MAE)
    print ('\nMean Percentage Error:')
    print (MPE)


calculate_performance()
