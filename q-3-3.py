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


def predict(row):

    x = row.values
    x = x.reshape([1,7])
    extra_col = np.ones([1,1])
    x = np.concatenate((extra_col,x),axis=1)

    y = np.dot(x, theta)

    return float(y)



from matplotlib import pyplot as plt

def plot_residuals(feature):

    feature_index = validation_data.columns.get_loc(feature)
    theta_feature = theta[feature_index+1]

    residue = []
    feature_val = []

    for index, row in validation_data.iterrows():
        y_predicted = theta_feature * float(row[feature]) + theta[0]
        y_actual = float(row[Label])

        residue.append(predict(row[Attributes])-y_actual)
#         denormalized_val = row[feature]*np.std(dataset[feature].values)+np.mean(dataset[feature].values)
        feature_val.append(row[feature])

    plt.figure(figsize=(12,8))
    plt.plot(feature_val, residue,  marker='o', linewidth = 3, linestyle=' ')
    plt.xlabel(str(feature)+' (normalized)')
    plt.ylabel('Residues')
    plt.title('Residue Plot of feature \''+str(feature)+'\'')
    plt.axhline(0, color='brown', lw=1)
    plt.show()

plot_residuals('GRE Score')
plot_residuals('SOP')
plot_residuals('Research')
