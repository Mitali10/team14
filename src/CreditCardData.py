import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

import sys
# def reduce_dataset_size(from_filepath='../data/creditcard.csv', to_filepath='../data/creditcard_sampled2.csv', proportion=0.02):
#     '''
#     Input: filepath to original credicard.csv dataset file
#     Output: proportion% of the data of the file located at from_filepath is stored in to_filepath.
#     '''
#     csv_data_og = np.loadtxt(open(from_filepath, "rb"), dtype=str, delimiter=",", skiprows=1)
#     # csv_data_og = csv_data_og.astype(float)
#     idx = np.random.randint(len(csv_data_og), size=int(len(csv_data_og) *proportion))
#     csv_data_sampled = csv_data_og[idx, :]
#     np.savetxt(to_filepath, csv_data_sampled, fmt="%s")

# reduce_dataset_size()


class CreditCardData:
    '''
    Class for Credit Card Data. Can (and should) be used by all classes that handle this data. 
    '''
    def __init__(self, filepath="../data/creditcard.csv", split=False, test_prop=0.2):
        csv_data_og = np.loadtxt(open(filepath, "rb"), dtype=str, delimiter=",", skiprows=1, ndmin=2)
        self.data = np.array(csv_data_og[:, :-1]).astype(float)
        self.labels = np.array([row[-1].replace("\"", "") for row in csv_data_og]).astype(int)

        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.K = 2                      # genuine or fraud

        self.data_train, self.data_test, self.labels_train, self.labels_test = None, None, None, None
        if split:
            self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.data, self.labels, test_size=test_prop)
    
    def split_data(self, test_size = 0.2, iterations = 1): 
        X = self.data
        y = self.labels
        X_train, X_test, y_train, y_test = ([] for i in range(4))
        sss = StratifiedShuffleSplit(n_splits=iterations, test_size=test_size, random_state=0)
        sss.get_n_splits(X, y)
        for train_index, test_index in sss.split(X, y):
            X_train.append(X[train_index])
            X_test.append(X[test_index])
            y_train.append(y[train_index])
            y_test.append(y[test_index])
        return np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)
    
    
    def oversample(self, X_train, y_train): 
        X_resampled , y_resampled = ([] for i in range(2))
        for i in range(len(X_train)):
            smote = SMOTE(random_state = 42, sampling_strategy=0.1) # desired ratio minority/majority = 0.1
            X, y = smote.fit_resample(X_train[i], y_train[i])
            X_resampled.append(X)
            y_resampled.append(y)
        return np.asarray(X_resampled), np.asarray(y_resampled)

        


