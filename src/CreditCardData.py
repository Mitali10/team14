import numpy as np
from sklearn.model_selection import train_test_split

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


