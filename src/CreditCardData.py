import numpy as np

class CreditCardData:
    '''
    Class for Credit Card Data. Can (and should) be used by all classes that handle this data. 
    '''
    def __init__(self, filepath="../data/creditcard_sampled.csv"):
        csv_data_og = np.loadtxt(open(filepath, "rb"), dtype=str, delimiter=",", skiprows=1, ndmin=2)
        csv_data_og =np.array([k[0].replace("\"", "").split(" ") for k in csv_data_og])


        self.data = np.array(csv_data_og[:, :-1]).astype(float)
        self.labels = np.array(csv_data_og[:, -1]).astype(int)

        self.N = self.data.shape[0]
        self.D = self.data.shape[1]
        self.K = 2                      # genuine or fraud



def reduce_dataset_size(from_filepath='../data/creditcard.csv', to_filepath='../data/creditcard_sampled.csv', proportion=0.01):
    '''
    Input: filepath to original credicard.csv dataset file
    Output: proportion% of the data of the file located at from_filepath is stored in to_filepath.
    '''
    csv_data_og = np.loadtxt(open(from_filepath, "rb"), dtype=str, delimiter=",", skiprows=1)
    # csv_data_og = csv_data_og.astype(float)
    idx = np.random.randint(len(csv_data_og), size=int(len(csv_data_og) *proportion))
    csv_data_sampled = csv_data_og[idx, :]
    np.savetxt(to_filepath, csv_data_sampled, fmt="%s")