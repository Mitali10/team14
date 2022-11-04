import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from CreditCardData import CreditCardData

obj = CreditCardData()
X_train, X_test, y_train, y_test = obj.split_data()
print("Y COUNT",np.unique(y_train[0], return_counts=True) )
X_resampled, y_resampled = obj.oversample(X_train, y_train)
print("Y RESAMPLED COUNT", np.unique(y_resampled[0], return_counts=True))

    
