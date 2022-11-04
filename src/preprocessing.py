import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from CreditCardData import CreditCardData

obj = CreditCardData()
X_train, X_test, y_train, y_test = obj.split_data()
    
