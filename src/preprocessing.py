import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from CreditCardData import CreditCardData


## doing oversampling by itself
obj = CreditCardData()
X_train, X_test, y_train, y_test = obj.split_data()
print("Y COUNT", np.unique(y_train[0], return_counts=True))
X_resampled, y_resampled = obj.oversample(X_train, y_train)
print("Y RESAMPLED COUNT", np.unique(y_resampled[0], return_counts=True))


## doing undersampling by itself
new_obj = CreditCardData()
X_train1, X_test1, y_train1, y_test1 = new_obj.split_data()
##print(new_obj.tomek_undersample(X_train1, y_train1)) --> can't use tomek bc too many features in data
print("Y COUNT", np.unique(y_train1[0], return_counts = True))
X_resampled1, y_resampled1 = new_obj.oversample(X_train1, y_train1)
print("Y RESAMPLED COUNT", np.unique(y_resampled1[0], return_counts=True))


## can also combine oversampling and undersampling
## idea from machine learning mastery blog: https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
