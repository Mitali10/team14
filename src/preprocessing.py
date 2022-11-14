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
X_resampled1, y_resampled1 = new_obj.undersample(X_train1, y_train1)
print("Y RESAMPLED COUNT", np.unique(y_resampled1[0], return_counts=True))


## can also combine oversampling and undersampling
obj2 = CreditCardData()
X_train2, X_test2, y_train2, y_test2 = obj2.split_data()
print("Y COUNT", np.unique(y_train1[0], return_counts = True))
X_resampled2, y_resampled2 = obj2.oversample_and_undersample(X_train2, y_train2)
print("Y RESAMPLED COUNT", np.unique(y_resampled2[0], return_counts=True))

obj3 = CreditCardData()
X_train3, X_test3, y_train3, y_test3 = obj3.split_data()
print("Y COUNT", np.unique(y_train1[0], return_counts = True))
X_resampled3, y_resampled3 = obj2.oversample_and_undersample(X_train2, y_train2)
print("Y RESAMPLED COUNT", np.unique(y_resampled2[0], return_counts=True))

## idea from machine learning mastery blog: https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
