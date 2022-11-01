import matplotlib.pyplot as plt
import numpy as np
from CreditCardData import CreditCardData
from sklearn import mixture


ccd = CreditCardData()
ccd.split_train_test()


gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(ccd.data_train)
labels = gmm.predict(ccd.data_test)



plt.scatter(ccd.data_test[:, 0], ccd.data_test[:, 1], c=labels, s=40, cmap='viridis')
plt.show()