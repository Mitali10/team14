import matplotlib.pyplot as plt
import numpy as np
from CreditCardData import CreditCardData
from sklearn import mixture
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

ccd_obj = CreditCardData()
X_train, X_test, y_train, y_test = ccd_obj.split_data()
model = svm.SVC(kernel='linear')

X_train = np.squeeze(X_train)
y_train = np.squeeze(y_train)
X_test = np.squeeze(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_test = np.squeeze(y_test)

print("Recall:",metrics.recall_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


''' Attempt at Making Graph
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = "winter")

ax = plt.gca()
xlim = ax.get_xlim()


y_train = np.squeeze(y_train)
ax.scatter(X_test[:,0], X_test[:,1], cmap = "winter", marker = "s")

w = model.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (model.intercept_[0]/w[1])
plt.plot(xx, yy)

plt.title("Support Vector Machine Classification")
plt.show()
'''

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
