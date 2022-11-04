from sklearn.linear_model import LogisticRegression
from CreditCardData import CreditCardData
import numpy as np


ccd = CreditCardData(split=True)

# print("sanity check", set(ccd.labels_test))

# logisticRegr = LogisticRegression()
# logisticRegr.fit(ccd.data_train, ccd.labels_train)
# predictions = logisticRegr.predict(ccd.data_test)
# score = logisticRegr.score(ccd.data_test, ccd.labels_test)
# print("Score", score)

# import matplotlib.pyplot as plt
# from sklearn import metrics

# cm = metrics.confusion_matrix(ccd.labels_test, predictions)
# print("Confusion Matrix", cm)

# plt.figure(figsize=(9,9))
# plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
# plt.title('Confusion matrix', size = 15)
# plt.colorbar()
# tick_marks = np.arange(10)
# plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
# plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
# plt.tight_layout()
# plt.ylabel('Actual label', size = 15)
# plt.xlabel('Predicted label', size = 15)
# width, height = cm.shape
# for x in range(width):
#  for y in range(height):
#   plt.annotate(str(cm[x][y]), xy=(y, x), 
#   horizontalalignment='center',
#   verticalalignment='center')