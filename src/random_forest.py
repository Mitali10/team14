import matplotlib.pyplot as plt
import numpy as np
from CreditCardData import CreditCardData
from sklearn.ensemble import RandomForestClassifier
from sklearn import mixture, metrics
from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score

import time

ccd = CreditCardData(split=True)
ccd.split_data()

smote = SMOTE(random_state=42) # desired ratio minority/majority = 0.1
X_smote, y_smote = smote.fit_resample(ccd.data_train, ccd.labels_train)

# gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
# gmm.fit(ccd.data_train)
# labels_pred = gmm.predict(ccd.data_test)


# from sklearn.neighbors import KDTree as kdtree
# import pandas as pd


# clustering = DBSCAN(eps=5, min_samples=2).fit(ccd.data_train)

# labels_pred_train = clustering.labels_

# # labels_pred_test = clustering.fit_predict(ccd.data_test)

# print(np.unique(labels_pred_train).shape)
# print("TRAIN ACCURACY------------------")
# confusion_matrix = metrics.confusion_matrix(ccd.labels_train, labels_pred_train, normalize = 'true')
# print("CONFUSION SHAPE", confusion_matrix.shape); 


# score = metrics.adjusted_mutual_info_score(ccd.labels_train, labels_pred_train)
# print("Adjusted Mutual Information (AMI)", score); 


# print("TEST ACCURACY------------------")
# confusion_matrix = metrics.confusion_matrix(ccd.labels_test, labels_pred_test, normalize = 'true')
# print(confusion_matrix.shape); 

# recall = metrics.recall_score(ccd.labels_test, labels_pred_test, average='macro')
# print("Recall:", recall)

# # kmeans_model = KMeans(n_clusters=2, random_state=1).fit(ccd.data_train)
# # labels = kmeans_model.labels_

# sil_score = metrics.silhouette_score(ccd.data_train, labels_pred_train, metric='euclidean')
# print("Silhoutte Scoree", sil_score)

# fk_score = metrics.cluster.fowlkes_mallows_score(ccd.labels_test, labels_pred_test)


# plt.scatter(ccd.data_test[:, 0], ccd.data_test[:, 1], c=labels_pred, s=40, cmap='viridis')
# plt.show()

info = {
  "Recall": [],
  "Balanced Accuracy": [],
  "Precision": [],
  "AUC-ROC": [],
  "CM": [],
  "Accuracy": [],
}

maxdepths = range(10, 20, 2)
for i in maxdepths:
  print (f"\n\n------- i = {i} --------------")
  start = time.time()
  forest = RandomForestClassifier(max_depth=i, random_state=0, verbose=True)
  forest.fit(X_smote, y_smote)
  labels_pred_test = forest.predict(ccd.data_test)
  score = forest.score(ccd.data_test, ccd.labels_test)
  print("SCORE FOR FOREST", score)

  info["Accuracy"].append(score)

  confusion_matrix = metrics.confusion_matrix(ccd.labels_test, labels_pred_test, normalize = 'true')
  print(confusion_matrix); 

  recall = metrics.recall_score(ccd.labels_test, labels_pred_test, average='macro')
  print("Recall:", recall)
  info["Recall"].append(recall)


  balanced_accuracy = metrics.balanced_accuracy_score(ccd.labels_test, labels_pred_test)
  print("balanced_accuracy:", balanced_accuracy)
  info["Balanced Accuracy"].append(balanced_accuracy)


  precision_score = metrics.precision_score(ccd.labels_test, labels_pred_test)
  print("precision_score:", precision_score)
  info["Precision"].append(precision_score)


  report = metrics.classification_report(ccd.labels_test, labels_pred_test)
  print(report)


  ############################## POST-PROCESSING: CONFUSION MATRIX ##############################

  aucroc = roc_auc_score(ccd.labels_test, labels_pred_test)
  print("AUCROC:", aucroc)

  info["AUC-ROC"].append(aucroc)



  # calc confusion matix
  cm = metrics.confusion_matrix(ccd.labels_test, labels_pred_test)
  print("Confusion Matrix\n", cm)
  info["CM"].append(cm)
  end = time.time()
  print("TIME", (end - start)/60)

# # show confusion matrix
# plt.figure(figsize=(9,9))
# plt.imshow(cm, cmap='Blues')
# plt.title('Confusion matrix', size=15)
# plt.colorbar()

# plt.xticks([0, 1], ["0", "1"], rotation=45, size=2)
# plt.yticks([0, 1], ["0", "1"], size = 2)
# plt.tight_layout()
# plt.ylabel('Actual label', size=15)
# plt.xlabel('Predicted label', size = 15)
# width, height = cm.shape
# for x in range(width):
#  for y in range(height):
#   plt.annotate(str(cm[x][y]), xy=(y, x), 
#     horizontalalignment='center',
#     verticalalignment='center')

# plt.show()

print(info)

# plot lines
plt.plot(maxdepths, info["Recall"], label = "Recall")
# plt.plot(maxdepths, info["AUC-ROC"], label = "AUC-ROC")
plt.plot(maxdepths, info["Precision"], label = "Precision")
plt.plot(maxdepths, info["Accuracy"], label = "Accuracy")
plt.title('Performance vs Max Tree Depth')
plt.xlabel('Max Tree Depth')
plt.ylabel('Performance')
plt.legend()
plt.show()

