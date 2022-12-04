import matplotlib.pyplot as plt
import numpy as np
from CreditCardData import CreditCardData
from sklearn.ensemble import RandomForestClassifier
from sklearn import mixture, metrics
from sklearn.cluster import KMeans, DBSCAN
from imblearn.over_sampling import SMOTE


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


forest = RandomForestClassifier(max_depth=20, random_state=0)
forest.fit(X_smote, y_smote)
labels_pred_test = forest.predict(ccd.data_test)
score = forest.score(ccd.data_test, ccd.labels_test)
print("SCORE FOR FOREST", score)


confusion_matrix = metrics.confusion_matrix(ccd.labels_test, labels_pred_test, normalize = 'true')
print(confusion_matrix); 

recall = metrics.recall_score(ccd.labels_test, labels_pred_test, average='macro')
print("Recall:", recall)


balanced_accuracy = metrics.balanced_accuracy_score(ccd.labels_test, labels_pred_test)
print("balanced_accuracy:", balanced_accuracy)

precision_score = metrics.precision_score(ccd.labels_test, labels_pred_test)
print("precision_score:", precision_score)

report = metrics.classification_report(ccd.labels_test, labels_pred_test)
print(report)


############################## POST-PROCESSING: CONFUSION MATRIX ##############################

# calc confusion matix
cm = metrics.confusion_matrix(ccd.labels_test, labels_pred_test)
print("Confusion Matrix\n", cm)

# show confusion matrix
plt.figure(figsize=(9,9))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion matrix', size=15)
plt.colorbar()

plt.xticks([0, 1], ["0", "1"], rotation=45, size=2)
plt.yticks([0, 1], ["0", "1"], size = 2)
plt.tight_layout()
plt.ylabel('Actual label', size=15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
    horizontalalignment='center',
    verticalalignment='center')

plt.show()




