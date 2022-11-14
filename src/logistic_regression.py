import numpy as np
import matplotlib.pyplot as plt

from CreditCardData import CreditCardData

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE



############################## GET DATA ##############################

# get data
ccd = CreditCardData(split=True)
print("sanity check", set(ccd.labels_test))


############################## MODEL -- NO SMOTE ##############################

# # train model
# pipe = make_pipeline(StandardScaler(), LogisticRegression())          # without cross val
# pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=10))   # with cross val
# pipe.fit(ccd.data_train, ccd.labels_train) 

# # predict with test data
# predictions = pipe.predict(ccd.data_test)

# # assess model
# score = pipe.score(ccd.data_test, ccd.labels_test)
# print("Accuracy Score:", score)

############################## MODEL -- WTIH SMOTE ##############################

# train model
# pipe = make_pipeline(SMOTE(random_state=42), StandardScaler(), LogisticRegression())       # without cross val
# pipe = make_pipeline(SMOTE(random_state=42), StandardScaler(), LogisticRegressionCV(cv=10))   # with cross val
# pipe.fit(ccd.data_train, ccd.labels_train) 

# # predict with test data
# predictions = pipe.predict(ccd.data_test)

# # assess model
# score = pipe.score(ccd.data_test, ccd.labels_test)
# print("Accuracy Score with SMOTE:", score)

############################## MODEL -- WTIH UNDERSAMPLING ##############################

# train model
pipe = make_pipeline(RandomUnderSampler(sampling_strategy='majority'), StandardScaler(), LogisticRegression())       # without cross val
pipe = make_pipeline(RandomUnderSampler(sampling_strategy='majority'), StandardScaler(), LogisticRegressionCV(cv=10))       # with cross val
pipe.fit(ccd.data_train, ccd.labels_train) 

# predict with test data
predictions = pipe.predict(ccd.data_test)

# assess model
score = pipe.score(ccd.data_test, ccd.labels_test)
print("Accuracy Score with UNDERSAMPLING:", score)

############################## POST-PROCESSING: EVERYTHING? ##############################

report = classification_report(ccd.labels_test, predictions)
print(report)

############################## POST-PROCESSING: RECALL ##############################


recall = recall_score(ccd.labels_test, predictions, average='macro')
print("Recall:", recall)


############################## POST-PROCESSING: PRECISION ##############################

# Get the predicited probability of testing data
y_score = pipe.predict_proba(ccd.data_test)[:, 1]


average_precision = average_precision_score(ccd.labels_test, y_score)
print("Average Precision:", average_precision)

############################## POST-PROCESSING: CONFUSION MATRIX ##############################

# calc confusion matix
cm = metrics.confusion_matrix(ccd.labels_test, predictions)
print("Confusion Matrix\n", cm)

# show confusion matrix
plt.figure(figsize=(9,9))
plt.imshow(cm, cmap='Pastel1')
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