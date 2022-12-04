## Logistic Regression:

Score: 0.9993153330290369
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.95      0.63      0.76        98

    accuracy                           1.00     56962
   macro avg       0.98      0.82      0.88     56962
weighted avg       1.00      1.00      1.00     56962

Recall: 0.8163001518840514
Average Precision: 0.8540956812714939
Confusion Matrix
 [[56861     3]
 [   36    62]]


## Logistic Regression, with CV:

Accuracy Score: 0.9990695551420246
/home/mitali/.local/lib/python3.8/site-packages/sklearn/linear_model/_logistic.py:444: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Accuracy Score with SMOTE: 0.9907657736736772
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56856
           1       0.15      0.89      0.26       106

    accuracy                           0.99     56962
   macro avg       0.58      0.94      0.63     56962
weighted avg       1.00      0.99      0.99     56962

Recall: 0.938876035054464
Average Precision: 0.7400267225389436
Confusion Matrix
 [[56342   514]
 [   12    94]]





## Logistic Regression, with SMOTE:
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Accuracy Score with SMOTE: 0.9911695516309118
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56862
           1       0.16      0.94      0.27       100

    accuracy                           0.99     56962
   macro avg       0.58      0.97      0.63     56962
weighted avg       1.00      0.99      0.99     56962

Recall: 0.9656297703211283
Average Precision: 0.7710722978476139
Confusion Matrix
 [[56365   497]
 [    6    94]]



## Logistic Regression with CV and SMOTE:
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
Accuracy Score with SMOTE: 0.9918366630385169
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56859
           1       0.16      0.84      0.27       103

    accuracy                           0.99     56962
   macro avg       0.58      0.92      0.63     56962
weighted avg       1.00      0.99      0.99     56962

Recall: 0.9183817335917139
Average Precision: 0.7276010533615198
Confusion Matrix
 [[56410   449]
 [   16    87]]



## Logistic Regression with Undersampling:
Accuracy Score with UNDERSAMPLING: 0.9738948772866122
              precision    recall  f1-score   support

           0       1.00      0.97      0.99     56856
           1       0.06      0.88      0.11       106

    accuracy                           0.97     56962
   macro avg       0.53      0.93      0.55     56962
weighted avg       1.00      0.97      0.99     56962

Recall: 0.9257166731710167
Average Precision: 0.4661108077223297
Confusion Matrix
 [[55382  1474]
 [   13    93]]


## Logistic Regression with Undersampling and CV

failed to converge error:

Accuracy Score with UNDERSAMPLING: 0.9657842070152031
              precision    recall  f1-score   support

           0       1.00      0.97      0.98     56871
           1       0.04      0.92      0.08        91

    accuracy                           0.97     56962
   macro avg       0.52      0.94      0.53     56962
weighted avg       1.00      0.97      0.98     56962

Recall: 0.9444647332762541
Average Precision: 0.6771432351413127
Confusion Matrix
 [[54929  1942]
 [    7    84]]
