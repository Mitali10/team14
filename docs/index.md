---
title: Midterm Report
---

## Introduction & Background
This is an increasing problem around the globe. Financial Fraud Action UK, for example, a trade association for UK financial services reported that losses from fraud totaled 618 million pounds which is a 9% increase from the year before. Additionally, this contributed to the fifth consecutive year of increased credit card fraud.

[Credit card fraud detection is an active area of research](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html#): existing analyses of credit card fraud have included techniques including deep learning, correlation matrices, and confusion matrices. We noted some potential complexities to credit card fraud data from our literature review: few fraudulent entries, no null values, and high false positive or high false negative rate.


## Problem Definition
According to Experian, credit card CVV’s sell on the dark web for as little as $5 and include personal information such as name, Social Security number, date of birth, and complete account numbers.In the digital age, it is important to have sophisticated fraud detection techniques to protect people’s livelihoods. Our project focuses on the detection of credit card fraud, to identify, analyze, and prevent financial fraud.


## Dataset
We found a Kaggle dataset to suit our purpose, titled [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains anonymized credit card transactions from September 2013 collected from European cardholders by the University of Brusells and the Machine Learning Group. The data is labeled as either fraudulent or genuine and is highly unbalanced, as the positive class/frauds account for 0.172% of all transactions with a total of 284,807 transactions.

## Methods
As of now, we plan to detect fraud using GMM clustering, regression analysis, and a multi-layer perceptron neural network. GMM results in a soft-clustering output that we can compare against the dataset’s labeled output to determine the model’s accuracy. Regression analysis (either linear, logistic, or both) can be used to identify the correlation between the input parameters and fraudulence. Finally, a well-tuned MLP neural network for classification will likely serve as the basis of comparison.


## Potential Results & Discussion
From a literature review, we found that Precision, Recall, F-score, and AUC would be the most suitable metrics for the unbalanced nature of our dataset. Specifically, for GMM, we plan to use the specificity, recall, and Balanced Accuracy of GMM. For regression analysis, we aim to use a straightforward mean squared or root squared error. Lastly, for MLP, we plan to use recall, precision and F-1 score. The area under the precision-recall curve (AUPRC) is also a useful performance metric in our problem setting since we care a lot about finding the positive examples.

## Contribution Table
![Contribution Table](images/contrib.png)

_Note: Person(s) in parenthesis are supporters, to help with research, ideas, and issues. Person(s) outside of parenthesis are hold main responsibilities for code implementation._

## Gantt Chart

[Please visit this link](https://docs.google.com/spreadsheets/d/1po2iJ1vFaG1CmB4J_Djwdk4p0TJhpWTq/edit?usp=sharing&ouid=108835189571718457513&rtpof=true&sd=true)

## Project Proposal Video

[Please visit this link to view the video.](https://drive.google.com/file/d/1r7rxllHFUsLnQqLfOq9kNSNOA2zKj3Yu/view?usp=sharing) [You can also visit this link to see our slides!](https://docs.google.com/presentation/d/1XgGfCcNP65d0k5lsMeJvIVTUFTnzbxMga3VwtFhcM-E/edit?usp=sharing)

If there are any issues, please let us know!

## Results and Discussion

# Data Cleaning

# Data Pre-Processing

# Logistic Regression

## References 

Bachmann, J. M., Bhat, N., & Luo. (2021, May 3). Credit Card Fraud Detection. 

Draelos, R. (2019, March 2). Measuring Performance: AUPRC and Average Precision [web log]. Retrieved October 7, 2022, from https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/. 

Department of Justice, Federal Trade Commission. 2021, February. Consumer Sentinel Network: Data Book 2020. https://www.ftc.gov/system/files/documents/reports/consumer-sentinel-network-data-book-2020/csn_annual_data_book_2020.pdf

Gomila, R. (2019, July 11). Logistic or linear? Estimating causal effects of experimental treatments on binary outcomes using regression analysis. https://doi.org/10.31234/osf.io/4gmbv

Kulatilleke. (2022). Challenges and Complexities in Machine Learning based Credit Card Fraud Detection. https://doi.org/10.48550/arXiv.2208.10943

Le Borgne, Y., Siblini, W., Lebichot, B., and Bontempi, G. (2022). Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook. https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook

Porkess, & Mason, S. (2012). Looking at debit and credit card fraud. Teaching Statistics, 34(3), 87–91. https://doi.org/10.1111/j.1467-9639.2010.00437.x


