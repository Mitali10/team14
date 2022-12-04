import numpy as np
import matplotlib.pyplot as plt

from CreditCardData import CreditCardData
from imblearn.over_sampling import SMOTE



ccd = CreditCardData(split=True)


smote = SMOTE(random_state=42) # desired ratio minority/majority = 0.1
X_smote, y_smote = smote.fit_resample(ccd.data_train, ccd.labels_train)


############################## DISPLAY -- WTIH SMOTE ##############################


import matplotlib.pyplot as plt

values, counts = np.unique(ccd.labels_train, return_counts=True)
values_smote, counts_smote = np.unique(y_smote, return_counts=True)
print(values, counts)
print(values_smote, counts_smote)

labels = ['Before', 'After']
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

p1 = ax.bar(labels, [counts[0], counts_smote[0]],  width, label='Authentic', color='green')
p2 = ax.bar(labels, [counts[1], counts_smote[1]], width, bottom=[counts[0], counts_smote[0]],label='Fraud', color='red')
ax.bar_label(p1, label_type='center')
ax.bar_label(p2, label_type='center')

ax.set_ylabel('Count')
ax.set_title('Comparison Before and After Oversampling')
ax.legend()

plt.savefig('smote_edge.png', dpi=900)
plt.show()



# [0 1] [227466    379]
# [0 1] [227466 227466]

