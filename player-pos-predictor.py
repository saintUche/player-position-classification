import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

'''Fetching the data'''
DATA_PATH = "C:\projectsAZ\classification\player\playerData.csv"

def load_data(data_path=DATA_PATH):
    return pd.read_csv(data_path, sep=";")

DATA = load_data()


'''Taking a look at the data'''
x = DATA
#print(x.describe())
#print(x.info()) # 2644 rows, 400 columns, 391 entries are floats 9 are objects
#print(x.select_dtypes(include="object").columns.values) # taking a look at the columns which are not floats 

''' looking at the data, position2 is the specific position of the player.
On further inspection on position 2, we can deducde it sensible to make tehe following adjustments to the following categories.
'''

#print(x["position2"].unique())
x = x.replace({
    'Central Midfield': 'Midfielder - Central Midfield',
    'Forward - Second Striker' : 'Forward - Centre-Forward' 
    })

''' We must handle the different position categories and convert them to numerical values. '''
ordinal_encoder = OrdinalEncoder()
x[["position2"]] = ordinal_encoder.fit_transform(x[["position2"]])
y = x["position2"]

'''Now we can drop the Non numercial data'''
x = x.select_dtypes(exclude=['object'])

'''Some data is numercial but does not benefit to classifying player positions so must be dropped. 
I have selected data which I think is of no benefit including and which may also affect the ability of the model to predict the player position.
Reasons for this are detailed in excel sheet. 
'''

COLUMN_DATA_PATH = "C:\projectsAZ\classification\player\pc.csv"
column_headings = pd.read_csv(COLUMN_DATA_PATH, encoding='latin-1', header=None, on_bad_lines='skip')
x = x.drop(list(column_headings[0]), axis='columns')
x = x.drop("position2", axis="columns")

'''Now we can define y to be the encoded player positions. We can now inspect them to determine a suitable sampling method'''

positions, counts = np.unique(y, return_counts=True)
#plt.pie(counts, labels=positions, autopct='%1.1f%%', explode=len(positions)*[0.2], labeldistance=1.1, pctdistance=0.5)
#plt.show()

'''looking at the data a stratified sample is necessary before making a training and test set.'''

split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for train_index, test_index in split.split(x, y):
   x_train, x_test = x.loc[train_index], x.loc[test_index]
   y_train, y_test = y.loc[train_index], y.loc[test_index]

y_train_cb = (y_train == 2.0)
y_test_cb = (y_test == 2.0)



'''Stochastic Gradient Descent'''

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve


sgd_clf = SGDClassifier(random_state=42)
sgd_clf = sgd_clf.fit(x_train.values, y_train_cb)


#print(sgd_clf.predict([x.loc[0]]))
#print(cross_val_score(sgd_clf, x_train, y_train, cv=12, scoring="accuracy"))


#in this paticular case the classifier guess incorrectly

y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_cb, cv=3)
#print(confusion_matrix(y_train_cb, y_train_pred))
#print(precision_score(y_train_cb, y_train_pred))
#print(recall_score(y_train_cb, y_train_pred))

y_scores = cross_val_predict(sgd_clf, x_train, y_train, cv=3, method="decision_function")

print(y_scores)
precisions, recalls, threshold = precision_recall_curve(y_train_cb, y_scores)

def plot_precisions_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="recall")
    [...]

plot_precisions_recall_vs_threshold(precisions, recalls, threshold)
plt.show()