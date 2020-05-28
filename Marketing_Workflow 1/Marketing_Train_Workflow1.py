#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Read Training Data

# In[2]:


training_data = pd.read_csv("data/marketing_training.csv", sep = ",", encoding = "utf-8")


# In[3]:


training_data.head()


# In[4]:


testing_data = pd.read_csv("data/marketing_test.csv", sep = ",", encoding = "utf-8")
testing_data.head()


# ### There are missing values. Lets handle them!

# In[5]:


missig_value_fill_method = 'majority' # takes 'majority', 'ffill', 'missing'


# In[7]:


if missig_value_fill_method == 'ffill':
    training_data = training_data.fillna(value = None, method = missig_value_fill_method)
    testing_data = testing_data.fillna(value = None, method = missig_value_fill_method)
elif missig_value_fill_method == 'majority':
    training_data = training_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
    testing_data = testing_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
else:
    training_data = training_data.apply(lambda x:x.fillna('missing'))
    testing_data = testing_data.apply(lambda x:x.fillna('missing'))
    
    #df = df.fillna(df.mode().iloc[0])


# In[8]:


training_data.head()


# In[9]:


testing_data.head()


# ### Now that we have replaced all missing values, lets check data type of each column and check balance of data with respect to classes (responded, not responded)

# In[10]:


testing_data.dtypes


# In[11]:


training_data.dtypes


# In[12]:


class_counts = np.unique(training_data.responded, return_counts = True)
print(class_counts[0][0],":",class_counts[1][0])
print(class_counts[0][1],":",class_counts[1][1])


# ### Convert string attributes to integer

# In[13]:


def convert_to_integer(column_values, col_name):
    values_set = set()
    column_int = []
    for column_value in column_values:
        values_set.add(column_value)
    val2idx = {v: i for i, v in enumerate(values_set)}
    idx2val = {i: v for v, i in val2idx.items()}
    
    for column_value in column_values:
        column_int.append(val2idx[column_value])
    return val2idx, idx2val, column_int   


# In[14]:


original_training_data = training_data
original_testing_data = testing_data


# In[15]:


val2idx_profession, idx2val_profession, training_data['profession'] = convert_to_integer(training_data['profession'].values, 'profession')
val2idx_marital, idx2val_marital, training_data['marital'] = convert_to_integer(training_data['marital'].values, 'marital')
val2idx_schooling, idx2val_schooling, training_data['schooling'] = convert_to_integer(training_data['schooling'].values, 'schooling')
val2idx_default, idx2val_default, training_data['default'] = convert_to_integer(training_data['default'].values, 'default')
val2idx_housing, idx2val_housing, training_data['housing'] = convert_to_integer(training_data['housing'].values, 'housing')
val2idx_loan, idx2val_loan, training_data['loan'] = convert_to_integer(training_data['loan'].values, 'loan')
val2idx_contact, idx2val_contact, training_data['contact'] = convert_to_integer(training_data['contact'].values, 'contact')
val2idx_month, idx2val_month, training_data['month'] = convert_to_integer(training_data['month'].values, 'month')

val2idx_day_of_week, idx2val_day_of_week, training_data['day_of_week'] = convert_to_integer(training_data['day_of_week'].values, 'day_of_week')
val2idx_poutcome, idx2val_poutcome, training_data['poutcome'] = convert_to_integer(training_data['poutcome'].values, 'poutcome')
val2idx_responded, idx2val_responded, training_data['responded'] = convert_to_integer(training_data['responded'].values, 'responded')


# In[16]:


testing_data['profession'] = testing_data['profession'].map(val2idx_profession) 
testing_data['marital'] = testing_data['marital'].map(val2idx_marital) 
testing_data['schooling'] = testing_data['schooling'].map(val2idx_schooling) 
testing_data['default'] = testing_data['default'].map(val2idx_default) 
testing_data['housing'] = testing_data['housing'].map(val2idx_housing) 
testing_data['loan'] = testing_data['loan'].map(val2idx_loan) 
testing_data['contact'] = testing_data['contact'].map(val2idx_contact) 
testing_data['month'] = testing_data['month'].map(val2idx_month) 
testing_data['day_of_week'] = testing_data['day_of_week'].map(val2idx_day_of_week) 
testing_data['poutcome'] = testing_data['poutcome'].map(val2idx_poutcome) 


# In[17]:


testing_data = testing_data.drop(columns=['ID'])
testing_data.head()


# In[18]:


training_data.head()


# # Train Models

# In[19]:


cv_results = {}
X = np.array(training_data.iloc[:,:-1])
Y = np.array(training_data.iloc[:,-1])


# In[20]:


# DECISION TREE CLASSIFIER

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify = Y)

clf = DecisionTreeClassifier(random_state=0)


cv_results['dt'] = cross_val_score(clf, X, Y, cv=10)
#clf = clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
clf = clf.fit(X, Y)
decision_tree_output = clf.predict(np.array(testing_data))


# In[21]:


# NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

cv_results['nb'] = cross_val_score(clf, X, Y, cv=10)
#clf = clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
clf = clf.fit(X, Y)
naiveBayes_output = clf.predict(np.array(testing_data))


# In[25]:


# SVM

from sklearn import svm

clf = svm.SVC(gamma='scale')

cv_results['svm'] = cross_val_score(clf, X, Y, cv=10)
#clf = clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
clf = clf.fit(X, Y)
svm_output = clf.predict(np.array(testing_data))


# In[26]:


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', max_iter = 100)

cv_results['lr'] = cross_val_score(clf, X, Y, cv=10)
#clf = clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
clf = clf.fit(X, Y)
lr_output = clf.predict(np.array(testing_data))


# In[ ]:





# ### Lets also take a look at neural network and how it performs. Just to be sure we leave nothing out of testing.

# In[27]:


from keras.layers import Input
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
def get_model():
    ip = Input(shape=(X.shape[1],))
    x = Dense(20, activation = 'relu')(ip)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(10, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(2, activation = 'softmax')(x)
    model = Model(inputs=ip, outputs=x)
    return model


# In[28]:


model = get_model()
model.summary()


# In[29]:


from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    y_cat_train = to_categorical(Y[train], num_classes=2, dtype='float32')
    y_cat_test = to_categorical(Y[test], num_classes=2, dtype='float32')
    
    model = get_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X[train], y_cat_train, epochs=20, batch_size=100, verbose=0)
    scores = model.evaluate(X[test], y_cat_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1])


# In[30]:


cv_results['nn'] = cvscores


# In[32]:


from matplotlib import pyplot as plt
plt.figure(figsize = (10,6))
plt.plot(cv_results['dt'], label = "Decision Tree", color='green', marker='o', linestyle='dashed',
         linewidth=1, markersize=6)
plt.plot(cv_results['nb'], label = "Naive Bayes", color='red', marker='+', linestyle='dashed',
         linewidth=1, markersize=6)

plt.plot(cv_results['svm'], label = "SVM", color='blue', marker='*', linestyle='dashed',
         linewidth=1, markersize=6)

plt.plot(cv_results['lr'], label = "Logistic Regression", color='orange', marker='^', linestyle='dashed',
         linewidth=1, markersize=6)

plt.plot(cv_results['nn'], label = "Neural Network", color='magenta', marker='D', linestyle='dashed',
         linewidth=1, markersize=6)

plt.xticks(np.arange(start=0, stop=10, step=1))
plt.legend(loc = 'best')
plt.xlabel("Folds")
plt.ylabel("Accuracy")
plt.show()


# ### And Finally, add predictions of all classifiers to the testing file. By looking at the results above, we belive "SVM" and "Logistic Regression" perform equally well, hence we report the outputs for both of them.

# In[33]:


df = pd.DataFrame()
df['ID'] = original_testing_data['ID']
df['svm_responded'] = svm_output
df['lr_responded'] = svm_output


# In[34]:


df['svm_responded'] = df['svm_responded'].map(idx2val_responded) 
df['lr_responded'] = df['lr_responded'].map(idx2val_responded)


# In[35]:


df.head()


# In[37]:


df.to_csv("result.csv", index=False, encoding = "utf-8")


# In[ ]:




