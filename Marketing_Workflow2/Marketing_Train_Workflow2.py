#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


# # Data Preprocessing and Transformation

# In[ ]:





# In[2]:


df = pd.read_csv('marketing_training.csv')


# In[3]:


df.head()


# In[ ]:





# In[4]:


#printing unique values
unique = []
for i in list(df.select_dtypes(include=["object_"]).columns):
    unique.append(list(df[i].unique()))

    


# In[5]:


#creating dataframe of unqiue values of each categorical columns
unique_values = pd.DataFrame()
unique_values['columns'] = list(df.select_dtypes(include=["object_"]).columns)
unique_values['unique_values'] = unique


# In[6]:


pd.set_option('expand_frame_repr', True)
pd.set_option('max_colwidth', 200)
unique_values


# There are columns with unknown and nonexistent as values. These values can also be treated as NaNs

# In[7]:


df = df.replace('unknown', np.nan)
df = df.replace('nonexistent', np.nan)


# In[8]:


df.head()


# In[9]:


df.columns


# In[10]:


df[['pdays','pmonths','previous','poutcome']]


# It is clear the customers who were not contacted or failing to contact is filled with values 999 in pdays and pmonths.
# To solve this we can create a boolean variable to show if the customer was previously contacted or not

# In[11]:


# True and False are equivalent to 1 and 0 in numerical computations
df['previously_contacted'] = df.pdays!=999 


# pdays and pmonths are identical in a way they represent the same and must highly correlated. So we need to remove pmonths from the datset

# In[12]:


df[['previously_contacted','responded']]


# In[13]:


pd.crosstab(df.previously_contacted, df.responded, dropna=False)


# good success ratio when they are previously contacted

# In[14]:


df.pdays.unique()


# The values in pdays can skew the data and this can be considered outlier as it is too far from the other values. We can substitute 999
# with -1

# In[15]:


df.loc[df.pdays==999, 'pdays'] = -1


# ## Variable transformation

# The categorical variables can transformed using one hot encoding. One hot encoding is way to transform the categorical data into binary variables with its unique values becoming columns and filled with 0,1 for non-existent and existent 

# In[16]:


df_tranformed = pd.get_dummies(df, dummy_na=True, drop_first=True)


# In[17]:


df_tranformed


# In[18]:


#removing all the columns having NA values
df_tranformed = df_tranformed.loc[:, (df_tranformed != 0).any(axis=0)] 


# In[20]:


df_tranformed.head()


# ## Imputing missing values

# In[21]:


#filling with mean
df_tranformed['impute_age'] = df_tranformed['custAge'].isnull()
imputed_age = df_tranformed.mean().custAge
df_tranformed['custAge'] = df_tranformed['custAge'].fillna(imputed_age)


# In[22]:


df_tranformed =df_tranformed.drop('impute_age',axis= 1)


# ## Feature Standardization of continous variables

# In[23]:


from sklearn.preprocessing import MinMaxScaler
X = df_tranformed.drop('responded_yes', axis=1).values.astype(float)
y = df_tranformed.responded_yes.values.astype(float)

X_scaler = MinMaxScaler(feature_range=(0,1))
X = X_scaler.fit_transform(X)


# In[24]:




list(df_tranformed.columns)[:-1]


# ## Dealing Imbalaced Data

# In[25]:


df.head()


# In[26]:


df_tranformed


# In[28]:


target_count = df_tranformed.responded_yes.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)');


# # Resampling

# A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and / or adding more examples from the minority class (over-sampling).

# ### Under-sampling: Tomek links

# Tomek links are pairs of very close instances, but of opposite classes. Removing the instances of the majority class of each pair increases the space between the two classes, facilitating the classification process.

# In[29]:


from imblearn.under_sampling import TomekLinks
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)

print('Removed indexes:', id_tl)

plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')


# ### Over-sampling: SMOTE

# SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

# In[30]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')


# ### Over-sampling followed by under-sampling

# Now, we will do a combination of over-sampling and under-sampling, using the SMOTE and Tomek links techniques:

# In[31]:


from imblearn.combine import SMOTETomek

smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)

plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')


# # Now we have created 3 datasets
# 1.  Tomek Links 
# 2. SMOTE
# 3. SMOTE + Tomek

# # Modelling

# # Model training and ROC validation
# 
# I created a function `roc_analysis` that fits a classifer to a subset of the data and then cross-validates it on the remainder of the data. This fit-and-validate cycle is repeated k times (k-fold cross validation). 
# 
# The function also produces ROC curves to quantify model performance. In the presence of skewed classes, ROC curves better characterize algorithm performance than classification accuracy/error.

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def parse_function_name(s):
    end = s.find( '(' )
    if end != -1:
        return s[:end]
    return 'error parsing function name'

def roc_analysis(classifier, X,y, number_splits=5): 
    
   
    cross_validator = StratifiedKFold(n_splits=number_splits, 
                                      shuffle=True)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cross_validator.split(X, y)):
        probabilities = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probabilities[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='classifier that ignores features', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % 
             (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                     color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(parse_function_name(classifier.__str__()))
    plt.legend(loc="lower right")
    plt.show()


# #Logistic Regresion with normal data

# In[36]:


from sklearn.linear_model import LogisticRegression
roc_analysis(LogisticRegression(), X,y)


# Mean area under the ROC curve (AUC) is 0.79. That's pretty good performance: it's about half-way between a classifier that ignores features (AUC = 0.5) and a perfect classifier (AUC = 1). 
#  

# Model can be affected with high bias to check this we increase the model complexity by using MLP classifier

# 

# In[37]:


from sklearn.neural_network import MLPClassifier
roc_analysis(MLPClassifier(hidden_layer_sizes=(100), alpha=1), X,y) 


# Mean AUC is 0.80 which is very close to Logistic model so the model is not having high bias
#  

# # Logistic Regression with Tomek resampled data

# In[38]:


roc_analysis(LogisticRegression(), X_tl, y_tl) 


# In[39]:


#MLP with Tomek resampled data
roc_analysis(MLPClassifier(hidden_layer_sizes=(100), alpha=1), X_tl, y_tl) 


# In[ ]:


#Logistic Regression with SMOTE resampled data


# In[41]:


roc_analysis(LogisticRegression(), X_sm, y_sm) 


# In[42]:


#MLP with SMOTE resampled data
roc_analysis(MLPClassifier(hidden_layer_sizes=(100), alpha=1), X_sm, y_sm) 


# #Logistic Regression with SMOTE + Tomek data

# In[43]:


roc_analysis(LogisticRegression(), X_smt, y_smt) 


# In[45]:


#MLP with SMOTE+Tomek resampled data
roc_analysis(MLPClassifier(hidden_layer_sizes=(100), alpha=1), X_smt, y_smt)


# In[ ]:


# SVM


# # SVM

# In[48]:


from sklearn.svm import SVC
roc_analysis(SVC(probability=True), X,y)


# In[49]:


roc_analysis(SVC(probability=True), X_tl,y_tl)


# In[50]:


roc_analysis(SVC(probability=True), X_sm,y_sm)


# In[51]:


roc_analysis(SVC(probability=True), X_smt,y_smt)


# In[ ]:





# # Decision Tree

# In[52]:


from sklearn.tree import DecisionTreeClassifier
roc_analysis(DecisionTreeClassifier(), X,y)


# In[53]:


roc_analysis(DecisionTreeClassifier(), X_tl,y_tl)


# In[54]:


roc_analysis(DecisionTreeClassifier(), X_sm,y_sm)


# In[ ]:


roc_analysis(DecisionTreeClassifier(), X_smt,y_smt)


# In[ ]:





# # Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
roc_analysis(RandomForestClassifier(), X,y)
roc_analysis(RandomForestClassifier(), X_tl,y_tl)
roc_analysis(RandomForestClassifier(), X_sm,y_sm)
roc_analysis(RandomForestClassifier(), X_smt,y_smt,10)


# In[57]:


roc_analysis(RandomForestClassifier(), X_smt,y_smt,10)


# In[ ]:


#combining X_smt and y_smt into a dataframe to feed into calibration curve
data = pd.DataFrame(X_smt,columns=list(df_tranformed.columns)[:-1])


# In[ ]:


data['responded']= y_smt


# In[ ]:


#Use 5-fold cross-validation instead, and take the average of all of my data to make the calibration plot.
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

# store probability predictions and true labels here
decision_kfold_probability = []
svm_kfold_probaility = []
rf_kfold_probability = []

kfold_true_label = []


# In[73]:


#Train and predict on each split.
for train_index, validate_index in kf.split(data):
    #gives a list a index to split into train and validation
    kfold_train, kfold_validate = data.iloc[train_index], data.iloc[validate_index]
    #gives kfold_train and kfold_validate data
    feature_cols = list(df_tranformed.columns)[:-1]
    #getting input features except the last one which is responded

    train_features = kfold_train[feature_cols]
    train_labels = kfold_train['responded']
    validate_features = kfold_validate[feature_cols]
    validate_labels = kfold_validate['responded']
    #fitting DT  and RF
    #SVM_model = SVC().fit(X=train_features ,y=train_labels)
    decision_model = DecisionTreeClassifier().fit(X=train_features ,y=train_labels)
    rf_model = RandomForestClassifier().fit(X=train_features, y=train_labels)
    #getting predicted probabilties
    decision_kfold_probability.append(decision_model.predict_proba(validate_features)[:,1])
    #svm_kfold_probaility.append(SVM_model.predict_proba(validate_features)[:,1])
    rf_kfold_probability.append(rf_model.predict_proba(validate_features)[:,1])
    #storing the true label
    kfold_true_label.append(validate_labels)


# In[ ]:


#Concatenate the results and compute bins for calibration curves.
decision_kfold_probability_stacked = np.hstack(decision_kfold_probability)
rf_kfold_probability_stacked = np.hstack(rf_kfold_probability)
kfold_true_label_stacked = np.hstack(kfold_true_label)
#Now we have class probabilities and labels, we can compute the bins for calibration plot
from sklearn.calibration import calibration_curve   

#sklearn.calibration.calibration_curve returns the x,y coordinates for my logistic regression model predictions on the calibration plot.                               
dec_y, dec_x = calibration_curve(kfold_true_label_stacked, decision_kfold_probability_stacked, n_bins=10)
rf_y, rf_x = calibration_curve(kfold_true_label_stacked, rf_kfold_probability_stacked, n_bins=10)


# In[76]:


#Plot calibration curves
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# calibration curves
fig, ax = plt.subplots()
plt.plot(dec_y, dec_x , marker='o', linewidth=1, label='logreg')
plt.plot(rf_x, rf_y, marker='o', linewidth=1, label='rf')

# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.legend()
plt.show()


# #Preparing test data

# In[ ]:





# In[ ]:


def sanitize(data_frame):
    data_frame = data_frame.replace('unknown', np.nan)
    data_frame = data_frame.replace('nonexistent', np.nan)
    data_frame = data_frame.drop('pmonths',axis=1)
    data_frame['previously_contacted'] = data_frame.pdays!=999 
    data_frame.loc[data_frame.pdays==999, 'pdays'] = -1
    data_frame = track_missing_ages(data_frame)
    data_frame = impute_age(data_frame)
    return data_frame


# In[ ]:


def get_values(data_frame): 
    return data_frame.values.astype(float)

def track_missing_ages(data_frame): 
    data_frame['impute_age'] = data_frame['custAge'].isnull()
    return data_frame

def impute_age(data_frame): 
    imputed_age = data_frame.mean().custAge
    data_frame['custAge'] = data_frame['custAge'].fillna(imputed_age)
    return data_frame

def one_hot_encode(data_frame): 
    data_frame = pd.get_dummies(data_frame, dummy_na=True, drop_first=True)
    
    return data_frame.loc[:, (data_frame != 0).any(axis=0)] 
def encode_train_test(): 
    df_train = sanitize(pd.read_csv('marketing_training.csv'))
    df_train_X = df_train.drop('responded', axis=1)
    df_train_y = one_hot_encode(pd.DataFrame(df_train['responded']))

    df_test_X = sanitize(pd.read_csv('marketing_test.csv')).drop('Unnamed: 0', axis=1)

    df_X = pd.concat([df_train_X, df_test_X])
    df_X = one_hot_encode(df_X)
    split = len(df_train_X)
    df_train_X = df_X[:split]
    df_test_X = df_X[split:]
        
    return (get_values(df_train_X), 
            get_values(df_train_y), 
            get_values(df_test_X))


# In[ ]:





# In[ ]:


def compute_normalized_X_y(): 
    X_train, y_train, X_test = encode_train_test() 
    X_scaler = MinMaxScaler(feature_range=(0,1))
    X_train = X_scaler.fit_transform(X_train)
    X_smt, y_smt = smt.fit_sample(X_train, y_train)
    X_test = X_scaler.transform(X_test)
    return X_smt, y_smt, X_test

def convert_to_yes_or_no(x): 
    if int(x) == 0: 
        return 'no'
    else: 
        return 'yes'

def predict_with_final_model():
    X_train, y_train, X_test = compute_normalized_X_y()
    final_model = RandomForestClassifier()
    final_model1 =  DecisionTreeClassifier()
    predictions_ran = final_model.fit(X_smt, y_smt.ravel()).predict(X_test)
    predictions_ran = list(map(convert_to_yes_or_no, predictions_ran))
    predictions_dec = final_model1.fit(X_smt, y_smt.ravel()).predict(X_test)
    predictions_dec = list(map(convert_to_yes_or_no, predictions_dec))
    df_test = pd.read_csv('marketing_test.csv')
    df_test['predict_RF'] = predictions_ran
    df_test['predict_DT'] = predictions_dec
    df_test = df_test[['predict_RF','predict_DT']]
    df_test.to_csv('marketing_test_with_predictions.csv')
    


# In[85]:


predict_with_final_model()


# In[ ]:





# In[ ]:




