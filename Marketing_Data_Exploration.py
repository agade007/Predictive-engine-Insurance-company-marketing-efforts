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


# In[2]:


marketing = pd.read_csv('marketing_training.csv')


# In[3]:


marketing.head()


# In[4]:


marketing.shape


# In[5]:


print('Number of training examples: {0}'.format(marketing.shape[0]))
print('Number of features for each example: {0}'.format(marketing.shape[1]))


# In[6]:


marketing.info()


# In[7]:


#List of columns
pd.DataFrame(data = {'Feature Label': marketing.columns})


# In[ ]:





# # Visualizing Missing values

# In[9]:


import missingno as msno 


# In[10]:


msno.matrix(marketing) 


# We can see that there is pattern of missing values in custAge, schooling, day_of_week

# In[11]:


msno.bar(marketing) 


# There are three columns that have missing values: custAge, schooling, day_of_week

# # Data Exploration

# In[14]:


# for generation of interactive data visualization
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
labels = ["no", "yes"]
values = marketing["responded"].value_counts().values

trace = go.Pie(labels = labels, values = values)
layout = go.Layout(title = 'Distribution of Target Variable - Responded')

fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# As we can see, the target variable is highly skewed, with only 11.3% of responded said yes. This is something that we will have to keep in mind when evaluating the results of our predictions later on.

# We should also get some information on the categorical variables in this dataset. We can do this by subsetting the categorical feature columns.

# In[15]:


marketing.select_dtypes(include=["object_"]).columns


# In[16]:


import numpy as np
df1 = marketing.replace(np.nan, 'missing', regex=True)


# In[15]:


list(df1.select_dtypes(include=["object_"]).columns)


# In[17]:


for i in list(df1.select_dtypes(include=["object_"]).columns):
    df1[i] =df1[i].astype('category')


# In[18]:


df1.select_dtypes(exclude=["number","bool_","object_"])


# In[19]:


list(df1.select_dtypes(include=["category"]).columns)[0]


# In[20]:


df1['custAge'] = df1['custAge'].astype('str')


# In[21]:


df1['custAge'][df1.custAge == 'missing'] = 0


# In[22]:


df1['custAge'] = df1['custAge'].astype('float')


# In[23]:


df1['custAge'][df1.custAge == 0] = df1.custAge.mean()


# In[24]:


df1[list(df1.select_dtypes(include=["category"]).columns)[1]].unique()


# In[24]:


for i in df1.select_dtypes(include=["category"]).columns:
    print("Categories in column " + str(i) + str(df1[i].unique()))
    print('')


# Below is a visualization of the distribution within each categorical feature. The legend on the right shows the labels provided each category.

# In[26]:


cat_columns =list(df1.select_dtypes(include=["category"]).columns)
cat_counts = df1[list(df1.select_dtypes(include=["category"]).columns)].apply(pd.value_counts)

trace = []
for i in range(cat_counts.shape[0]):
    trace_temp = go.Bar(
        x= np.asarray(cat_columns),
        y= cat_counts.values[i],
        name = cat_counts.index[i]
    )
    trace.append(trace_temp)

layout = go.Layout(
    barmode = 'stack',
    title = 'Distribution of Categorical Features'
)

fig = go.Figure(data = trace, layout = layout)
iplot(fig)


# In[27]:


misc_columns = list(df1.select_dtypes(include=["float_"]).columns)
misc_columns


# In[28]:


list(df1.select_dtypes(include=["float_"]).columns)[:-2]


# In[29]:


#create boxplots for 'ind' columns
ind_columns = list(df1.select_dtypes(include=["float_"]).columns)[:-2]
trace1 = []
for i in range(len(ind_columns)):
    trace_temp = go.Box(
        y= df1[ind_columns[i]],
        name = ind_columns[i]
    )

    trace1.append(trace_temp)

layout1 = go.Layout(
    title = 'Distribution of "numerical" Features'
)
fig1 = go.Figure(data = trace1, layout = layout1)
iplot(fig1)


# We can see from the distributions above that while some features have relatively small ranges while others have relatively larger ranges, it may be useful to consider normalization in order to improve the performance of our chosen model when it comes time to optimize our predictions.
# 
# 

# In[43]:


## Analyzing target variable


# In[44]:


with sns.axes_style('white'):
    g = sns.factorplot("responded", data=df1, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=1)


# It is clear that the dataset is imbalanced

# In[47]:


for i in list(df1.select_dtypes(include=["category"]).columns):
    with sns.axes_style('white'):
        g = sns.factorplot(i, data=df1, aspect=4.0, kind='count',
                       hue='responded')
        g.set_xticklabels(step=1)


# The dataset imbalance so there not a strong finding.But the people who said yes seems to have following traits:
# 1. They are in high profile jobs
# 2. Attained high education
# 3. Do not have loan

# In[30]:


ax = sns.boxplot(x="responded", y="custAge", data=df1)


# In[31]:


list(df1.select_dtypes(include=["float_"]).columns)[:-2]


# In[32]:


ax = sns.boxplot(x="responded", y='emp.var.rate', data=df1)


# In[55]:


ax = sns.boxplot(x="responded", y='cons.price.idx', data=df1)


# In[56]:


ax = sns.boxplot(x="responded", y='euribor3m', data=df1)


# In[ ]:




