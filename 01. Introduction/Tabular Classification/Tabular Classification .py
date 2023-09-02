#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastbook
fastbook.setup_book()

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Tabular Classification :

# Tabular classification using Fastai involves training a machine learning model to predict categories based on tabular data (structured data with rows and columns). To illustrate, let's consider a scenario where we want to predict whether a person's income is above or below a certain threshold using the "Adult Sample" dataset.

# First, we import the necessary Fastai modules. Next, we specify the data path using untar_data and load the dataset as a TabularDataLoaders object. We define categorical and continuous column names, along with the target column name ('salary' in this case). We also apply preprocessing steps like categorifying categorical variables, filling missing values, and normalizing continuous features.

# With the data loaded and preprocessed, we create a tabular_learner instance, which initializes a neural network suitable for tabular data classification. Finally, we train the model using the fit_one_cycle method.

# In[2]:


from fastai.tabular.all import *                                 # importing libraries. 
PATH = untar_data(URLs.ADULT_SAMPLE) 


# In[3]:


dls = TabularDataLoaders.from_csv(                               # tabular data loader . 
      PATH/"adult.csv",                                          # dataframe initializing 
      path=PATH,                                                 # path of dataframe 
      y_names="salary",                                          # target column of dataframe 
      cat_names=["workclass", "education", "marital-status",     #categorical columns
                 "occupation", "relationship", "race"],           
      cont_names=["age", "fnlwgt", "education-num"],             # continous columns. 
      procs=[Categorify, FillMissing, Normalize]) 


# In[4]:


learn = tabular_learner(dls, metrics=accuracy)                   # training model . 
learn.fit_one_cycle(3)                                           # training for 3 cycle 


# In[ ]:




