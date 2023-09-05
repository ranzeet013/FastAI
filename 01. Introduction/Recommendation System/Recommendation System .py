#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastbook
fastbook.setup_book()

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Recommendation System : 

# I begin by downloading the ML_SAMPLE dataset, likely containing user-item interaction data. Then, I create a collaborative filtering DataLoader (dls) from a "ratings.csv" file, structuring the data for training. I set up a collaborative filtering model with y_range=(0.5, 5.5) to predict ratings within that range. Lastly, I fine-tune the model for 10 epochs, enhancing its ability to make accurate user-item recommendations.

# In[23]:


from fastai.collab import *                                   
PATH = untar_data(URLs.ML_SAMPLE)                             


# In[24]:


dls = CollabDataLoaders.from_csv(PATH/"ratings.csv")          


# In[27]:


learn = collab_learner(dls, y_range=(0.5, 5.5))               
learn.fine_tune(10)                                           


# In[28]:


learn.show_results()                                              

