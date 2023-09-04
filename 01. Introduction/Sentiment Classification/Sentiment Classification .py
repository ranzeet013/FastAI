#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fastbook
fastbook.setup_book()

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


from fastai.text.all import *                           
PATH = untar_data(URLs.IMDB)                             


# ### Sentiment Classification :

# 
# 
# I start by downloading the IMDb dataset using the `untar_data` function provided by the fastai library. This dataset contains movie reviews and their corresponding sentiment labels. The `PATH` variable now holds the path to the directory where the dataset is stored on my local machine.
# 
# Next, I create a `TextDataLoaders` object called `dls` using the `from_folder` method. This object is essential for handling the data preprocessing tasks required for text classification. I specify that the test split of the dataset should be used as the validation set by setting `valid="test"` when creating `dls`.
# 
# Moving on, I create a text classifier learner named `learn` using the `text_classifier_learner` function. This learner will be based on the AWD-LSTM architecture, which is a type of recurrent neural network designed for text-related tasks. I also specify that I want to use the accuracy metric to evaluate the model's performance during training.
# 
# Finally, it's time to fine-tune the model using the `fine_tune` method. I decide to fine-tune it for 2 epochs with a learning rate of 1e-2. Fine-tuning allows the pre-trained AWD-LSTM model to adapt to the specific characteristics of the IMDb dataset, ultimately making it better at classifying movie review sentiments.
# 

# In[15]:


dls = TextDataLoaders.from_folder(PATH, valid="test")    


# In[17]:


learn = text_classifier_learner(dls, AWD_LSTM,          
                                metrics=accuracy)       
learn.fine_tune(2, 1e-2)                                 


# In[18]:


learn.predict("I am bored and tired with that movie!")    

