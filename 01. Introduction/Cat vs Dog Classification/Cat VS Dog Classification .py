#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Cat VS Dog Classification :

# I start by organizing the Oxford-IIIT Pet Dataset, splitting it into training and validation sets. After setting up my environment and augmenting the data, I create a DataLoader. I load a pretrained model, fine-tune it for cat vs. dog classification, and train it while monitoring key metrics. Post-validation, I use the model for inference or deploy it for real-world applications, leveraging fastai's user-friendly APIs to streamline the process.

# In[4]:


from fastai.vision.all import *                            
PATH = untar_data(URLs.PETS)/"images"                      


# In[5]:


def is_cat(x):                                             
  return x[0].isupper()                                   . 
 
dls = ImageDataLoaders.from_name_func(                    
      PATH, get_image_files(PATH), valid_pct=0.2,          
      seed=42, label_func=is_cat,                          
      item_tfms=Resize(224))                               


# In[7]:


learn = cnn_learner(dls, resnet34,                        # using pretrained resnet model
                    metrics=error_rate)                   
learn.fine_tune(1)                                        # training  


# In[8]:


PATH = "/content/Bal.JPG"                       # dataset path 
img = PILImage.create(PATH)                      
img.to_thumb(224)                                


# In[9]:


is_cat, _, probs = learn.predict(img)               #predicting
print(f"Is this a cat?: {is_cat}.")
print(f"Probability: {probs[1].item():.6f}")

