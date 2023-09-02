#!/usr/bin/env python
# coding: utf-8

# # FastAI :

# 
# Fastai is an open-source deep learning library designed to make it easier for both beginners and experienced practitioners to build and train high-quality machine learning models. It provides a high-level API that simplifies the process of creating complex neural networks and applying them to various tasks.

# In[1]:


import fastbook
fastbook.setup_book()

get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')


# In[2]:


from fastai.vision.all import *                                             #importing all libraries from fastai
PATH = untar_data(URLs.CAMVID_TINY)                                         #path to dataset


# # Semantic Segmentation :

# Semantic segmentation is a computer vision task where the goal is to label each pixel in an image with the category it belongs to. It's like coloring a black and white picture where each color represents a different object or region. This task is used to understand and analyze images in detail, enabling applications like identifying objects in scenes, autonomous driving, medical image analysis, and more. It involves pixel-level classification, doesn't distinguish between instances of the same class, and is crucial for tasks requiring fine-grained image understanding. Here's how you can do it:
# 
# 
# 
# We need a dataset with images and their corresponding pixel-wise labels. Fastai's SegDataLoaders can be used to load and preprocess this data.
# 
# path = untar_data(URLs.CAMVID_TINY)
# 
# 
# Load the images and labels into a DataLoaders object.
# 
# dls = SegmentationDataLoaders.from_label_func(                              
#       PATH,                                                                 
#       bs=8,                                                                 
#       fnames = get_image_files(PATH/"images"),                              
#       label_func=lambda o: PATH/"labels"/f"{o.stem}_P{o.suffix}",           
#       codes=np.loadtxt(PATH/"codes.txt", dtype=str)) 
#       
#       
# Choose a segmentation architecture. Fastai provides several pre-defined architectures that you can use.
# 
# learn = unet_learner(dls, resnet34)                                         
# 
# learn.fine_tune(4) 
# 
# 
# 
# Use the trained model to make predictions and visualize the segmentation masks.
# 
# learn.show_results (max_n = 3, figsize = (8, 8))  

# In[5]:


dls = SegmentationDataLoaders.from_label_func(                              #dataloader for semantic segmentation           
      PATH,                                                                 #path to data directory
      bs=8,                                                                 #batch size
      fnames = get_image_files(PATH/"images"),                              #getting image file path
      label_func=lambda o: PATH/"labels"/f"{o.stem}_P{o.suffix}",           #lambda function to generate label path
      codes=np.loadtxt(PATH/"codes.txt", dtype=str))                        #load class code from "code.txt"


# In[6]:


learn = unet_learner(dls, resnet34)                                         #initializing resnet34 pretrained mmodel  

learn.fine_tune(4)                                                          #training for 4 eochs 


# In[10]:


learn.show_results (max_n = 3, figsize = (8, 8))                                 #viewing the segmented image results


# In[ ]:




