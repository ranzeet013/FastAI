#!/usr/bin/env python
# coding: utf-8

# In[1]:


import socket,warnings
try:
    socket.setdefaulttimeout(1)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
except socket.error as ex: raise Exception


# In[2]:


import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


# ### Download Birds And Non-Birds Image :

# In[3]:


get_ipython().system('pip install -Uqq duckduckgo_search')


# In[4]:


from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')


# Let's start by searching for a bird photo and seeing what kind of result we get. We'll start by getting URLs from a search:

# In[5]:


urls = search_images('bird photos', max_images=1)
urls[0]


#  then download a URL and take a look at it:

# In[6]:


from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)
im.to_thumb(256,256)


# In[7]:


download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)


# In[8]:


searches = 'forest','bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)


# In[9]:


failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# ### Training Model :

# To train a model, i will create DataLoaders, which contains a training set (the images used to create a model) and a validation set (the images used to check the accuracy of a model -- not used during training). In fastai i can create that easily using a DataBlock, and view sample images from it:

# In[10]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=6)


# The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest").
# 
# get_items=get_image_files, 
# 
# To find all the inputs to our model, run the get_image_files function (which returns a list of all image files in a path).
# 
# splitter=RandomSplitter(valid_pct=0.2, seed=42),
# 
# Split the data into training and validation sets randomly, using 20% of the data for the validation set.
# 
# get_y=parent_label,
# 
# The labels (y values) is the name of the parent of each file (i.e. the name of the folder they're in, which will be bird or forest).
# 
# item_tfms=[Resize(192, method='squish')]

# In[11]:


learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# Generally when I run this I see 100% accuracy on the validation set (although it might vary a bit from run to run).
# 
# "Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset (called the pretrained model), and adjusting the weights a little bit so that the model learns to recognise your particular dataset. In this case, the pretrained model was trained to recognise photos in imagenet, and widely-used computer vision dataset with images covering 1000 categories)

# In[12]:


is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")


# In[ ]:




