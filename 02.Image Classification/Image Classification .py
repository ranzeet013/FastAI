#!/usr/bin/env python
# coding: utf-8

# In[1]:


#@ INITIALIZATION: 
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# **LIBRARIES AND DEPENDENCIES:**
# - I have downloaded all the libraries and dependencies required for the project in one particular cell.

# In[41]:


#@ DOWNLOADING LIBRARIES AND DEPENDENCIES: 
from fastbook import *                                  # Getting all the Libraries. 
from fastai.callback.fp16 import *
from fastai.vision.all import *                         # Getting all the Libraries. 


# **GETTING THE DATA:**
# - I will download the [**Pets**](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset. 

# In[5]:


#@ GETTING THE DATA: 
path = untar_data(URLs.PETS)                           # Getting Path to the Dataset. 
path.ls()                                              # Inspecting the Path. 


# **Note:**
# - The dataset provides images and annotations directories. The [**Pets**](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset website tells that the annotations directory contains information about where the pets are rather than what they are. Since it is a **Classification** rather than **Localization**, I will ignore the annotations directory for now. 

# In[6]:


#@ INSPECTING IMAGES DIR: 
(path/"images").ls()                                    # Inspecting Images. 


# In[7]:


#@ GETTING ONE IMAGE: 
fname = (path/"images").ls()[0]                          # Getting an Image. 
re.findall(r"(.+)_\d+.jpg$", fname.name)                 # Extracting. 


# **INITIALIZING DATABLOCK AND DATALOADERS:**

# In[9]:


#@ INITIALIZING DATABLOCK: 
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),                        # Initializing DataBlock. 
                 get_items=get_image_files,                                 # Getting Image Files. 
                 splitter=RandomSplitter(seed=42),                          # Getting Random Splitting of Dataset. 
                 get_y=using_attr(RegexLabeller(r"(.+)_\d+.jpg$"),"name"),  # Getting Labels.  
                 item_tfms=Resize(460),                                     # Resizing Images. 
                 batch_tfms=aug_transforms(size=224, min_scale=0.75))       # Batch Augmentation.

#@ INITIALIZING DATALOADERS: 
dls = pets.dataloaders(path/"images")                                       # Initializing DataLoaders. 


# **Note:**
# - I have used **Resize** as an item transform with a large size and **RandomResizedCrop** as a batch transform with a smaller size. **RandomResizedCrop** will be added if min scale parameter is passed in aug transforms function as was done in **DataBlock** call above. 

# In[10]:


#@ INSPECTING IMAGES: 
dls.show_batch(nrows=1, ncols=3)                                             # Inspecting Images. 


# **CHECKING AND DEBUGGING DATABLOCK:**
# 

# In[11]:


#@ INITIALIZING DATABLOCK: 
pets1 = DataBlock(blocks=(ImageBlock, CategoryBlock),                       # Initializing DataBlock. 
                 get_items=get_image_files,                                 # Getting Image Files. 
                 splitter=RandomSplitter(seed=42),                          # Getting Random Splitting of Dataset. 
                 get_y=using_attr(RegexLabeller(r"(.+)_\d+.jpg$"),"name"))  # Getting Labels. 

#@ INSPECTING SUMMARY: UNCOMMENT BELOW:
# pets1.summary(path/"images") 


# **TRAINING THE MODEL:**

# In[13]:


#@ TRAINING THE MODEL: INITIALI TEST: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)                      # Initializing Pretrained Convolutions. 
learn.fine_tune(2)                                                          # Training the Model. 


# **CROSS ENTROPY LOSS:**
# - **Cross Entropy Loss** is a loss function which works even when the dependent variable has more than two categories. It results faster and more reliable training. 

# In[14]:


#@ INSPECTING ACTIVATIONS AND LABELS: 
x, y = dls.one_batch()                           # Getting a Batch of Data. 
y                                                # Inspecting Dependent Variable. 


# In[15]:


#@ GETTING PREDICTIONS:
preds, _ = learn.get_preds(dl=[(x, y)])          # Getting Predictions. 
preds[0]                                         # Inspecting Predictions. 


# In[16]:


#@ INSPECTING PREDICTIONS: 
len(preds[0]), preds[0].sum()


# **SOFTMAX ACTIVATION FUNCTION:**
# - Here, The **Softmax Activation Function** is used in the final layer to ensure that the activations are all between 0 and 1 and that they sum to 1. **Softmax** is similar to the **Sigmoid** function. 

# In[18]:


#@ INSPECTING SIGMOID FUNCTION: 
plot_function(torch.sigmoid, min=-4, max=4)             # Sigmoid Function. 


# In[19]:


#@ EXAMPLE: 
acts = torch.randn((6, 2))*2                            # Random Numbers.
acts


# In[20]:


#@ GETTING SIGMOID: 
acts.sigmoid()                                         # Implementation of Sigmoid. 
(acts[:, 0] - acts[:, 1]).sigmoid()                    # Implementation of Sigmoid. 


# In[21]:


#@ INITIALIZING SOFTMAX FUNCTION: 
def softmax(x):                                        # Defining Softmax Function. 
    return exp(x) / exp(x).sum(dim=1, keepdim=True)


# **EXPONENTIAL FUNCTION**
# - **Exponential Function** is defined as e\*\*x where e is a special number approximately equal to 2.718. It is the inverse of natural logarithm function. **Exponential Function** is always positive and increases very rapidly. 

# In[22]:


#@ IMPLEMENTATION OF SOFTMAX FUNCTION: 
sm_acts = torch.softmax(acts, dim=1)                    # Implementation of Softmax. 
sm_acts


# **Note:**
# - **Softmax** is the multicategory equivalent of **Sigmoid**. 

# **INITIALIZING LOG LIKELIHOOD**
# 

# In[23]:


#@ DEFINING THE FUNCTION: 
def mnist_loss(inputs, targets):                         # Initializing the Function. 
    inputs = inputs.sigmoid()                            # Initializing Sigmoid Activation. 
    return torch.where(targets==1, 1-inputs, inputs).\
           mean()                                        # Getting Mean of Loss. 


# In[24]:


#@ EXAMPLE OF LABELS: 
targ = tensor([0, 1, 0, 1, 1, 0])                        # Initializing Tensor. 
idx = range(6)
sm_acts[idx, targ]                                       # Implementation. 


# In[25]:


#@ IMPLEMENTATION OF NEGATIVE LOG LIKEHOOD: 
F.nll_loss(sm_acts, targ, reduction="none")              # Implementation. 


# In[26]:


#@ INSPECTING LOGARITHMIC FUNCTION: 
plot_function(torch.log, min=0, max=4)                   # Inspection. 


# **Note:**
# - When we first take the **Softmax** and then the **Log Likelihood** of that, that combination is called **Cross Entropy Loss**. 

# In[27]:


#@ INITIALIZING CROSS ENTROPY: 
loss_func = nn.CrossEntropyLoss()                       # Instantiation. 
loss_func(acts, targ)                                   # Getting Mean Loss.
F.cross_entropy(acts, targ)                             # Getting Mean Loss. 
nn.CrossEntropyLoss(reduction="none")(acts, targ)       # Getting Loss. 


# **MODEL INTERPRETATION**

# In[28]:


#@ GETTING CONFUSION MATRIX: 
interp = ClassificationInterpretation.from_learner(learn)       # Initializing Interpretation. 
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)          # Plotting Confusion Matrix. 


# In[29]:


#@ GETTING MOST INCORRECT PREDICTIONS: 
interp.most_confused(min_val=3)                                 # Getting Confusion Matrix. 


# **THE LEARNING RATE FINDER**

# In[30]:


#@ TRAINING THE MODEL: PRETRAINED MODEL: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)          # Initializing Convolution Network. 
learn.fine_tune(1, base_lr=0.1)                                 # Training the Model with High LR. 


# In[31]:


#@ TRAINING THE MODEL: PRETRAINED MODEL: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)          # Initializing Convolution Network.
learn.lr_find()                                                 # Getting Learning Rate Finder. 


# In[32]:


#@ TRAINING THE MODEL: PRETRAINED MODEL: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)          # Initializing Convolution Network. 
learn.fine_tune(2, base_lr=0.0007)                              # Training the Model with Optimal. 


# **UNFREEZING AND TRANSFER LEARNING**

# In[34]:


#@ TRANSFER LEARNING: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)          # Initializing Convolution Network. 
learn.fit_one_cycle(3, 3e-3)                                    # Training the ModeL. 


# In[35]:


#@ UNFREEZING THE MODEL: 
learn.unfreeze()                                                # Initializing. 
learn.lr_find()                                                 # Learning Rate Finder. 


# In[36]:


#@ TRAINING THE MODE: OPTIMAL LR: 
learn.fit_one_cycle(6, lr_max=5.2481e-05)                       # Training the Model. 


# **DISCRIMINATIVE LEARNING RATES**

# In[38]:


#@ TRANSFER LEARNING WITH DISCRIMINATIVE LR: 
learn = cnn_learner(dls, resnet34, metrics=error_rate)          # Initializing Convolution Network. 
learn.fit_one_cycle(3, 3e-3)                                    # Training the Model. 
learn.unfreeze()                                                # Unfreezing the Model. 
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4))               # Training the Model with Discriminative LR. 


# In[39]:


#@ INSPECTING LOSS: 
learn.recorder.plot_loss()                                       # Plotting Loss. 


# In[43]:


#@ DEEPER ARCHITECTURES: 
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()     # Initializing Convolutions. 
learn.fine_tune(6, freeze_epochs=3)                                  # Training the Model. 

