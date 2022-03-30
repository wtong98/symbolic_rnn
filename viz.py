"""
Inspect the operation of the model

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import matplotlib.pyplot as plt
import numpy as np

from model import *

# %%
model = BinaryAdditionLSTM()
model.load('save/mini')

# <codecell>
### PLOT EMBEDDINGS
weights = model.embedding.weight.data.numpy()
plt.scatter(weights[:,0], weights[:,1], c=['b', 'r', 'y', 'g', 'k'])


# %%
