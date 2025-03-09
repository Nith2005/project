import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
df = load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,3))
for ax, image, label in zip(axes, df.images, df.target):
  ax.set_axis_off()
  ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title("Training: %i" % label)

df.images.shape

df.images[0]

df.images[0].shape

len(df.images)

n_samples = len(df.images)
data = df.images.reshape((n_samples,-1))

data[0]

data[0].shape

len(df.images)

n_samples = len(df.images)
data = df.images.reshape((n_samples,-1))
data[0]

data[0].shape

data.shape

