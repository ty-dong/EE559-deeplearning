import torch
from torch import Tensor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import numpy as np

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels]

def generate_disk_dataset(n_points ,one_hot_labels=True):
    input = Tensor(n_points, 2).uniform_(0,1)
    center = Tensor([1/2,1/2]).view(1, -1)
    label = ((input - center).norm(p=2, dim=1) < 1/math.sqrt(2*math.pi)).long()
    if(one_hot_labels):
        label = one_hot_embedding(label, 2)
    return input, label

input, label = generate_disk_dataset(1000, True)

def plot_with_labels(input, labels, ax):
    colors = []
    for b in labels:
        if (b==1):
            colors.append('r')
        else:
            colors.append('b')
    ax.scatter(input[:,0],input[:,1],color = colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
train_label = label.max(dim = 1)[1].long()
fig_train, axes = plt.subplots(nrows=1, ncols=1, figsize=(8,8), sharex=True)
plot_with_labels(input, train_label, axes)
