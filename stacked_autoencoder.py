# Stacked Autoencoder

# 0 - Import the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# 1 - Importing the dataset
ratings = pd.read_csv("ml-1m/ratings.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv("ml-1m/users.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# 2 - Preparing the training- and test set.
""" The dataset has been split into 5 subsets for the
purpose of conducting a k-fold cross validation later."""

training_set = pd.read_csv("ml-100k/u1.base", delimiter = '\t')
training_set = np.array(training_set, dtype = "int")

test_set = pd.read_csv("ml-100k/u1.test", delimiter = '\t')
test_set = np.array(test_set, dtype = "int")

# 3 - Getting the number of users and movies
nb_users = int(max(max(training_set[:,0])), max(max(test_set[:,0])))
nb_movies = max(max(test_set[:,0]))