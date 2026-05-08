# essential
import os
import numpy as np
import pandas as pd
import re
import logging
import time
import warnings
import argparse
import itertools
import json
import math, random, csv
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score


# visualization
import matplotlib.pyplot as plt
import seaborn as sns


# torch
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import subgraph, to_scipy_sparse_matrix
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree, to_undirected
from torch.utils.data import ConcatDataset



# for graph, signal
import networkx as nx
import mne
from scipy.signal import resample
from scipy.signal import welch
from scipy.signal import coherence, hilbert, butter, filtfilt
from scipy.stats import skew, kurtosis, entropy
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score

