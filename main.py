import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals.joblib import dump, load

class BinaryClassification(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassification, self).__init__()
       
        self.layer_1 = nn.Linear(input_size, 128) 
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


def load_model(model_path, scl_path):
    model_cls = BinaryClassification(1399)
    model_cls.load_state_dict(torch.load(model_path))
    model_cls.eval()
    
    scl = load(scl_path)
    
    return model_cls, scl

def predict(model, scl, inpt):
    inpt = inpt.reshape(1, -1)
    inpt = scl.transform(inpt)
    inpt = torch.FloatTensor(inpt)
    out = model(inpt)
    y_test_pred = torch.sigmoid(out)
    y_pred_tag = torch.round(y_test_pred)
    
    return y_pred_tag.item()

model, scaler = load_model('model.pth', 'scaler.scl')

data = np.random.normal(size=(1399))

predict(model, scaler, data)