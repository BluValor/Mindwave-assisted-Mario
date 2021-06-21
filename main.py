import numpy as np
import torch
import torch.nn as nn
import sklearn
from joblib import dump, load

import os
from NeuroPy import NeuroPy as mp
from time import sleep
from pynput import keyboard
import time
import pyautogui
from attributes_holder import *


history_len = 40
history_params_order = ['rawValue', 'theta', 'delta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']


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
    model_cls = BinaryClassification(history_len * len(history_params_order))
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


model, scaler = load_model('model_no_jump_hist40.pth', 'scaler_no_jump_hist40.scl')
neuropy = mp.NeuroPy(port='COM4')
results = Attrdict()


def set_callback(neuropy, name, print_result=False, function=None):
    def default_function(value):
        results[name] = value
        if print_result:
            print(name, value)
    results[name] = 0
    selected_function = function if function else default_function
    neuropy.setCallBack(name, selected_function)


is_pressed_manually = False


def on_press(key):
    try:
        if key.char is 'v':
            print("pressed", end="")
            is_pressed_manually = True
    except AttributeError:
        pass


def on_release(key):
    try:
        if key.char is 'v':
            print()
            is_pressed_manually = False
    except AttributeError:
        pass


historic_data = {}
for param in history_params_order:
    historic_data[param] = []

try:
    for param in history_params_order:
        set_callback(neuropy, param)
    neuropy.start()

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()

    keys = list(dict(results).keys())

    for i in range(history_len):
        sleep(interval)
        for param in history_params_order:
            historic_data[param] += [results[param]]

    count = 0
    space_down = False
    # counter = 0

    while True:
        sleep(interval)
        data = []
        for param in history_params_order:
            historic_data[param].pop(0)
            historic_data[param] += [results[param]]
            data += historic_data[param]
        data = np.array(data)
        result = predict(model, scaler, data)

        if result == 1.0:
            count += 1
            if count > 2 and not space_down:
                pyautogui.keyDown('space')
                space_down = True
                if not is_pressed_manually:
                    print()
                count = 0
            # counter += 1
            # print(counter)
            print("-", end="")
        else:
            count = 0
            if space_down:
                pyautogui.keyUp('space')
                space_down = False
finally:
    neuropy.stop()
    listener.stop()
