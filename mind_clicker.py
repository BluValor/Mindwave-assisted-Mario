import os
from NeuroPy import NeuroPy as mp
from time import sleep
from pynput import keyboard
import time
import pyautogui
from attributes_holder import *


prev_result = 0


def on_press(key):
    global prev_result
    try:
        if key.char is 'z' and prev_result == 0:
            prev_result = 1
            pyautogui.keyDown('space')
    except AttributeError:
        if key == keyboard.Key.space:
            print("space pressed")


def on_release(key):
    global prev_result
    try:
        if key.char is 'z' and prev_result == 1:
            prev_result = 0
            pyautogui.keyUp('space')
    except AttributeError:
        if key == keyboard.Key.space:
            print("space released")


try:
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    while True:
        sleep(20)
        pyautogui.press("space")
finally:
    listener.stop()


