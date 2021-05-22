import os
from NeuroPy import NeuroPy as mp
from time import sleep
from pynput import keyboard
import time


neuropy = mp.NeuroPy(port='COM3')
jump_name = 'jump'
run_fire_name = 'run_fire'
file_name = f'./results_{time.time_ns()}.csv'
time_name = 'timestamp_ns'
interval = 0.1


class Attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return str(dict(self))


results = Attrdict()
results[jump_name] = 0
results[run_fire_name] = 0


def print_object_methods(o):
    object_methods = [method_name for method_name in dir(o)
                      if callable(getattr(o, method_name))]
    print(object_methods)


def set_callback(neuropy, name, print_result=False, function=None):
    def default_function(value):
        results[name] = value
        if print_result:
            print(name, value)
    results[name] = 0
    selected_function = function if function else default_function
    neuropy.setCallBack(name, selected_function)


def on_press(key):
    try:
        if key.char is 'x':
            results[jump_name] = 1
        elif key.char is 'z':
            results[run_fire_name] = 1
    except AttributeError:
        pass


def on_release(key):
    try:
        if key.char is 'x':
            results[jump_name] = 0
        elif key.char is 'z':
            results[run_fire_name] = 0
    except AttributeError:
        pass


try:
    with open(file_name, 'a') as file:
        start_time = time.time_ns()
        results[time_name] = 0

        # attention,meditation,rawValue,delta,theta,lowAlpha,highAlpha,lowBeta,highBeta,lowGamma,midGamma, poorSignal, blinkStrength

        # set_callback(neuropy, "attention")
        # set_callback(neuropy, "meditation")
        set_callback(neuropy, "rawValue")
        set_callback(neuropy, "delta")
        set_callback(neuropy, "theta")
        set_callback(neuropy, "lowAlpha")
        set_callback(neuropy, "highAlpha")
        set_callback(neuropy, "lowBeta")
        set_callback(neuropy, "highBeta")
        set_callback(neuropy, "lowGamma")
        set_callback(neuropy, "midGamma")
        set_callback(neuropy, "poorSignal")
        # set_callback(neuropy, "blinkStrength")
        neuropy.start()

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()

        if os.stat(file_name).st_size == 0:
            file.write(','.join(dict(results).keys()) + '\n')
            file.flush()

        keys = list(dict(results).keys())
        index = 0
        time_since = 0.0
        output_buffer = ''

        while True:
            sleep(interval)
            time_diff = time.time_ns() - start_time
            results[time_name] = time_diff
            tmp = ','.join([str(x) for x in list(dict(results).values())])
            file.write(tmp + '\n')
            file.flush()
            print('\r', (str(time_diff) + ":").ljust(20), output_buffer, end='')
            if time_since > 1:
                time_since = 0.0
                output_buffer = f'{keys[index].ljust(15)} {results[keys[index]]}'
                index = (index + 1) % len(keys)
            time_since += interval
finally:
    neuropy.stop()
    listener.stop()
