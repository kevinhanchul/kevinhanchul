from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
import matplotlib.pyplot as plt

# def celsius_to_fahrenheit(x):
#     return x + 32   #=> 기본 내용만,,

a = []
a.append([0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56])
a.append([0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56, 0.23, 0.56])
# data_F = [0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58 0.12 0.58]

print(a)
# print(data_F)

# scaled_data_C = data_C / 100
# print(scaled_data_C)
