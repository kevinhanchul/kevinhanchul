#타이타닉 로직을 알아보자

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

a = pd.read_excel('/workspace/ai_test.xlsx')

a1 = list(a.columns)
keep = a1.pop(0)
# print(keep)
target = a[[keep]]

# a = a.dropna(axis=1, thresh=500)
# print(target)

value_data = a[['b', 'c']]
# print(value_data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(value_data)
value_data = pd.DataFrame(scaled_data, columns=value_data.columns)
# print(value_data)

onehot_data = pd.get_dummies(a,
            columns=a.columns)

# print(onehot_data)

training_data = pd.concat((onehot_data, value_data), axis=1)

# print(training_data)

X_train, X_test, Y_train, Y_test = train_test_split(
    training_data, target, test_size=0.2)

print(X_train.shape, Y_train.shape)

# print(X_test.shape, Y_test.shape)


model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='adam',
              metrics=['binary_accuracy'])

fit_hist = model.fit(
    X_train, Y_train, batch_size=50, epochs=10,
    validation_split=0.2, verbose=1)
