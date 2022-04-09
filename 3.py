

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



column_name = ['a', 'b', 'c', 'd', 'e','f', 'g', 'h']

raw_data = pd.read_excel('/workspace/ai_test.xlsx',
                         header=None, names=column_name)

# print(raw_data.head(20))
# print(raw_data.describe().T)

# raw_data.info()
clean_data = raw_data.replace('?', np.nan)
clean_data = clean_data.dropna()

# print(clean_data.info())
# print(clean_data)

keep = column_name.pop()

print(keep)

training_data = clean_data[column_name]

target = clean_data[[keep]]

# print(training_data)

# print(target)

# print(training_data.head())

# print(target.head())

# print(target['HeartDisease'].sum())

# print(target['HeartDisease'].mean())

# print(scaled_data.describe().T)

# scaled_data.boxplot(column=column_name,

#                     showmeans=True)

# plt.show()

from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
#
# scaled_data = scaler.fit_transform(training_data)
#
# # print(type(scaled_data))
#
# scaled_data = pd.DataFrame(scaled_data, columns=column_name)
# #
# # print(scaled_data.head())
#
# # print(scaled_data)
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, Y_train, Y_test = train_test_split(
#
#     scaled_data, target, test_size=0.30)

# print(X_train)
#
# print('X_train :', X_train.shape)

# print('X_test :', X_test.shape)
#
# print('Y_train :', Y_train.shape)
#
# print('Y_test :', Y_test.shape)
#
#
#
#
#
# model = Sequential()
#
# model.add(Dense(512, input_dim=13, activation='relu'))
#
# model.add(Dropout(0.25))
#
# model.add(Dense(256, activation='relu'))
#
# model.add(Dropout(0.2))
#
# model.add(Dense(128, activation='relu'))
#
# model.add(Dropout(0.1))
#
# model.add(Dense(1, activation='sigmoid'))
#
# model.summary()
#
#
#
#
#
# model.compile(loss='mse', optimizer='adam',
#
#               metrics=['binary_accuracy'])
#
#
#
# fit_hist = model.fit(X_train, Y_train,
#
#             batch_size=50, epochs=10,
#
#             validation_split=0.2, verbose=1)
#

