# 삼성전자 주가 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pickle

raw_data = pd.read_csv('/datasets/samsung200129_220328.csv')
print(raw_data.head())
raw_data.info()

raw_data['Date'] = pd.to_datetime(raw_data['Date'])
raw_data.set_index('Date', inplace=True)
print(raw_data.head())

minmaxscaler = MinMaxScaler()
scaled_data = minmaxscaler.fit_transform(raw_data)
print(scaled_data[:5])
print(scaled_data.shape)

sequence_X = []
sequence_Y = []

for i in range(len(scaled_data)-30):
    x = scaled_data[i:i+30]
    y = scaled_data[i+30][3]
    sequence_X.append(x)
    sequence_Y.append(y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)
print(sequence_X[0])
print(sequence_Y[0])
print(sequence_X.shape)
print(sequence_Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    sequence_X, sequence_Y, test_size=0.2)
xy = X_train, X_test, Y_train, Y_test
np.save('./samsung_preprocessed_30.npy', xy)

with open('./samsung_minmaxscaler.pickle', 'wb') as f:
    pickle.dump(minmaxscaler, f)

with open('./samsung_minmaxscaler.pickle', 'rb') as f:
    minmaxscaler = pickle.load(f)

model = Sequential()
model.add(LSTM(50, input_shape=(30, 6),
               activation='tanh'))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=30)
fit_hist = model.fit(X_train, Y_train, batch_size=128,
        epochs=500, callbacks=[early_stopping], verbose=1,
        validation_data=(X_test,Y_test), shuffle=False)

plt.plot(fit_hist.history['loss'][-450:], label='loss')
plt.plot(fit_hist.history['val_loss'][-450:], label='val_loss')
plt.legend()
plt.show()

pred = model.predict(X_test)

plt.plot(Y_test[:30], label='actual')
plt.plot(pred[:30], label='predict')
plt.legend()
plt.show()

last_data_30 = scaled_data[-30:].reshape(1, 30, 6)
today_close = model.predict(last_data_30)

print(today_close)

today_close_value = minmaxscaler.inverse_transform(
    [[0, 0, 0, today_close, 0, 0]])[0][3]
print(today_close_value)

today_actual = [[70000.0, 70300.0, 69800.0, 70200.0, 70200.0, 13686208]]
scaled_today_actual = minmaxscaler.transform(today_actual)[0]
print(scaled_today_actual)

last_data_29 = scaled_data[-29:]
last_30_data = np.append(last_data_29, scaled_today_actual)

last_30_data = last_30_data.reshape(1, 30, 6)
print(last_30_data.shape)

tomorrow_pred = model.predict(last_30_data)
tomorrow_predicted_value = minmaxscaler.inverse_transform(
    [[0, 0, 0, tomorrow_pred, 0, 0]])[0][3]
print('%d 원'%tomorrow_predicted_value)

