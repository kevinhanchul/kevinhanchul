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

raw_data = sns.load_dataset('titanic')
print(raw_data.head())
print(raw_data.isnull().sum())

clean_data = raw_data.dropna(axis=1, thresh=500)
print(clean_data.columns)

mean_age = clean_data['age'].mean()
print(mean_age)

clean_data['age'].fillna(mean_age, inplace=True)
print(clean_data.head())

clean_data.drop(['embark_town', 'alive'], axis=1, inplace=True)

print(clean_data.info())

clean_data['embarked'].fillna(
    method='ffill', inplace=True)

print(clean_data.isnull().sum())

label = list(clean_data.columns)
keep = label.pop(0)
target = clean_data[[keep]]
training_data = clean_data[label]

print(training_data.head())
print(target.head())

value_data = training_data[['age', 'fare']]

print(value_data.head())



scaler = StandardScaler()

scaled_data = scaler.fit_transform(value_data)

value_data = pd.DataFrame(scaled_data, columns=value_data.columns)

print(value_data.describe())



training_data.info()

training_data.drop(['age', 'fare'], axis=1, inplace=True)

training_data.pclass.unique()



onehot_data = pd.get_dummies(training_data,

            columns=training_data.columns)



print(onehot_data.head())

print(onehot_data.info())



training_data = pd.concat((onehot_data, value_data), axis=1)

print(training_data.info())





X_train, X_test, Y_train, Y_test = train_test_split(

    training_data, target, test_size=0.2)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)



model = Sequential()

model.add(Dense(128, input_dim=34, activation='relu'))

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





plt.plot(fit_hist.history['binary_accuracy'])

plt.plot(fit_hist.history['val_binary_accuracy'])

plt.show()



score = model.evaluate(X_test, Y_test, verbose=0)

print('loss', score[0])

print('accuracy', score[1])



