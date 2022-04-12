# x_train에는 총 60000개의 28×28 크기의 이미지가 담겨 있으며


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import datasets
from keras.utils import np_utils


# a = pd.read_excel('/workspace/ai_test.xlsx')


(X_train, Y_train), (X_test, Y_test) = datasets.mnist.load_data()

print('=====   X_train의 첫번째 shape   =====')
print(X_train[0].shape)

print('=====   X_train, Y_train의 전체 shape   =====')
print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

print('=====   my_sample 데이타   =====')
my_sample = np.random.randint(60000)
print(my_sample)

print('=====   X_train 데이타중에 하나 그림으로  =====')
plt.imshow(X_train[my_sample], cmap='gray')
plt.show()

print('=====   Y_train 데이타중에 하나   =====')
print(Y_train[my_sample])

print('=====   X_train 데이타중에 하나   =====')
print(X_train[my_sample])

y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

print('=====   y_train 데이타중에 하나, 5,000번째 데이타   =====')
print(y_train[5000])
print('=====   y_test 데이타중에 하나, 5,000번째 데이타   =====')
print(y_test[5000])


x_train = X_train.reshape(-1, 28 * 28)  # 28*28(=>784)로 앞에 행은 알아서 나오도록 => 데이타는 행열로 정리가 되어야,,
x_test = X_test.reshape(-1, 28 * 28)
x_train = x_train / 255
x_test = x_test / 255

print('=====   x_train reshape하고, 255로 나눈 데이타   =====')
print(x_train.shape)

model = Sequential()
model.add(Dense(128, input_dim= 28 * 28,
                activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

print('=====   model 썸머리 데이타   =====')
print(model.summary())

fit_hist = model.fit(x_train, y_train, batch_size=128,
    epochs=15, validation_split=0.2, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set loss :', score[0])
print('Final test set accurecy :', score[1])

# plt.plot(fit_hist.history['acc'])
# plt.plot(fit_hist.history['val_acc'])
# plt.show()


my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample], cmap='gray')
print(Y_test[my_sample])
pred = model.predict(x_test[my_sample].reshape(-1, 28 * 28))   #여기에서 결과값을 확인함
print(pred)
print(np.argmax(pred))
