import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def hankel_matrix(x,seq_len):
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-seq_len+1, seq_len), strides=(stride,stride)).copy()

def triangle_wave(t, m=10000):
    arange = np.arange(1, m+1)
    return 8/(np.pi**2)*np.sum(np.sin(arange*np.pi/2)*np.sin(arange*t)/(arange**2))

seed = 0
hidden_unit = 100
random.seed(seed)
np.random.seed(seed)

# dataset
N = 500
x_n = []
ts = np.linspace(0, 500, N)
for t in ts:
    tmp = triangle_wave(t)
    x_n.append(tmp)
x_n = np.asarray(x_n)
seq_len = 100
H = hankel_matrix(x_n, seq_len=seq_len)
X = H[:-1,:].reshape(-1, seq_len, 1)
Y = H[1:,-1]

# plot data
plt.plot(x_n[:50])
plt.savefig('data.png')
plt.show()

cp_cb = ModelCheckpoint(filepath = 'model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# stateless
train_model = Sequential()
train_model.add(LSTM(hidden_unit, input_shape=(seq_len, 1)))
train_model.add(Dense(1))
train_model.compile(loss='mean_squared_error', optimizer='adam')
train_model.fit(X, Y, epochs=20, callbacks = [cp_cb], validation_split=0.3, shuffle=True)

# stateful
predict_model = Sequential()
predict_model.add(LSTM(hidden_unit, batch_input_shape=(1, 1, 1), stateful=True))
predict_model.add(Dense(1))
predict_model.load_weights('model.hdf5')


x_n = []
xs = []
ts = np.linspace(500, 1000, N)
for t in ts:
    tmp = triangle_wave(t)
    x_n.append(tmp)
    xs.append(tmp)
xs = xs[:seq_len]

# test input
for o in xs:
    p = predict_model.predict(np.asarray([o]).reshape(1, 1, 1))[0,0]

# predict
predict_len = 200
xs.append(p)
for i in range(predict_len-1):
    p = predict_model.predict(np.asarray([p]).reshape(1, 1, 1))[0,0]
    xs.append(p)

# plot predict
plt.figure(figsize=(16, 4))
plt.plot(xs[:seq_len], label='input value')
plt.plot(np.arange(predict_len) + seq_len, x_n[seq_len:][:predict_len], label='true value')
plt.plot(np.arange(predict_len) + seq_len, xs[seq_len:], label='lstm predict value')
plt.xlabel('time')
plt.legend(loc='upper right')
plt.savefig('fig.png')
plt.show()