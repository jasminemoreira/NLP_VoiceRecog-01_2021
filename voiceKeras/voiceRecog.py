"""
#pip install tensorflow-gpu
pip install tensorflow
pip install keras
pip install librosa
pip install matplotlib
"""

import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
with tf.device("gpu:0"):
   print("tf.keras code in this scope will run on GPU")

os.chdir("C:\\Users\\jasmi\\OneDrive\\Área de Trabalho\\PLN RV\\voiceKeras")
from preprocess import *

feature_dim_2 = 11
save_data_to_array(max_len=feature_dim_2)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = get_train_test()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train_hot, batch_size=100, epochs=100, verbose=1, validation_data=(X_test, y_test_hot))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

sample = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
sample_reshaped = sample.reshape(1, 20, 11, 1)
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
])







