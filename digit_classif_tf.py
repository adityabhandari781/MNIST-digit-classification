import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
warnings.simplefilter(action='ignore', category=FutureWarning)


n_hidden = 25
epochs = 40

X = np.load("data/X.npy")
y = np.load("data/y.npy")
m, n = X.shape
n_labels = len(np.unique(y))

model = Sequential([
    InputLayer((n,)),
    Dense(n_hidden, activation='relu'),
    Dense(n_labels, activation='linear'),
], name="my_model")

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(X, y, epochs=epochs, verbose=2)

y_pred = np.argmax(model.predict(X), axis=1).reshape(-1, 1)

print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, zero_division=1))