import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

tf.__version__

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], 'float32')
target_data = np.array([[0],[1],[1]],[0])

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(4 ,input_dim=2 ,activation = "relu"))
model.add(tf.keras.layers.Dense(1 , activation = "sigmoid"))


model.compile(loss='mean_squared_error',
	optimizer='adam',
	metrics=['binary accuracy'])

model.summary()


history = model.fit(tarining_data, target_data, nb_epoch=500, verbose=2)

print(model.predict(tarining_data).round())

