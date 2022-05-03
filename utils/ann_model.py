import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Activation


# def learning_rate():
#     lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=0.001,
#         decay_steps=10000,
#         decay_rate=0.0000000001)
#     return lr_schedule

# def optimizer(learning_rate):
#     opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     return opt


def ann_model(x_train):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1:])))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mae", optimizer='adam')
    return model