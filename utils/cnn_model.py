from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.layers import Conv1D, Dense,Flatten, Input,BatchNormalization
#from utils.ann_model import learning_rate, optimizer

def cnn_model(x_train, loss, optimizer, num_classes, kernel_size):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(x_train.shape[1:])))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model