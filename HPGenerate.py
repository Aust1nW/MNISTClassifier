import keras
from keras import models, layers
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


class LyrParams:
    drop = 1.0
    neurons = 32
    reg = -1
    maxPool = 2

    def __init__(self, drop_out, neurons, regularization, max_pool):
        self.drop = drop_out
        self.neurons = neurons
        self.reg = regularization
        self.maxPool = max_pool


# Builds model based on best config found in HPTest.py
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3),
                            activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# Helper function to print out current configuration
def print_hyper_parameters(model_params):
    for layer in model_params:
        if layer.reg == '-1':
            reg = 'None'
        else:
            reg = 'L2(' + layer.reg + ')'
        print('channels: ' + layer.neurons + ' dropout '
              + layer.drop + ' reg ' + reg)


# Returns an ImageDataGenerator object for data augmentation
def build_generator(data):
    image_gen = ImageDataGenerator(
        rotation_range=3,
        zoom_range=0.1)
    image_gen.fit(data)
    return image_gen


# Load data and create validation set
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
val_data, val_labels = train_data[:10000], train_labels[:10000]
train_data, train_labels = train_data[10000:], train_labels[10000:]

# Reorganize all data as sets of flat vectors of 0-1.0 values
train_data = train_data.reshape((50000, 28, 28, 1)).astype('float32')/255
val_data = val_data.reshape((10000, 28, 28, 1)).astype('float32')/255


# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)


# Data augmentation
train_data_gen = build_generator(train_data)


# Early stopping to stop the model once validation accuracy decreases
es = EarlyStopping(monitor='val_acc', mode='max', verbose=0)
train_data_flow = train_data_gen.flow(train_data, train_labels, batch_size=128)
nn = build_model()
# Run model
nn.compile(optimizer='adam', loss='categorical_crossentropy',
           metrics=['acc'])
hst = nn.fit_generator(train_data_flow,
                       steps_per_epoch=1000,
                       epochs=10,
                       validation_data=(val_data, val_labels),
                       validation_steps=1000,
                       callbacks=[es],
                       verbose=1)

# Save the model to MNIST.h5
nn.save('MNIST.h5')
