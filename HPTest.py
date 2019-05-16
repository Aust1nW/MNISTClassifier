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
    max_pool = 2

    def __init__(self, layer_type, drop_out, neurons, regularization,
                 max_pool):
        self.drop = drop_out
        self.neurons = neurons
        self.reg = regularization
        self.max_pool = max_pool
        self.type = layer_type


def build_model(model_params):
    model = models.Sequential()
    for x in range(0, len(model_params)):
        neurons = int(model_params[x].neurons)
        max_pool = int(model_params[x].max_pool)
        if model_params[x].type == 'C' and x == 0:
            model.add(layers.Conv2D(neurons, (3, 3), activation='relu',
                                    input_shape=(28, 28, 1)))
        elif model_params[x].type == 'C' and x != 0:
            model.add(layers.Conv2D(neurons, (3, 3), activation='relu'))
        elif model_params[x].type == 'M':
            model.add(layers.MaxPool2D((max_pool, max_pool)))
        elif model_params[x].type == 'F':
            model.add(layers.Flatten())
        elif model_params[x].type == 'X':
            model.add(layers.Dropout(float(model_params[x].drop)))
        elif model_params[x].type == 'D':
            model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# Helper function to print out current configuration
def print_hyper_parameters(model_params, file):
    for layer in model_params:
        if layer.reg == '-1':
            reg = 'None'
        else:
            reg = 'L2(' + layer.reg + ')'
        if layer.type == 'M' or layer.type == 'F':
            continue
        if layer.type == 'X':
            print('dropout ' + layer.drop + ' reg None')
            file.write('\ndropout ' + layer.drop + ' reg None')
            continue
        print('channels: ' + layer.neurons + ' dropout '
              + layer.drop + ' reg ' + reg)

        file.write('\nchannels: ' + layer.neurons + ' dropout '
                   + layer.drop + ' reg ' + reg)


# Returns an ImageDataGenerator object for data augmentation
def build_generator(data):
    image_gen = ImageDataGenerator(
        rotation_range=5,
        shear_range=0.2)
    image_gen.fit(data)

    return image_gen


# Parses config.txt and builds a list of lists with desired model parameters
f = open("config.txt")
modelList = []
# Only reads for 5 configs
for j in range(0, 5):
    configList = []
    for i in range(0, 10):
        line = f.readline()
        lineElements = line.split(",")
        if len(lineElements) == 1:
            break
        configList.append(LyrParams(lineElements[0], lineElements[1],
                                    lineElements[2], lineElements[3],
                                    lineElements[4]))
    modelList.append(configList)
    f.readline()
f.close()

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
callback_list = [es]

train_data_flow = train_data_gen.flow(train_data, train_labels, batch_size=128)

# Open an output file to write to
file_out = open("HPTest.out", "w")

# Run model configs until validation accuracy decreases and print results
for m in modelList:
    nn = build_model(m)
    nn.compile(optimizer='rmsprop',
               loss='categorical_crossentropy', metrics=['acc'])
    hst = nn.fit_generator(train_data_flow,
                           steps_per_epoch=1000,
                           epochs=10,
                           validation_data=(val_data, val_labels),
                           validation_steps=1000,
                           callbacks=callback_list,
                           verbose=1)
    print_hyper_parameters(m, file_out)
    val_acc = hst.history['val_acc'].pop()
    val_loss = hst.history['val_loss'].pop()
    epoch = len(hst.history['val_acc'])
    print('Best validation at epoch ' + '{0:.5f}'.format(epoch) +
          ' with loss ' + str(val_loss) + ' and accuracy ' + str(val_acc))

    file_out.write('\nBest validation at epoch ' + str(epoch) + ' with loss ' +
                   '{0:.5f}'.format(val_loss) + ' and accuracy ' +
                   str(val_acc) + '')
file_out.close()
