from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from sys import exit

nn = models.Sequential()
nn.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # Output shape (.., 26, 26, 32)
nn.add(layers.MaxPooling2D((2, 2)))                  # Output shape (:, 13, 13, 32)
nn.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output shape (:, 11, 11, 64)
nn.add(layers.MaxPooling2D((2, 2)))                  # Output shape (:, 5, 5, 64)
nn.add(layers.Conv2D(64, (3, 3), activation='relu')) # Output shape (:, 3, 3, 64)
nn.add(layers.Flatten())                             # Output shape (:, 576)
nn.add(layers.Dense(64, activation='relu'))          # Output shape (:, 64)
nn.add(layers.Dense(10, activation='softmax')) 

# Process it all, configure parameters, and get ready to train
nn.compile(
    optimizer="rmsprop",             # Improved backprop algorithm
    loss='categorical_crossentropy', # "Misprediction" measure
    metrics=['accuracy']             # Report CCE value as we train
)

(train_data, train_labels), (test_data, test_labels) \
 = mnist.load_data()
 
# Create validation set
val_data, val_labels = train_data[:10000], train_labels[:10000]
train_data, train_labels = train_data[10000:], train_labels[10000:]

# Reorganize all data as sets of flat vectors of 0-1.0 values
train_data = train_data.reshape((50000, 28, 28, 1)).astype('float32')/255
val_data = val_data.reshape((10000, 28, 28, 1)).astype('float32')/255
test_data = test_data.reshape((10000, 28, 28, 1)).astype('float32')/255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

hst = nn.fit(train_data, train_labels, epochs = 4, batch_size = 64,
             validation_data = (val_data, val_labels))

hst = hst.history
x_axis = range(len(hst['acc']))

plt.plot(x_axis, hst['acc'], 'bo')
plt.plot(x_axis, hst['val_acc'], 'ro')
plt.show()

nn.save('MNIST.model')
