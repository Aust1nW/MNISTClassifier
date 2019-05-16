import numpy as np
from keras.datasets import mnist
from keras.models import load_model


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
nn = load_model('MNIST.h5')

test_data = test_data.reshape(
    (len(test_labels), 28, 28, 1)).astype('float32')/255

output_labels = nn.predict(test_data)

for i in output_labels:
    print(np.argmax(i))
