# MNISTClassifier

Implementation of the classic MNIST classifier, uses data augmentation and
callbacks to test different hyperparameters in config.txt and prints out
training accuracy/loss and validation accuracy/loss to HPTest.out

HPGenerate outputs MNIST.h5 which contains the trained model including weights

Using MNIST.h5, MNISTClassify outputs prediction labels for the standard test set
provided by the classic MNIST data set

Highest final accuracy is 99.43% +-0.2 depending on HPGenerate output
