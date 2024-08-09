# Author: Erfan Panahi

# Packages -----------------------------------------------------------------------------------------------------------------------------------------------------------------

from keras.layers import Dense, Flatten, Input, MaxPool2D, Dropout, BatchNormalization, Conv2D, ReLU
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Loading and Preprocessing Data -------------------------------------------------------------------------------------------------------------------------------------------

def load_data(): 
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5), transforms.RandomHorizontalFlip(p=0.5)])

    train_MNIST = datasets.MNIST(root = 'data', transform = tr, train = True, download = True)
    test_MNIST = datasets.MNIST(root = 'data', transform = tr, train = False, download = True)
    train_Fashion = datasets.FashionMNIST(root = 'data', transform = tr, train = True, download = True)
    test_Fashion = datasets.FashionMNIST(root = 'data', transform = tr, train = False, download = True)
    train_CIFAR10 = datasets.CIFAR10(root = 'data', transform = tr, train = True, download = True)
    test_CIFAR10 = datasets.CIFAR10(root = 'data', transform = tr, train = False, download = True)

    random_images(train_MNIST.data, train_MNIST.targets)
    plt.suptitle('Random Images - MNIST Dataset')
    random_images(train_Fashion.data, train_Fashion.targets)
    plt.suptitle('Random Images - Fashion-MNIST Dataset')
    random_images(train_CIFAR10.data, train_CIFAR10.targets)
    plt.suptitle('Random Images - CIFAR10 Dataset')

    return train_MNIST, test_MNIST, train_Fashion, test_Fashion, train_CIFAR10, test_CIFAR10

def random_images(train_images, train_labels):
    plt.figure(figsize = (22,3))
    for i in range(8):
        rnd = np.random.randint(0, len(train_labels))
        plt.subplot(1, 8, i+1)
        plt.imshow(train_images[rnd], cmap = 'grey')
        plt.title('class: {}'.format(train_labels[rnd]))

def prepare_data(train, test):
    x_train = train.data.numpy()
    y_train = train.targets.numpy()
    x_test = test.data.numpy()
    y_test = test.targets.numpy()
    return x_train, to_categorical(y_train), x_test, to_categorical(y_test) 

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Neural Network Functions -------------------------------------------------------------------------------------------------------------------------------------------------

def design_scnnb(type, input_shape, output_shape):
    inputs = Input(shape = input_shape)
    model = inputs
    model = Conv2D(32, (3, 3), input_shape = input_shape)(model)
    if type == 'SCNNB': model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D((2, 2))(model)
    model = Conv2D(64, (3, 3))(model)
    if type != 'SCNNB_b': model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D((2, 2))(model)
    model = Flatten()(model)
    model = Dense(1280, activation = 'relu')(model)
    model = Dropout(rate = 0.5)(model)
    model = Dense(output_shape, activation = 'softmax')(model)
    return Model(inputs, model, name = type)

def train(model, x_train, y_train, x_test, y_test, Epoch_Num):
    opt = Adam(learning_rate = 0.001, weight_decay = 0.00005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])
    History = model.fit(x_train, y_train, 
                        epochs = Epoch_Num,
                        batch_size = 256,
                        validation_data = (x_test, y_test))
    return History

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# SCNNB Class --------------------------------------------------------------------------------------------------------------------------------------------------------------

class SCNNB:
    def __init__(self, input_shape, output_shape, scnnb = None, scnnb_a = None, scnnb_b = None, history_scnnb = None, history_scnnb_a = None, history_scnnb_b = None):
        super().__init__
        self.scnnb = design_scnnb('SCNNB', input_shape, output_shape)
        self.scnnb_a = design_scnnb('SCNNB_a', input_shape, output_shape)
        self.scnnb_b = design_scnnb('SCNNB_b', input_shape, output_shape)
    
    def training(self, x_train, y_train, x_test, y_test, Epoch_Num):
        self.history_scnnb_b = train(self.scnnb_b, x_train, y_train, x_test, y_test, Epoch_Num)
        self.history_scnnb_a = train(self.scnnb_a, x_train, y_train, x_test, y_test, Epoch_Num)
        self.history_scnnb = train(self.scnnb, x_train, y_train, x_test, y_test, Epoch_Num)

         
    def result(self):
        plt.figure(figsize = (12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.history_scnnb.history['loss'], '-.', label = 'SCNNB')
        plt.plot(self.history_scnnb_a.history['loss'], '-.', label = 'SCNNB_a') 
        plt.plot(self.history_scnnb_b.history['loss'], '-.', label = 'SCNNB_b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training Data')
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.plot(self.history_scnnb.history['val_loss'], '-.', label = 'SCNNB')
        plt.plot(self.history_scnnb_a.history['val_loss'], '-.', label = 'SCNNB_a') 
        plt.plot(self.history_scnnb_b.history['val_loss'], '-.', label = 'SCNNB_b')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Validation Data')
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(self.history_scnnb.history['accuracy'], '-.', label = 'SCNNB')
        plt.plot(self.history_scnnb_a.history['accuracy'], '-.', label = 'SCNNB_a') 
        plt.plot(self.history_scnnb_b.history['accuracy'], '-.', label = 'SCNNB_b')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid()
        plt.subplot(2, 2, 4)
        plt.plot(self.history_scnnb.history['val_accuracy'], '-.', label = 'SCNNB')
        plt.plot(self.history_scnnb_a.history['val_accuracy'], '-.', label = 'SCNNB_a') 
        plt.plot(self.history_scnnb_b.history['val_accuracy'], '-.', label = 'SCNNB_b')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid()

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

