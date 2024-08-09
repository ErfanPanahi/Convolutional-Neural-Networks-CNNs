The use of deep neural networks for image classification has seen significant advancements. However, these networks are usually very deep and require substantial computational power and system memory. Additionally, the depth of these networks results in very long training times. In the attached paper ([Part1.pdf](https://github.com/ErfanPanahi/Convolutional-Neural-Networks-CNNs/blob/main/Q1/Part1.pdf)), a new architecture is proposed that achieves accuracy nearly comparable to deep convolutional neural networks, but with fewer layers.

Before starting the explanations, I should mention that I use the `Keras` library to implement the networks related to this section.

The other packages used for processing and reading the datasets are added as follows.

```python
# Packages -----------------------------------------------------------------------------------------------------------------------------------------------------------------

from keras.layers import Dense, Flatten, Input, MaxPool2D, Dropout, BatchNormalization, Conv2D, ReLU
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Data preparation and preprocessing

According to the explanations provided in the article, we perform the desired preprocessing on the data from the three datasets: MNIST, Fashion-MNIST, and CIFAR10. For this purpose:
* We flip the data horizontally with a probability of 0.5 ($p=0.5$).
* We normalize the data. For this, we consider the mean and variance of the final data to be 0.5 ($\mu = \sigma = 0.5$). This ensures that the distribution of pixels falls within a similar range, making it easier and more comprehensible for the model to learn features. It also prevents phenomena such as vanishing gradients or exploding gradients.

The functions for reading the datasets and preprocessing are as follows:

```python
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
```

```python
train_MNIST, test_MNIST, train_Fashion, test_Fashion, train_CIFAR10, test_CIFAR10 = load_data()
x_train_mnist, y_train_mnist_cat, x_test_mnist, y_test_mnist_cat = prepare_data(train_MNIST, test_MNIST)
x_train_fashion, y_train_fashion_cat, x_test_fashion, y_test_fashion_cat = prepare_data(train_Fashion, test_Fashion)
x_train_cifar10, y_train_cifar10_cat, x_test_cifar10, y_test_cifar10_cat = train_CIFAR10.data, to_categorical(train_CIFAR10.targets), test_CIFAR10.data, to_categorical(test_CIFAR10.targets) 
```

A number of random images from each dataset are shown in **Figure 1-1**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/44afee26-bded-4f7f-8e97-bcb8ae7c1ef8" width="1000" height="500" >
</p>

## Explanation of the various layers of the network architecture

The architecture of the network, which is also referred to as SCNNB in the article, is shown in **Figure 1-2**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f22414f-40dc-4c63-b3c3-d6c9bf04bad9" width="300" height="400" >
</p>

Additionally, two other versions of the network, named SCNNB-a and SCNNB-b, are defined according to **Figure 2-3**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ca8f1e15-931a-4b5f-8d53-61c8bcf4f0d5" width="500" height="200" >
</p>

The function written for the architecture of the SCNNB networks is as follows.

```python
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

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
I define the following `class` for all three networks related to a dataset. This class is intended to simplify the process and provide easy access to the network information for each dataset.

```python
class SCNNB:
    def __init__(self, input_shape, output_shape, scnnb = None, scnnb_a = None, scnnb_b = None, history_scnnb = None, history_scnnb_a = None, history_scnnb_b = None):
        super().__init__
        self.scnnb = design_scnnb('SCNNB', input_shape, output_shape)
        self.scnnb_a = design_scnnb('SCNNB_a', input_shape, output_shape)
        self.scnnb_b = design_scnnb('SCNNB_b', input_shape, output_shape)
```

The architecture of the SCNNB network for the two datasets, MNIST and CIFAR10, is shown in **Figure 1-4**. (The difference is in the size and dimension of the input data to the network.)

```python
mnist = SCNNB((28, 28, 1), 10)
mnist.scnnb.summary()
```

```python
cifar10 = SCNNB((32, 32, 3), 10)
cifar10.scnnb.summary()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/efa63a70-ca7b-40cb-bdde-762309618e58" width="700" height="370" >
</p>

The architecture of the SCNNB-a network is shown in **Figure 1-5**.

```python
mnist.scnnb_a.summary()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/03dadbdf-970c-4c4d-8206-c39548584435" width="350" height="350" >
</p>

The architecture of the SCNNB-b network is shown in **Figure 1-6**.

```python
mnist.scnnb_b.summary()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d66bb6a-fe9a-4fce-ae05-571d4244d4a1" width="350" height="350" >
</p>



## Implementation of the architecture

The following function is related to training a network.

```python
# Neural Network Functions -------------------------------------------------------------------------------------------------------------------------------------------------

def train(model, x_train, y_train, x_test, y_test, Epoch_Num):
    opt = Adam(learning_rate = 0.001, weight_decay = 0.00005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])
    History = model.fit(x_train, y_train, 
                        epochs = Epoch_Num,
                        batch_size = 256,
                        validation_data = (x_test, y_test))
    return History

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

The following function is related to training all three networks for each type of dataset (within a `class`).

```python
    def training(self, x_train, y_train, x_test, y_test, Epoch_Num):
        self.history_scnnb_b = train(self.scnnb_b, x_train, y_train, x_test, y_test, Epoch_Num)
        self.history_scnnb_a = train(self.scnnb_a, x_train, y_train, x_test, y_test, Epoch_Num)
        self.history_scnnb = train(self.scnnb, x_train, y_train, x_test, y_test, Epoch_Num)
```

Finally, I will attempt to **train** the designed networks using the data for each dataset.

```python
mnist.training(x_train_mnist, y_train_mnist_cat, x_test_mnist, y_test_mnist_cat, 25)
```

```python
fashion.training(x_train_fashion, y_train_fashion_cat, x_test_fashion, y_test_fashion_cat, 50)
```

```python
cifar10.training(x_train_cifar10, y_train_cifar10_cat, x_test_cifar10, y_test_cifar10_cat, 50)
```

## Implementation results

Finally, we present the implementation results as a function, showing the **loss** and **accuracy** of each model during the learning process.


The following function shows the results related to the learning of each dataset (within the `class`).

```python
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
```


The results of the trained model using the `MNIST` dataset are shown in **Figure 1-7**.



```python
mnist.result()
plt.suptitle('Training Results (Loss and Accuracy) - MNIST Dataset')
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/57d0a60b-db5e-4c20-8a4a-f0beada1d8fc" width="1000" height="750" >
</p>

The results of the trained model using the `Fashion-MNIST` dataset are shown in **Figure 1-8**.


```python
fashion.result()
plt.suptitle('Training Results (Loss and Accuracy) - Fashion-MNIST Dataset')
```


<p align="center">
  <img src="https://github.com/user-attachments/assets/32672377-b449-4dfa-a570-6a72d8d88b29" width="1000" height="750" >
</p>

The results of the trained model using the `CIFAR10` dataset are shown in **Figure 1-9**.


```python
cifar10.result()
plt.suptitle('Training Results (Loss and Accuracy) - CIFAR10 Dataset')
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/304b2e31-1d4a-4614-bce1-1b03701825ae" width="1000" height="750" >
</p>

