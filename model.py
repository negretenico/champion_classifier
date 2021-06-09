import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, \
    GlobalAveragePooling2D
from tensorflow.python.client import device_lib
from GenerateData  import  DataGenerator
import warnings
warnings.filterwarnings('ignore')
class CNN:
    def __init__(self):
        self.dg = DataGenerator()
        self.names = self.dg.categories
        pickle_in = open("trainX.pickle", "rb")
        self.images = pickle.load(pickle_in)

        pickle_in = open("testY.pickle", "rb")
        self.labels = pickle.load(pickle_in)
        self.data_augmentation = Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",
                                                 input_shape=(  self.dg.IMG_SIZE,
                                                                self.dg.IMG_SIZE,3)),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.1),

  ]
)
        self.batch_size =64
        n_filters  = 32
        self.model = Sequential([ 
                                self.data_augmentation,
                                #preprocess
                                layers.experimental.preprocessing.Rescaling(1. / 255),

                                #first layer
                                layers.Conv2D(filters=n_filters, kernel_size=(9,9), activation=tf.nn.relu),
                                layers.MaxPool2D(pool_size=(2,2)),

                                #second layer layer
                                layers.Conv2D(filters=2*n_filters, kernel_size=(9,9), activation=tf.nn.relu),
                                layers.MaxPool2D(pool_size=(2,2)),

                                #third layer
                                layers.Conv2D(filters=4*n_filters, kernel_size=(9,9), activation=tf.nn.relu),
                                layers.MaxPool2D(pool_size=(2,2)),

                                #fourth layer
                                layers.Conv2D(filters=6*n_filters, kernel_size=(9,9), activation=tf.nn.relu),
                                layers.MaxPool2D(pool_size=(2,2)),
                               
                                #fifth layer
                                layers.Flatten(),
                                layers.Dense(n_filters*8, activation=tf.nn.relu),

                                #final layer
                                layers.Dense(len(self.names),activation = tf.nn.softmax)
                                  ])
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()

    def training(self,epochs):
        with tf.device('/gpu:0'):
            history =  self.model.fit(self.images, np.array(self.labels), batch_size=self.batch_size, epochs=epochs, validation_split=0.3)
        return history
    def evaluate(self):
        self.model.evaluate(self.images,np.array(self.labels))

    def viewPredictions(self):
        predictions = self.model.predict(self.images)
        plt.figure(figsize=(150, 150))
        classes = np.argmax(predictions, axis=1)
        self.images = self.images.squeeze()
        print(f"Accuracy is {np.average(classes ==self.labels)}")
        for i in range(len(classes)):
            plt.grid(False)
            plt.imshow( self.images[i], cmap=plt.cm.binary)
            plt.xlabel("Actual: " + self.names[self.labels[i]])
            plt.title("Prediction: " + self.names[classes[i]])
            plt.show()
    def save(self, path):
        self.self.model.save(path)
    def predict(self,images):
        return self.model.predict(images)


def test(model):
    pickle_in = open("testingImages.pickle", "rb")
    images = pickle.load(pickle_in)
    pickle_in = open("testingLabels.pickle", "rb")
    labels = pickle.load(pickle_in)
    names = ['Basements', 'BathRoom', 'BedRoom', 'DiningRoom', 'Kitchen', 'Living Room']
    predictions = model.predict(images)
    classes = np.argmax(predictions, axis=1)
    print(f"Accuracy is {np.average(classes ==labels)}")
    for i in range(len(classes)):
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + names[labels[i]])
        plt.title("Prediction: " + names[classes[i]])
        plt.show()

#model = tf.keras.models.load_model('saved_model/new_model1')
model = CNN()
epochs = 2000
history = model.training(epochs)
#plot(epochs,history)
model.model.save('saved_model/new_model1')
model.evaluate()
model.viewPredictions()