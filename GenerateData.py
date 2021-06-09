from typing import cast
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

class DataGenerator():
    def __init__(self):
        self.DATADIR = os.getcwd() + '\\Champions'
        self.TESTING = os.getcwd()+'\\Testing'
        dir = os.getcwd()
        self.categories = []
        with open(dir + "\\champs.txt", encoding="utf8") as file:
            self.categories =  [champ.strip("\n") for champ in file.readlines()]
        self.IMG_SIZE = 150



    def showImages(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            print(path)
            for images in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, images))
                self.IMG_SIZE = 150
                newImg = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                plt.imshow(newImg)
                plt.show()

    def generateTrainingData(self):
        for cat in self.categories:
            # path for the folder
            path = os.path.join(self.DATADIR, cat)
            classNum = self.categories.index(cat)
            for images in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, images))
                    IMG_SIZE = 150
                    newImg = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    self.trainingData.append([newImg, classNum])
                except Exception as e:
                    pass

    def generateImagesAndLabels(self):
        for img, label in self.trainingData:
            self.labels.append(label)
            self.imgs.append(img)
        self.imgs = np.array(self.imgs)

        pickle_out = open("trainX.pickle", "wb")
        pickle.dump(self.imgs, pickle_out)
        pickle_out.close()

        pickle_outY = open("testY.pickle", "wb")
        pickle.dump(self.labels, pickle_outY)
        pickle_out.close()


dg = DataGenerator()