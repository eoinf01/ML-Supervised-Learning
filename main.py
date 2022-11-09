# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def q1():
    data = pd.read_csv('product_images.csv')
    target = data["label"]
    data.drop('label',inplace=True,axis=1)
    data = data.to_numpy()
    print(type(target))
    print("Number of samples of sneakers: ",len(target[target==0]))
    print("Number of samples of ankle boots: ",len(target[target==1]))

    for digit in set(target):
        plt.figure()
        index = random.randint(0, sum(target == digit) - 1)
        plt.imshow(data[target == digit][index].reshape(28, 28))
    plt.show()

    return data,target


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data,target = q1()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
