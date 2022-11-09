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

def q2(clf_type,sample_size,data,target,clf_variable):
    target = target.to_numpy()
    sample_indexes = np.random.randint(data.shape[0],size=sample_size)
    newData = data[sample_indexes, :]
    newTarget = target[sample_indexes]

    kf = model_selection.KFold(n_splits=5)
    training_time = []
    prediction_time = []
    accuracy = []
    if clf_type == "Perceptron":
        clf = linear_model.Perceptron()
    elif clf_type == "SVM":
        clf = svm.SVC(gamma=clf_variable)
    else:
        clf = linear_model.Perceptron()

    for train_index, test_index in kf.split(newData, newTarget):
        startTrain = time.time()
        clf.fit(newData[train_index], newTarget[train_index])
        endTrain = time.time()
        trainTime = endTrain - startTrain
        startPrediction = time.time()
        prediction = clf.predict(newData[test_index])
        endPrediction = time.time()
        predictionTime = endPrediction-startPrediction
        confusion = metrics.confusion_matrix(newTarget[test_index],prediction)
        print(confusion)
        accuracyMetric = metrics.accuracy_score(newTarget[test_index],prediction)
        training_time.append(trainTime)
        prediction_time.append(predictionTime)
        accuracy.append(accuracyMetric)
    print("Max training time",max(training_time))
    print("Min training time",min(training_time))
    print("average training time",np.mean(training_time))
    print("Max prediction time", max(prediction_time))
    print("Min prediction time", min(prediction_time))
    print("Average prediction time", np.mean(prediction_time))
    print("Max accuracy score",max(accuracy))
    print("Min accuracy score",min(accuracy))
    print("Average accuracy score",np.mean(accuracy))

    return training_time,prediction_time,accuracy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data,target = q1()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
