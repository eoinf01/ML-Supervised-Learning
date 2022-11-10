# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

from sklearn import datasets
from sklearn import tree
from sklearn import linear_model
from sklearn import neighbors
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
        print("#### PERCEPTON CLASSIFIER ####")
        clf = linear_model.Perceptron()
    elif clf_type == "SVM":
        print("#### SVM RADIAL BASIS CLASSIFIER ####")
        clf = svm.SVC(kernel="rbf",gamma=clf_variable)
    elif clf_type == "KNN":
        print("#### KNN CLASSIFIER ####")
        clf = neighbors.KNeighborsClassifier(n_neighbors=clf_variable)
    elif clf_type == "DTC":
        clf = tree.DecisionTreeClassifier()
    else:
        clf = linear_model.Perceptron()
    fold = 0
    for train_index, test_index in kf.split(newData, newTarget):
        fold += 1
        print("Current K-Fold: ",fold)
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

    return training_time,prediction_time,np.mean(accuracy)


def q3():
    input_size = []
    training_eval =[]
    prediction_eval = []
    accuracy_svc = []

    training, prediction, accuracy = q2("Perceptron", len(data), data, target, 0)
    print("Mean prediction accuracy: ",accuracy)
    for x in range(1000,len(data),1000):
        training,prediction,accuracy = q2("Perceptron", 14000, data, target,0)
        input_size.append(x)
        accuracy_svc.append(accuracy)
        training_eval.append(np.max(training))
        prediction_eval.append(np.max(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.show()
    print("\nMean prediction accuracy of percepton classifier: ",np.mean(accuracy_svc))

def q4():
    gamma_ranges = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    accucaries = []
    input_size= []

    for x in gamma_ranges:
        training, prediction, accuracy = q2("SVM", 3000, data, target, x)
        accucaries.append(np.mean(accuracy))
    best_accuracy = np.max(accucaries)
    best_y =  gamma_ranges[accucaries.index(best_accuracy)]
    print("\nBest Y value: ",best_y)
    print("Best average classification accuracy: ", best_accuracy)

    training_eval = []
    prediction_eval = []
    for x in range(1000, len(data), 1000):
        input_size.append(x)
        training, prediction, accuracy = q2("SVM", x, data, target, best_y)
        training_eval.append(np.mean(training))
        prediction_eval.append(np.mean(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.show()

def q5():
    accucaries = []
    k_values = [*range(1,15)]
    print(k_values)
    input_size = []
    for k in k_values:
        training, prediction, accuracy = q2("KNN", len(data), data, target, k)
        accucaries.append(np.mean(accuracy))
    best_accuracy = np.max(accucaries)
    best_k = k_values[accucaries.index(best_accuracy)]

    print("\nBest K value: ", best_k)
    print("Best mean classification accuracy: ", best_accuracy)

    training_eval = []
    prediction_eval = []
    for x in range(1000, len(data), 1000):
        input_size.append(x)
        training, prediction, accuracy = q2("KNN", x, data, target, best_k)
        training_eval.append(np.mean(training))
        prediction_eval.append(np.mean(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data,target = q1()
    q5()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
