# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

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

dict={}

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

def q2(clf_type,sample_size,data,target):
    target = target.to_numpy()
    sample_indexes = np.random.randint(data.shape[0],size=sample_size)
    newData = data[sample_indexes, :]
    newTarget = target[sample_indexes]

    kf = model_selection.KFold(n_splits=5)
    training_time = []
    prediction_time = []
    accuracy = []
    clf = clf_type
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

##Percepton Classifier Evaluation
def q3():
    input_size = []
    training_eval =[]
    prediction_eval = []
    accuracy_svc = []

    training, prediction, accuracy = q2(linear_model.Perceptron(), len(data), data, target)
    dict["Perceptron"] = {
        "training": 0,
        "prediction": 0,
        "accuracy": 0
    }
    dict["Perceptron"]["training"] = training
    dict["Perceptron"]["prediction"] = prediction
    dict["Perceptron"]["accuracy"] = accuracy

    print("Mean prediction accuracy: ",accuracy)
    for x in range(1000,len(data),1000):
        training,prediction,accuracy = q2(linear_model.Perceptron(), x, data, target)
        input_size.append(x)
        accuracy_svc.append(accuracy)
        training_eval.append(np.max(training))
        prediction_eval.append(np.max(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.title("Perceptron Times & Sample Size Relationship")
    plt.legend(loc='upper left')
    plt.show()
    print("\nMean prediction accuracy of percepton classifier: ",np.mean(accuracy_svc))

#SVM Classifier Evaluation
def q4():
    gamma_ranges = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    accucaries = []
    input_size= []
    best_training = []
    best_prediction = []
    for x in gamma_ranges:
        #Change the len(data) to another sample size if taking too long to run
        training, prediction, accuracy = q2(svm.SVC(kernel="rbf",gamma=x), len(data), data, target)
        best_training.append(training)
        best_prediction.append(prediction)
        accucaries.append(np.mean(accuracy))
    best_accuracy = np.max(accucaries)
    dict["SVM"] = {
        "training": 0,
        "prediction": 0,
        "accuracy": 0
    }
    dict["SVM"]["training"] = best_training[accucaries.index(best_accuracy)]
    dict["SVM"]["prediction"] = best_prediction[accucaries.index(best_accuracy)]
    dict["SVM"]["accuracy"] = best_accuracy

    best_y = gamma_ranges[accucaries.index(best_accuracy)]
    print("\nBest Y value: ",best_y)
    print("Best average classification accuracy: ", best_accuracy)

    training_eval = []
    prediction_eval = []
    for x in range(1000, len(data), 1000):
        input_size.append(x)
        training, prediction, accuracy = q2(svm.SVC(kernel="rbf",gamma=best_y), x, data, target)
        training_eval.append(np.mean(training))
        prediction_eval.append(np.mean(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.legend(loc='upper left')

    plt.title("SVM Times & Sample Size Relationship")
    plt.show()

#Nearest Neighbours Classifier Evaluation
def q5():
    accucaries = []
    k_values = [*range(1,15)]
    print(k_values)
    input_size = []
    best_training = []
    best_prediction = []
    for k in k_values:
        training, prediction, accuracy = q2(neighbors.KNeighborsClassifier(n_neighbors=k), len(data), data, target)
        accucaries.append(accuracy)
        best_training.append(training)
        best_prediction.append(prediction)

    best_accuracy = np.max(accucaries)
    best_k = k_values[accucaries.index(best_accuracy)]
    dict["KNN"] = {
        "training": 0,
        "prediction": 0
    }
    dict["KNN"]["training"] = best_training[accucaries.index(best_accuracy)]
    dict["KNN"]["prediction"] = best_prediction[accucaries.index(best_accuracy)]
    dict["KNN"]["accuracy"] = best_accuracy

    print("\nBest K value: ", best_k)
    print("Best mean classification accuracy: ", best_accuracy)

    training_eval = []
    prediction_eval = []
    for x in range(1000, len(data), 1000):
        input_size.append(x)
        training, prediction, accuracy = q2(neighbors.KNeighborsClassifier(n_neighbors=best_k), x, data, target)
        training_eval.append(np.mean(training))
        prediction_eval.append(np.mean(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.legend(loc='upper left')

    plt.title("KNN Times & Sample Size Relationship")
    plt.show()

#Decision Tree Classifier Evaluation
def q6():
    training, prediction, accuracy = q2(tree.DecisionTreeClassifier(), len(data), data, target)
    dict["DTC"] = {
        "training": 0,
        "prediction": 0,
        "accuracy": 0
    }
    dict["DTC"]["training"] = training
    dict["DTC"]["prediction"] = prediction
    dict["DTC"]["accuracy"] = accuracy

    print("Classification Accuracy: ",accuracy)
    input_size = []
    training_eval= []
    prediction_eval = []
    for x in range(1000, len(data), 1000):
        input_size.append(x)
        training, prediction, accuracy = q2(tree.DecisionTreeClassifier(), x, data, target)
        training_eval.append(np.mean(training))
        prediction_eval.append(np.mean(prediction))

    plt.plot(input_size, training_eval, label="Training times")
    plt.plot(input_size, prediction_eval, label="Test times")
    plt.xlabel("Sample Size")
    plt.ylabel("Runtimes")
    plt.legend(loc='upper left')

    plt.title("DTC Times & Sample Size Relationship")
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dict = {}
    data,target = q1()
    q3()
    q4()
    q5()
    q6()
    q7()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
