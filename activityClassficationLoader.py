import pickle

from matplotlib.ticker import FormatStrFormatter

from activityDataClass import activityDataClass
from activityDataRemodeler import activityDataRemodeler
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.client import device_lib
from keras import backend as K
from keras import optimizers
from keras.models import Model, load_model, save_model
import tensorflow as tf
import keras_metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pydotplus
from IPython.display import Image
import graphviz
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd
from sklearn.model_selection import GridSearchCV








############### VARIABLES #####################
storePath = "./activityClassificationData/labeledActivities.dat"
numLabels = 8
validationPercentage = 0.09
classPrecisionList = []
classRecallList = []
classF1List = []
confMatrixList = []

############### NN ############################

# Setting up a Keras convolutional model of: 4 Conv and Pool + Flat
def setupImageConvModel(inputShape):

    model = Sequential()



    # Convolution Layer 1
    convLayer = Conv2D(filters=12,
                       kernel_size=(11, 11),
                       strides=(4, 4),
                       activation='relu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3,4,5
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='elu')
    model.add(convLayer)
    convLayer = Conv2D(filters=24,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='relu')
    model.add(convLayer)
    convLayer = Conv2D(filters=24,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Dense Layer
    model.add(Dense(100, activation="relu"))

    # Flatten
    model.add(Flatten())

    return model

# This is the complete model which uses both the conv model output and the metadata input
def setupFinalModel(imageInputShape, metadataInputShape, outputShape):

    # First of all, we prepare the Convolutional model for the image classification
    imageModel = setupImageConvModel(imageInputShape)

    # Then, we prepare the Metadata model
    metadataModel = Sequential()
    metadataModel.add(Dense(metadataInputShape, activation="relu", input_shape=(metadataInputShape,)))

    # Eventually generating the final Merged model
    #mergedModel = Sequential()
    #mergedModel.add(Concatenate([metadataModel, imageModel]))
    merged = concatenate([metadataModel.output, imageModel.output])
    merged = Dense(50, activation="relu")(merged)
    # merged = Dropout(0.1, noise_shape=None, seed=None)(merged)
    # merged = Dense(50, activation="relu")(merged)
    out = Dense(outputShape, activation="softmax")(merged)
    mergedModel = Model([metadataModel.input, imageModel.input], out)


    mergedModel.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()]
    )

    mergedModel.summary()

    return mergedModel

def setupDataOnlyModel(metadataInputShape, outputShape):
    model = Sequential()
    # Input - Layer
    model.add(Dense(metadataInputShape, activation="relu", input_shape=(metadataInputShape,)))
    # Hidden - Layers
    # model.add(Dropout(0.20, noise_shape=None, seed=None))
    model.add(Dense(40, activation="relu"))
    # model.add(Dropout(0.1, noise_shape=None, seed=None))
    # model.add(Dense(5, activation="relu"))
    # model.add(Dense(50, activation="relu"))
    # model.add(Dense(5, activation="relu"))
    # Output- Layer
    model.add(Dense(outputShape, activation="softmax"))  # Was using sigmoid, but it is only between 0 and 1
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()]
    )

    return model


def singleTraining(imageShape, metadataShape, numLabels):

    # Instantiating model
    model = setupFinalModel(imageShape, metadataShape, numLabels)
    # model = setupDataOnlyModel(metadataShape, numLabels)

    # Training the model
    results = model.fit(
        x=[metaInputList, imageInputList],
        y=labelInputList,
        verbose=1,
        epochs=200,
        batch_size=24,
        shuffle=True,
        validation_split=validationPercentage
    )

    # results = model.fit(
    #                         x = metaInputList,
    #                         y = labelInputList,
    #                         verbose=1,
    #                         epochs=1000,
    #                         batch_size=32,
    #                         shuffle=True,
    #                         validation_split = validationPercentage
    #                         )

    # Plot metrics
    #  "Accuracy"
    plt.plot(results.history['acc'])
    plt.plot(results.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper left')
    plt.show()

    # Printing the validation results
    print("\n\nAverage Validation accuracy: " + str(np.mean(results.history["val_acc"])))
    # Finding best validation accuracy
    print("Best validation accuracy: " + str(max(results.history['val_acc'])))

    print("Average validation Precision and Recall: " + str(np.mean(results.history["val_precision"])) + " " + str(
        np.mean(results.history["val_recall"])))
    print("Best validation Precision and Recall: " + str(max(results.history["val_precision"])) + " " + str(
        max(results.history["val_recall"])))

    # prediction = model.predict(np.array([metaInputList[-1],]))
    # print("Prediction is actually " + str(prediction))

# HYP: value of K
def kNearestNeighbors(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage)
    # Features scaling to normalize dimensions
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # # Calculating error for K values between 1 and 40
    # accuracies = []
    # error = []
    # for i in range(1, 40):
    #     # Setting the value of K and fitting the training data
    #     knn = KNeighborsClassifier(n_neighbors=i)       # n_neighbors is the value of K
    #     knn.fit(X_train, y_train)
    #     # Predict labels for all the testing data
    #     pred_i = knn.predict(X_test)
    #     # Calculate the accuracy value for this K
    #     acc = accuracy_score(y_test, pred_i)
    #     accuracies.append(acc)
    #     print("Accuracy for K=" + str(i) + " is " + str(acc))
    #     # Calculate error for the chart
    #     error.append(np.mean(pred_i != y_test))
    #
    # # Plot the error for all the values of K
    # plt.figure(figsize=(12, 6))
    # plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Error Rate K Value')
    # plt.xlabel('K Value')
    # plt.ylabel('Mean Error')
    # plt.show()
    #
    # # Calculating baccuracy for best value of K
    # bestAccuracy = max(accuracies)
    # print("\nHighest accuracy that can be reached by the model is: " + str(bestAccuracy))

    # ##---------------------------##
    # knn = KNeighborsClassifier(n_neighbors=5, weights='distance')  # n_neighbors is the value of K
    # knn.fit(X_train, y_train)
    # # Predict labels for all the testing data
    # pred_i = knn.predict(X_test)
    # # Calculate the accuracy value for this K
    # accuracy = accuracy_score(y_test, pred_i)
    # precision = precision_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))
    # recall = recall_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))

    # Grid Search
    knnModel = KNeighborsClassifier()
    n_neighbors = [x for x in range(1,40)]
    metric = ['euclidean', 'hamming', 'canberra', 'manhattan']
    gridValues = {'n_neighbors': n_neighbors, 'metric': metric, 'weights': ['distance']}
    gridModel = GridSearchCV(knnModel, param_grid=gridValues, scoring='accuracy', cv=5)
    gridModel.fit(X_train, y_train)
    print('---> Best params after inner cross validation: ' + str(gridModel.best_params_))
    # Predicting the output
    y_pred = gridModel.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred, )
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # Plotting metrics
    # plot_grid_search(gridModel.cv_results_, n_neighbors, metric, 'Num Neighbors', 'Distance Metric')

    return accuracy, precision, recall

# HYP: num trees
def randomForest(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage)

    # # Evaluating performance of the model for various numbers of trees
    # accuraciesTrain = []
    # accuraciesTest = []
    # for i in range(2,60):
    #     # Defining the Random Forest Model
    #     forestModel = RandomForestClassifier(n_estimators=i)  # n_estimators is the number of trees
    #     # Training the model
    #     forestModel.fit(X_train, y_train)
    #     # Evaluating
    #     accuraciesTrain.append(forestModel.score(X_train, y_train))
    #     accuraciesTest.append(forestModel.score(X_test, y_test))
    #     print('Random Forest accuracy: TRAINING', forestModel.score(X_train, y_train))
    #     print('Random Forest accuracy: TESTING', forestModel.score(X_test, y_test))
    #
    # # Plotting results
    # # NOTE: The results of this chart are used to determine the correct minimum value of n_estimators (trees)
    # plt.plot(accuraciesTrain)
    # plt.plot(accuraciesTest)
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('n_estimators')
    # plt.legend(['Training set', 'Testing set'], loc='upper right')
    # plt.show()


    ##---------------------------##
    forestModel = RandomForestClassifier(n_estimators=25)  # n_estimators is the number of trees
    # Training the model
    forestModel.fit(X_train, y_train)
    # Predict labels for all the testing data
    pred_i = forestModel.predict(X_test)
    # Calculate the accuracy value for this K
    accuracy = accuracy_score(y_test, pred_i)
    precision = precision_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))
    recall = recall_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))

    # # Grid Search
    # forestModel = RandomForestClassifier()
    # num_estimators = [x for x in range(2, 40, 2)]
    # max_features = [2,5,8,12,16]
    # min_samples_leaf = [2,4,6,8,10]
    # gridValues = {'n_estimators': num_estimators, 'min_samples_leaf': min_samples_leaf, 'max_features':max_features}
    # gridModel = GridSearchCV(forestModel, param_grid=gridValues, scoring='accuracy', cv=5)
    # gridModel.fit(X_train, y_train)
    # print('---> Best params after inner cross validation: ' + str(gridModel.best_params_))
    # # Predicting the output
    # y_pred = gridModel.predict(X_test)
    # # Calculating metrics
    # accuracy = accuracy_score(y_test, y_pred, )
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # # Plotting metrics
    # # plot_grid_search(gridModel.cv_results_, max_depth, min_samples_leaf, 'Maximum Tree Depth', 'Minimum Samples at Leaves')
    # # plot_cv_results(gridModel.cv_results_, 'n_estimators', 'min_samples_leaf')

    # # Calculating the importance for each feature
    importances = forestModel.feature_importances_
    # Returns the standard deviation, a measure of the spread of a distribution, of the array element
    # In this specific case, it is referred to how the importance of a feature changes among all of the trees (estimators)
    std = np.std([tree.feature_importances_ for tree in forestModel.estimators_], axis=0)
    # Sorting the indices of the features basing upon their importance
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for i in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (i + 1, indices[i], importances[indices[i]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.ylabel('Importance')
    plt.xlabel('Feature ID')
    plt.show()

    return accuracy, precision, recall


def decisionTree(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)

    # # Trying different hyperparams
    # accuracies = []
    # for max_feature in range(2,16):
    #     treeModel = DecisionTreeClassifier(criterion='gini', max_features=max_feature)
    #     treeModel.fit(X_train, y_train)
    #     y_predict = treeModel.predict(X_test)
    #     acc = accuracy_score(y_test, y_predict)
    #     accuracies.append(acc)
    # # Plot the error for all the values of K
    # plt.figure(figsize=(12, 6))
    # plt.plot(accuracies, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    # plt.title('Minimum Samples Required for a Leaf Node')
    # plt.xlabel('Number of Features')
    # plt.ylabel('Accuracy')
    # plt.show()

    # # Defining the Decision tree
    # treeModel = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=5, min_samples_leaf=2)
    # # Fitting the training data
    # treeModel.fit(X_train, y_train)
    # # Predict the test data
    # y_predict = treeModel.predict(X_test)
    # accuracy = accuracy_score(y_test, y_predict)
    # precision = precision_score(y_test, y_predict, average='weighted', labels=np.unique(y_predict))
    # recall = recall_score(y_test, y_predict, average='weighted', labels=np.unique(y_predict))
    # print("\nAccuracy with Decision Tree is " + str(accuracy))
    # print("\nPrecision with Decision Tree is " + str(precision))
    # print("\nRecall with Decision Tree is " + str(recall))
    # features = [i for i in range(1,len(X_train[-1])+1)]
    # # # Visualizing tree
    # # dot_data = tree.export_graphviz(treeModel, out_file=None, class_names=features)
    # # graph = graphviz.Source(dot_data)
    # # graph.render("./outputFiles/decisionTreeVisualization")

    # Grid Search
    decisionTreeModel = DecisionTreeClassifier()
    max_depth = [x for x in range(1, 20)]
    min_samples_split = [2, 5, 10, 15]
    min_samples_leaf = [2,3,4,5,6]
    max_features = [2,4,6,8,10,12]
    gridValues = {'criterion':['gini'], 'max_features':max_features, 'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf, 'max_depth': max_depth}
    gridModel = GridSearchCV(decisionTreeModel, param_grid=gridValues, scoring='accuracy', cv=5)
    gridModel.fit(X_train, y_train)
    print('---> Best params after inner cross validation: ' + str(gridModel.best_params_))
    # Predicting the output
    y_pred = gridModel.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred, )
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # Plotting metrics
    # plot_grid_search(gridModel.cv_results_, max_depth, min_samples_leaf, 'Maximum Tree Depth', 'Minimum Samples at Leaves')
    # plot_cv_results(gridModel.cv_results_, 'max_depth', 'max_features')
    # Returning the metrics as results
    return accuracy, precision, recall

# HYP: kernel type (linear, polynomial, gaussian, etc), regularization (C parameter), gamma value
def supportVectorMachine(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)

    # Features scaling to normalize dimensions
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # ## LINEAR KERNEL ##
    # accuracies = []
    # precisions = []
    # recalls = []
    # # Trying different values of C parameter
    # for i in range(1, 50):
    #     svmClassifier = svm.SVC(kernel='linear', C=i, gamma='scale')  # Linear Kernel
    #     # Train the model using the training set
    #     svmClassifier.fit(X_train, y_train)
    #     # Predict the response for test dataset
    #     y_pred = svmClassifier.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # plt.plot(accuracies)
    # plt.title('SVM w/ Linear C values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C value')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()

    # #---------------------------##
    # svmClassifier = svm.SVC(kernel='linear', C=5)  # Linear Kernel
    # # Train the model using the training set
    # svmClassifier.fit(X_train, y_train)
    # # Predict the response for test dataset
    # y_pred = svmClassifier.predict(X_test)
    # # Calculating accuracy by comparing actual test labels and predicted labels
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # return accuracy, precision, recall
    # #---------------------------##

    # POLYNOMIAL KERNEL ##
    # # Try different degrees for the polynomial
    # accuracies = []
    # precisions = []
    # recalls = []
    # for i in range(1, 15):
    #     # Create a new SVM Classifier
    #     svmClassifier = svm.SVC(kernel='poly', degree=i, gamma='scale')  # Polynomial kernel for which we have to specify the degree
    #     # Train the model using the training set
    #     svmClassifier.fit(X_train, y_train)
    #     # Predict the response for test dataset
    #     y_pred = svmClassifier.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # xaxis = [x for x in range(1,15)]
    # plt.plot(xaxis, accuracies)
    # plt.title('SVM w/ Polynomial Kernel degrees')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Polynomial Degree')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()
    # print(accuracies)

    # Try different C values for the polynomial kernel
    # accuracies = []
    # precisions = []
    # recalls = []
    # for i in range(2, 50):
    #     # Create a new SVM Classifier
    #     svmClassifier = svm.SVC(kernel='poly', degree=2, gamma='auto',
    #                             C=i)  # Polynomial kernel for which we have to specify the degree
    #     # Train the model using the training set
    #     svmClassifier.fit(X_train, y_train)
    #     # Predict the response for test dataset
    #     y_pred = svmClassifier.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # plt.plot(accuracies)
    # plt.title('SVM w/ Polynomial C values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C value')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()
    #
    # ##---------------------------##
    # svmClassifier = svm.SVC(kernel='poly', C=30, degree=2, gamma='scale')  # Linear Kernel
    # # Train the model using the training set
    # svmClassifier.fit(X_train, y_train)
    # # Predict the response for test dataset
    # y_pred = svmClassifier.predict(X_test)
    # # Calculating accuracy by comparing actual test labels and predicted labels
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # return accuracy, precision, recall
    # ##---------------------------##


    # # RBF KERNEL ##
    # accuracies = []
    # precisions = []
    # recalls = []
    # # Trying different values of Gamma parameter
    # for i in range(1, 100):
    #     svmClassifier = svm.SVC(kernel='rbf', C=8, gamma='scale')                 # Gaussian Kernel
    #     # Train the model using the training set
    #     svmClassifier.fit(X_train, y_train)
    #     # Predict the response for test dataset
    #     y_pred = svmClassifier.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # plt.plot(accuracies)
    # plt.title('SVM w/ RBF gamma values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('gamma')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()

    # ##---------------------------##
    # svmClassifier = svm.SVC(kernel='rbf', C=4, gamma='scale')  # Linear Kernel
    # # Train the model using the training set
    # svmClassifier.fit(X_train, y_train)
    # # Predict the response for test dataset
    # y_pred = svmClassifier.predict(X_test)
    # # Calculating accuracy by comparing actual test labels and predicted labels
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # return accuracy, precision, recall
    # ##---------------------------##

    # Grid Search
    svmModel = svm.SVC()
    C =  np.logspace(-2, 2, 13)
    gamma = np.logspace(-7, 2, 16)
    degree = [2,3,4,5,6]
    kernel = ['linear', 'poly', 'rbf']
    gridValues = {'gamma': gamma, 'degree':degree, 'kernel':['poly'], 'C':C, 'kernel':kernel}
    gridModel = GridSearchCV(svmModel, param_grid=gridValues, scoring='accuracy', cv=5)
    gridModel.fit(X_train, y_train)
    print('---> Best params after inner cross validation: ' + str(gridModel.best_params_))
    # Predicting the output
    y_pred = gridModel.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred, )
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # Plotting metrics
    # plot_grid_search(gridModel.cv_results_, C, kernel, 'C Value', 'kernel')
    plot_cv_results(gridModel.cv_results_, 'gamma', 'degree')
    return accuracy, precision, recall




# Multinomial Naive Bayes
def naiveBayes(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)

    # Create a Gaussian Classifier
    nbModel = MultinomialNB()
    # Train the model using the training sets
    nbModel.fit(X_train, y_train)
    # Predict Output
    y_pred = nbModel.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred,)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # print("Naive Bayes accuracy is: " + str(accuracy))
    # print("Naive Bayes precision is: " + str(precision))
    # print("Naive Bayes recall is: " + str(recall))

    # Returning the metrics as results
    return accuracy, precision, recall


def logisticRegression(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)

    # Instantiating the logistic Regression model
    logisticRegr = LogisticRegression(solver='newton-cg', multi_class='ovr', C=1)
    # Fitting the model
    logisticRegr.fit(X_train, y_train)
    # Predicting the output
    y_pred = logisticRegr.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred, )
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    confMatrixList.append(confusion_matrix(y_test, y_pred, labels=[x for x in range(1,9)]))

    # # Grid Search
    # logisticRegr = LogisticRegression()
    # C = np.logspace(-4, 4, 20)
    # gridValues = {'C': C, 'multi_class': ['ovr'], 'solver': ['newton-cg','liblinear','lbfgs']}
    # gridModel = GridSearchCV(logisticRegr, param_grid=gridValues, scoring='accuracy', cv=5)
    # gridModel.fit(X_train, y_train)
    # print('---> Best params after inner cross validation: ' + str(gridModel.best_params_))
    # # Predicting the output
    # y_pred = gridModel.predict(X_test)
    # # Calculating metrics
    # accuracy = accuracy_score(y_test, y_pred, )
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # classMetricsCalculator(confusion_matrix(y_test, y_pred, labels=[x for x in range(1,9)]))
    # Plotting metrics
    # plot_grid_search(gridModel.cv_results_, C, kernel, 'C Value', 'kernel')
    # plot_cv_results(gridModel.cv_results_, 'C', 'solver')

    # # Appending per-class metrics
    # precClass, recClass, fscoreClass, support = score(y_test, y_pred)
    # classPrecisionList.append(precClass)
    # classRecallList.append(recClass)
    # classF1List.append(fscoreClass)

    # print("Logistic Regression accuracy is: " + str(accuracy))
    # print("Logistic Regression precision is: " + str(precision))
    # print("Logistic Regression recall is: " + str(recall))

    # # Try different C values for the Regression
    # C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
    # trainingAccuracies = []
    # accuracies = []
    # precisions = []
    # recalls = []
    # for i in C_param_range:
    #     # Instantiating the logistic Regression model
    #     logisticRegr = LogisticRegression(solver='newton-cg', multi_class='ovr', C=i)
    #     # Fitting the model
    #     logisticRegr.fit(X_train, y_train)
    #     # Predicting the output
    #     y_pred = logisticRegr.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     trainingAccuracies.append(accuracy_score(y_train, logisticRegr.predict(X_train)))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # plt.plot(C_param_range, accuracies)
    # plt.plot(C_param_range, trainingAccuracies)
    # plt.xscale('log')
    # plt.title('Logistic Regression C values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C Value')
    # plt.legend(['Test Accuracy', 'Training Accuracy'], loc='upper left')
    # plt.show()

    # # Saving model to file
    # filehandler = open('./savedKerasModels/my_model.dat', 'wb')
    # pickle.dump(logisticRegr, filehandler, protocol=2)
    # filehandler.close()
    # input()
    # filehandler = open('./savedKerasModels/my_model.dat','rb')
    # logisticRegr = pickle.load(filehandler)
    # y_pred = logisticRegr.predict(X_test)
    # # Calculating metrics
    # accuracy = accuracy_score(y_test, y_pred, )
    # precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # input()

    # Returning the metrics as results
    # print("Final Accuracy " + str(accuracy))
    return accuracy, precision, recall

def classMetricsCalculator(confusionMat):

    # Each cell of those arrays corresponds to a class
    fullIterationPrecision = []
    fullIterationRecall = []
    # Getting the confusion Matrix
    print("\n")
    print(confusionMat)
    print("\n")
    # Calculating metrics for each class separately
    for cat in range(0,len(confusionMat)):
        # Retrieving TP, FP, FN
        truePos = confusionMat[cat,cat]
        falsePos = sumColumn(confusionMat, cat) - confusionMat[cat,cat]
        falseNeg = sumRow(confusionMat, cat) - confusionMat[cat,cat]
        # Calculating actual metrics
        precision = truePos/(truePos+falsePos)
        if np.math.isnan(precision):
            precision = -1
        recall = truePos/(truePos+falseNeg)
        if np.math.isnan(recall):
            recall = -1
        # Saving metrics
        fullIterationPrecision.append(precision)
        fullIterationRecall.append(recall)
        print("CLASS " + str(cat) + " has Prec " + str(precision) + " and Recall " + str(recall))
    # Adding the just calculated metrics to the global array containing metrics for all iterations
    classPrecisionList.append(fullIterationPrecision)
    classRecallList.append(fullIterationRecall)


def calculateMetricsFromConfMat(confusionMat):
    # Calculating metrics for each class separately
    truePos = 0
    falsePos = 0
    falseNeg = 0
    for cat in range(0, len(confusionMat)):
        # Retrieving TP, FP, FN
        truePos = truePos + confusionMat[cat, cat]
        falsePos = falsePos + sumColumn(confusionMat, cat) - confusionMat[cat, cat]
        falseNeg = falseNeg + sumRow(confusionMat, cat) - confusionMat[cat, cat]
    # Calculating actual metrics
    precision = truePos / (truePos + falsePos)
    recall = truePos / (truePos + falseNeg)
    print("+++ CONF MAT: Prec " + str(precision) + " and Recall " + str(recall))

def sumColumn(m, column):
    total = 0
    for row in range(len(m)):
        total += m[row][column]
    return total

def sumRow(m, row):
    total = 0
    for col in range(len(m)):
        total += m[row][col]
    return total

def sumMatrices(mat1, mat2):
    # iterate through rows
    for i in range(len(mat1)):
        # iterate through columns
        for j in range(len(mat1)):
            mat1[i][j] = mat1[i][j] + mat2[i][j]
    return mat1

def Nfold(N, metaInputList, labelInputList, MLfunction):
    # Prepare the range of indexes for Cross Validation
    kfold = KFold(n_splits=N, shuffle=True)      #n_splits is the number of folds
    # Iterating through those ranges of indexes
    counter = 0
    totAcc = 0
    totPrec = 0
    totRecall = 0
    totConfMatrix = None
    for train_index, test_index in kfold.split(metaInputList):
        # Increasing iteration counter
        counter = counter + 1
        print("KFOLD ::: We are in iteration " + str(counter) + ":")
        # Actually splitting the training and test data
        X_train, X_test = metaInputList[train_index], metaInputList[test_index]
        y_train, y_test = labelInputList[train_index], labelInputList[test_index]
        # Executing the classification function
        acc, prec, recall = MLfunction(metaInputList, labelInputList, alreadySplit=True, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        # Calculating the confusion matrix
        # confusionMat = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8])
        # if totConfMatrix is None:
        #     totConfMatrix = confusionMat
        # else:
        #     totConfMatrix = sumMatrices(totConfMatrix, confusionMat)
        # Calculating general metrics
        totAcc = totAcc + acc
        totPrec = totPrec + prec
        totRecall = totRecall + recall
        print("Run " + str(counter) + " with accuracy " + str(acc) +  " with Prec " + str(prec) + " with recall " + str(recall) + "\n")
    # Measuring
    kfoldAcc = totAcc / N
    kfoldPrec = totPrec / N
    kfoldRecall = totRecall / N
    print("\n[N-FOLD] Avg accuracy is " + str(kfoldAcc))
    print("[N-FOLD] Avg precision is " + str(kfoldPrec))
    print("[N-FOLD]Avg recall is " + str(kfoldRecall))
    # # Calculating per class metrics
    # finalClassPrecisions = [0, 0, 0, 0, 0, 0, 0, 0]
    # finalClassRecalls = [0, 0, 0, 0, 0, 0, 0, 0]
    # for cla in range(0, 8):
    #     dividerPrec = N
    #     dividerRec = N
    #     for iteration in range(0,N):
    #         if classPrecisionList[iteration][cla] == -1:
    #             dividerPrec = dividerPrec - 1
    #         else:
    #             finalClassPrecisions[cla] = finalClassPrecisions[cla] + classPrecisionList[iteration][cla]
    #         if classRecallList[iteration][cla] == -1:
    #             dividerRec = dividerRec - 1
    #         else:
    #             finalClassRecalls[cla] = finalClassRecalls[cla] + classRecallList[iteration][cla]
    #     # Calculating average
    #     finalClassPrecisions[cla] = finalClassPrecisions[cla]/dividerPrec
    #     finalClassRecalls[cla] = finalClassRecalls[cla]/dividerRec
    # print("\n\nPer Class Metrics:\n")
    # for cla in range(0,8):
    #     print("CLASS " + str(cla) + " - Precision: " + str(finalClassPrecisions[cla]) + " Recall: " + str(finalClassRecalls[cla]))

    return kfoldAcc, kfoldPrec, kfoldRecall, totConfMatrix


def leaveOneOut(metaInputList, labelInputList, MLfunction):
    # Prepare the range of indexes for Leave One Out Cross Validation
    loo = LeaveOneOut()
    # Iterating through those ranges of indexes
    counter = 0
    totAcc = 0
    totPrec = 0
    totRecall = 0
    for train_index, test_index in loo.split(metaInputList):
        # Increasing iteration counter
        counter = counter + 1
        print("LOO ::: We are in iteration " + str(counter))
        # Actually splitting the training and test data
        X_train, X_test = metaInputList[train_index], metaInputList[test_index]
        y_train, y_test = labelInputList[train_index], labelInputList[test_index]
        # Executing the classification function
        acc, prec, recall = MLfunction(metaInputList, labelInputList, alreadySplit=True, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        totAcc = totAcc + acc
        totPrec = totPrec + prec
        totRecall = totRecall + recall
        print("Run " + str(counter) + " with accuracy " + str(acc) +  " with Prec " + str(prec) + " with recall " + str(recall))
    # Measuring
    kfoldAcc = totAcc / counter
    kfoldPrec = totPrec / counter
    kfoldRecall = totRecall / counter
    print("\n[LOO] Avg accuracy is " + str(kfoldAcc))
    print("[LOO] Avg precision is " + str(kfoldPrec))
    print("[LOO] Avg recall is " + str(kfoldRecall))

    # Final Metrics
    totMat = None
    for mat in confMatrixList:
        if totMat is None:
            totMat = mat
        else:
            totMat = totMat + mat
    calculateMetricsFromConfMat(totMat)


    return kfoldAcc, kfoldPrec, kfoldRecall


# This function simply prints the confusion matrix for a given pair of labels
def printConfusionMatrix(confMat):

    # Generating the confusion matrix
    # confMat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(), labels=[1,2,3,4,5,6,7,8])
    # Preparing the class names
    class_names = ["ToDo", "Ad", "Login", "List", "Portal", "Browser", "Map", "Messages"]

    # Preparing the chart
    df_cm = pd.DataFrame(
        confMat, index=class_names, columns=class_names,
    )
    fig = plt.figure()
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Showing the Confusion Matrix
    plt.show()


# This function executes the usual measures trying to modify the screen splitting point
def measureWithScreenRemodeling(labeledList):

    # Those Arrays contain data for each percentage
    finalAccList = []
    finalPrecList = []
    finalRecallList = []

    # Range to modify new screen splitting
    for percentage in range(0,101):
        # Obtain modified list
        newActivityList = activityDataRemodeler.splitIntoTwoScreenSections(labeledList, percentage)

        # Creating Input Array
        metaInputList = []  # Contains metadata || INPUT 1
        labelIntegerList = []  # Labels as integers (not binary)
        dataCounters = [0] * numLabels
        for activity in newActivityList:
            # Concatenating all metadata for the activity
            tmpArray = []
            tmpArray.append(activity.numClickableTop)
            tmpArray.append(activity.numClickableBot)

            tmpArray.append(activity.numSwipeableTop)
            tmpArray.append(activity.numSwipeableBot)

            tmpArray.append(activity.numEdittextTop)
            tmpArray.append(activity.numEdittextBot)

            tmpArray.append(activity.numLongclickTop)
            tmpArray.append(activity.numLongclickBot)

            tmpArray.append(activity.numPassword)
            tmpArray.append(activity.numCheckable)
            tmpArray.append(activity.presentDrawer)
            tmpArray.append(activity.numTotElements)

            metaInputList.append(tmpArray)

            # Concatenating the label
            labelIntegerList.append(activity.labelNumeric)

            # Incrementing the counter for that specific label
            dataCounters[activity.labelNumeric - 1] = dataCounters[activity.labelNumeric - 1] + 1
        print("Data was successfully parsed!\nLabels found: " + str(dataCounters))

        # Converting to numpy arrays
        metaInputList = np.array(metaInputList)
        labelIntegerList = np.array(labelIntegerList)

        # Executing an ALGORITHM for X times
        bestAcc = 0
        bestPrec = 0
        bestRecall = 0
        ### ------------------------------------------- ###
        for trialNum in range(0,3):

            acc, prec, recall = leaveOneOut(metaInputList, labelIntegerList, randomForest)

            # Finding Max
            if acc > bestAcc:
                bestAcc = acc
            if prec > bestPrec:
                bestPrec = prec
            if recall > bestRecall:
                bestRecall = recall
        ### ------------------------------------------- ###
        # Saving best values to final arrays
        print("Completed " + str(percentage) + "% with Acc: " + str(bestAcc) + " and Prec: " + str(bestPrec) + " and Recall: " + str(bestRecall))
        finalAccList.append(bestAcc)
        finalPrecList.append(bestPrec)
        finalRecallList.append(bestRecall)

    # Plotting final Results
    series = pd.Series(finalAccList)
    rollingMean = series.rolling(15).mean()
    plt.plot(finalAccList)
    plt.plot(rollingMean)
    plt.title('Accuracy according to Screen Splitting using RANDOM FOREST')
    plt.ylabel('Accuracy')
    plt.xlabel('Percentage of screen from the top used as splitting point for features')
    plt.legend(['Accuracy of LOO', 'Rolling Mean over 20 datapoints'], loc='upper right')
    plt.show()

    series = pd.Series(finalPrecList)
    rollingMean = series.rolling(15).mean()
    plt.plot(finalPrecList)
    plt.plot(rollingMean)
    plt.title('Precision according to Screen Splitting using RANDOM FOREST')
    plt.ylabel('Precision')
    plt.xlabel('Percentage of screen from the top used as splitting point for features')
    plt.legend(['Precision of LOO', 'Rolling Mean over 20 datapoints'], loc='upper right')
    plt.show()

    series = pd.Series(finalRecallList)
    rollingMean = series.rolling(15).mean()
    plt.plot(finalRecallList)
    plt.plot(rollingMean)
    plt.title('Recall according to Screen Splitting using RANDOM FOREST')
    plt.ylabel('Recall')
    plt.xlabel('Percentage of screen from the top used as splitting point for features')
    plt.legend(['Recall of LOO', 'Rolling Mean over 20 datapoints'], loc='upper right')
    plt.show()


    print("\n\nFinal Average Accuracy: " + str(np.average(finalAccList)))
    print("Final Average Precision: " + str(np.average(finalPrecList)))
    print("Final Average Recall: " + str(np.average(finalRecallList)))


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    print(scores_mean)
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores")
    ax.set_xlabel(name_param_1)
    ax.set_ylabel('CV Average Accuracy')
    ax.legend(loc="best")
    ax.grid('on')
    plt.show()

def plot_cv_results(cv_results, param_x, param_z, metric='mean_test_score'):
    """
    cv_results - cv_results_ attribute of a GridSearchCV instance (or similar)
    param_x - name of grid search parameter to plot on x axis
    param_z - name of grid search parameter to plot by line color
    """
    cv_results = pd.DataFrame(cv_results)
    col_x = 'param_' + param_x
    col_z = 'param_' + param_z
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.pointplot(x=col_x, y=metric, hue=col_z, data=cv_results, ci=99, n_boot=64, ax=ax)
    ax.set_title("Grid Search Scores")
    ax.set_xlabel(param_x)

    # get the current labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # Beat them into submission and set them back again
    ax.set_xticklabels([str(round(float(label), 7)) for label in labels])

    ax.set_ylabel('CV Average Accuracy')
    ax.legend(loc="best", title=param_z)
    ax.grid('on')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
    return fig


##############################################################

# LOADING ACTIVITY DATASET LIST
labeledList = []
exists = os.path.isfile(storePath)
if exists:
    # If the file  exists, then load it
    with open(storePath, "rb") as fp:  # Unpickling
        labeledList = pickle.load(fp, encoding='latin1')
    print ("[[ Loaded list with " + str(len(labeledList)) + " elements ]]")
else:
    # If the file does not exist yet, create the empty list
    print("Requested file does not exist")
    exit()

# # Trying to remodel the screen proportions for the features
# measureWithScreenRemodeling(labeledList)

# Adding new features to the Activity objects
labeledList = activityDataRemodeler.addNewFeatures(labeledList)

# PREPARING DATA TO BE IN THE CORRECT FORMAT FOR THE NN
# Creating Input Array
metaInputList = []      # Contains metadata || INPUT 1
imageInputList = []     # Contains screenshots || INPUT 2
labelInputList = []     # Contains labels || LABEL
labelIntegerList = []   # Labels as integers (not binary)
dataCounters = [0] * numLabels
for activity in labeledList:
    # Concatenating all metadata for the activity
    tmpArray = []
    tmpArray.append(activity.numClickableTop)
    tmpArray.append(activity.numClickableMid)
    tmpArray.append(activity.numClickableBot)
    # tmpArray.append( int(activity.numClickableTop + activity.numClickableMid + activity.numClickableBot) ) # Adding the sum too

    # tmpArray.append(activity.numSwipeableTop)
    # tmpArray.append(activity.numSwipeableMid)
    # tmpArray.append(activity.numSwipeableBot)
    tmpArray.append(int(activity.numSwipeableTop + activity.numSwipeableMid + activity.numSwipeableBot) ) # Adding the sum too

    tmpArray.append(activity.numEdittextTop)
    tmpArray.append(activity.numEdittextMid)
    tmpArray.append(activity.numEdittextBot)
    # tmpArray.append(int(activity.numEdittextTop + activity.numEdittextMid + activity.numEdittextBot) ) # Adding the sum too

    # tmpArray.append(activity.numLongclickTop)
    # tmpArray.append(activity.numLongclickMid)
    # tmpArray.append(activity.numLongclickBot)
    tmpArray.append(int(activity.numLongclickTop + activity.numLongclickMid + activity.numLongclickBot) ) # Adding the sum too

    tmpArray.append(activity.numFocusableTop)
    tmpArray.append(activity.numFocusableMid)
    tmpArray.append(activity.numFocusableBot)
    # tmpArray.append(int(activity.numFocusableTop + activity.numFocusableMid + activity.numFocusableBot) ) # Adding the sum too

    # tmpArray.append(activity.numEnabledTop)
    # tmpArray.append(activity.numEnabledMid)
    # tmpArray.append(activity.numEnabledBot)
    # tmpArray.append(int(activity.numEnabledTop + activity.numEnabledMid + activity.numEnabledBot) ) # Adding the sum too

    # tmpArray.append(activity.numImageViewsTop)
    # tmpArray.append(activity.numImageViewsMid)
    # tmpArray.append(activity.numImageViewsBot)
    tmpArray.append(int(activity.numImageViewsTop + activity.numImageViewsMid + activity.numImageViewsBot) ) # Adding the sum too

    tmpArray.append(activity.numPassword)
    tmpArray.append(activity.numCheckable)
    tmpArray.append(activity.presentDrawer)
    tmpArray.append(activity.numTotElements)

    metaInputList.append(tmpArray)

    # Concatenating the screenshot
    activity.screenshot = activity.screenshot[...,np.newaxis]   # Adding third dimension to indicate to Keras that we are using Grayscale images
    imageInputList.append(activity.screenshot)

    # Concatenating the label
    label = [0 for i in range(0, numLabels)]        # Transforming the label into the binary '00001000' format starting from decimal
    label[activity.labelNumeric-1] = 1
    labelInputList.append(label)
    labelIntegerList.append(activity.labelNumeric)

    # Incrementing the counter for that specific label
    dataCounters[activity.labelNumeric-1] = dataCounters[activity.labelNumeric-1] + 1

print("Data was successfully parsed!\nLabels found: " + str(dataCounters))


# Preparing the Neural Network model
imageShape = imageInputList[-1].shape
metadataShape = len(metaInputList[-1])
print("Images have a shape of " + str(imageShape) + " while metadata is made of " + str(metadataShape) + " elements")


# Converting to numpy arrays
metaInputList = np.array(metaInputList)
imageInputList = np.array(imageInputList)
labelInputList = np.array(labelInputList)
labelIntegerList = np.array(labelIntegerList)


# Writing arrays to file
filehandler = open('./savedKerasModels/X.dat', 'wb')
pickle.dump(metaInputList, filehandler, protocol=2)
filehandler.close()
filehandler = open('./savedKerasModels/Y.dat', 'wb')
pickle.dump(labelIntegerList, filehandler, protocol=2)
filehandler.close()

# Calling a Training function
# singleTraining(imageShape, metadataShape, numLabels)
# kNearestNeighbors( metaInputList, labelIntegerList)
# randomForest(metaInputList, labelInputList)
# decisionTree(metaInputList, labelInputList)
# supportVectorMachine(metaInputList, labelIntegerList)
# naiveBayes(metaInputList, labelIntegerList)
# logisticRegression(metaInputList, labelIntegerList)

# # Calculating average values over a specified number of runs
# bestAcc = 0
# bestPrec = 0
# bestRecall = 0
# for i in range(0,20):
#
#     maxRuns = 20
#     totAcc = 0
#     totPrec = 0
#     totRecall = 0
#     for run in range(1,maxRuns+1):
#         acc, prec, recall = logisticRegression(metaInputList, labelIntegerList)
#         totAcc = totAcc + acc
#         totPrec = totPrec + prec
#         totRecall = totRecall + recall
#         print("Run " + str(run) + " with accuracy " + str(acc) +  " with Prec " + str(prec) + " with recall " + str(recall))
#
#     avgAcc = totAcc / maxRuns
#     avgPrec = totPrec / maxRuns
#     avgRecall = totRecall / maxRuns
#     print("Avg accuracy is " + str(avgAcc))
#     print("Avg precision is " + str(avgPrec))
#     print("Avg recall is " + str(avgRecall))
#
#     acc = avgAcc
#     prec = avgPrec
#     rec = avgRecall
#     # Finding Max
#     if acc > bestAcc:
#         bestAcc = acc
#     if prec > bestPrec:
#         bestPrec = prec
#     if rec > bestRecall:
#         bestRecall = rec
# print("\n\n---> Best AVG20 values found are ACC: " + str(bestAcc) + " || PREC: " + str(bestPrec) + " || REC: " + str(bestRecall))


# Trying N-FOLD
bestAcc = 0
bestPrec = 0
bestRecall = 0
totConfMatrix = None
for i in range(0,5):
    acc, prec, rec, finalConfusionMatrix = Nfold(11, metaInputList, labelIntegerList, logisticRegression)
    if totConfMatrix is None:
        totConfMatrix = finalConfusionMatrix
    else:
        totConfMatrix = sumMatrices(totConfMatrix, finalConfusionMatrix)
    # Finding Max
    if acc > bestAcc:
        bestAcc = acc
    if prec > bestPrec:
        bestPrec = prec
    if rec > bestRecall:
        bestRecall = rec
print("\n\n---> Best 11-FOLD values found are ACC: " + str(bestAcc) + " || PREC: " + str(bestPrec) + " || REC: " + str(bestRecall))
# # Calculating per-class metrics
# classMetricsCalculator(totConfMatrix)
# printConfusionMatrix(totConfMatrix)

# # Leave One Out
# bestAcc = 0
# bestPrec = 0
# bestRecall = 0
# for i in range(0,1):
#     acc, prec, rec = leaveOneOut(metaInputList, labelIntegerList, logisticRegression)
#     # Finding Max
#     if acc > bestAcc:
#         bestAcc = acc
#     if prec > bestPrec:
#         bestPrec = prec
#     if rec > bestRecall:
#         bestRecall = rec
# print("\n\n---> Best LOO values found are ACC: " + str(bestAcc) + " || PREC: " + str(bestPrec) + " || REC: " + str(bestRecall))


# Managing per-class metrics
