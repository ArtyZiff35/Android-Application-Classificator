import pickle
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
from keras.models import Model
import tensorflow as tf
import keras_metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn import tree
from sklearn import svm
import pydotplus
from IPython.display import Image
import graphviz
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import seaborn as sns
import pandas as pd









############### VARIABLES #####################
storePath = "./activityClassificationData/labeledActivities.dat"
numLabels = 8
validationPercentage = 0.09

############### NN ############################

# Setting up a Keras convolutional model of: 4 Conv and Pool + Flat
def setupImageConvModel(inputShape):

    model = Sequential()



    # Convolution Layer 1
    convLayer = Conv2D(filters=12,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='relu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(2, 2),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)


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
        batch_size=12,
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

    ##---------------------------##
    knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors is the value of K
    knn.fit(X_train, y_train)
    # Predict labels for all the testing data
    pred_i = knn.predict(X_test)
    # Calculate the accuracy value for this K
    accuracy = accuracy_score(y_test, pred_i)
    precision = precision_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))
    recall = recall_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))

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
    #

    ##---------------------------##
    forestModel = RandomForestClassifier(n_estimators=21)  # n_estimators is the number of trees
    # Training the model
    forestModel.fit(X_train, y_train)

    # Predict labels for all the testing data
    pred_i = forestModel.predict(X_test)
    # Calculate the accuracy value for this K
    accuracy = accuracy_score(y_test, pred_i)
    precision = precision_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))
    recall = recall_score(y_test, pred_i, average='weighted', labels=np.unique(pred_i))

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
    plt.ylabel('Importance (Entropy)')
    plt.xlabel('Feature ID')
    plt.show()

    return accuracy, precision, recall


# HYP: criterions(gini, etc)
def decisionTree(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)
    # Defining the Decision tree
    treeModel = DecisionTreeClassifier(criterion='gini')
    # Fitting the training data
    treeModel.fit(X_train, y_train)
    # Predict the test data
    y_predict = treeModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average='weighted', labels=np.unique(y_predict))
    recall = recall_score(y_test, y_predict, average='weighted', labels=np.unique(y_predict))
    # print("\nAccuracy with Decision Tree is " + str(accuracy))
    # print("\nPrecision with Decision Tree is " + str(precision))
    # print("\nRecall with Decision Tree is " + str(recall))
    # features = [i for i in range(1,len(X_train[-1])+1)]
    # # Visualizing tree
    # dot_data = tree.export_graphviz(treeModel, out_file=None, class_names=features)
    # graph = graphviz.Source(dot_data)
    # graph.render("./outputFiles/decisionTreeVisualization")

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

    ## LINEAR KERNEL ##
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
    # svmClassifier = svm.SVC(kernel='linear', C=5, gamma='auto')  # Linear Kernel
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
    # for i in range(2, 15):
    #     # Create a new SVM Classifier
    #     svmClassifier = svm.SVC(kernel='poly', degree=i, gamma='auto')  # Polynomial kernel for which we have to specify the degree
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
    # plt.title('SVM w/ Polynomial Kernel degrees')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Polynomial Degree')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()

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
    # plt.xlabel('Polynomial Degree')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()

    ##---------------------------##
    svmClassifier = svm.SVC(kernel='poly', C=30, degree=2, gamma='auto')  # Linear Kernel
    # Train the model using the training set
    svmClassifier.fit(X_train, y_train)
    # Predict the response for test dataset
    y_pred = svmClassifier.predict(X_test)
    # Calculating accuracy by comparing actual test labels and predicted labels
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    return accuracy, precision, recall
    ##---------------------------##


    # GAUSSIAN KERNEL ##
    # accuracies = []
    # precisions = []
    # recalls = []
    # # Trying different values of Gamma parameter
    # for i in range(1, 50):
    #     svmClassifier = svm.SVC(kernel='rbf', C=i, gamma='auto')                 # Gaussian Kernel
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
    # plt.title('SVM w/ Gaussian C values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C value')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()

    # ##---------------------------##
    # svmClassifier = svm.SVC(kernel='rbf', C=4, gamma='auto')  # Linear Kernel
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




# Multinomial Naive Bayes
def naiveBayes(metaInputList, labelInputList, alreadySplit=False, X_train=None, X_test=None, y_train=None, y_test=None):

    # Splitting test and training data
    if alreadySplit==False:
        X_train, X_test, y_train, y_test = train_test_split(metaInputList, labelInputList, test_size=validationPercentage, stratify=labelInputList)

    # Create a Gaussian Classifier
    nbModel = GaussianNB()
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
    logisticRegr = LogisticRegression(solver='newton-cg', multi_class='ovr')
    # Fitting the model
    logisticRegr.fit(X_train, y_train)
    # Predicting the output
    y_pred = logisticRegr.predict(X_test)
    # Calculating metrics
    accuracy = accuracy_score(y_test, y_pred, )
    precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    # print("Logistic Regression accuracy is: " + str(accuracy))
    # print("Logistic Regression precision is: " + str(precision))
    # print("Logistic Regression recall is: " + str(recall))

    # # Try different C values for the Regression
    # accuracies = []
    # precisions = []
    # recalls = []
    # for i in range(2, 25):
    #     # Instantiating the logistic Regression model
    #     logisticRegr = LogisticRegression(solver='newton-cg', multi_class='ovr', C=i)
    #     # Fitting the model
    #     logisticRegr.fit(X_train, y_train)
    #     # Predicting the output
    #     y_pred = logisticRegr.predict(X_test)
    #     # Calculating accuracy by comparing actual test labels and predicted labels
    #     accuracies.append(accuracy_score(y_test, y_pred))
    #     precisions.append(precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #     recalls.append(recall_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))
    #
    # # Plotting for the polynomial
    # plt.plot(accuracies)
    # plt.title('Logistic Regression C values')
    # plt.ylabel('Accuracy')
    # plt.xlabel('C Value')
    # plt.legend(['Accuracy'], loc='upper right')
    # plt.show()


    # Returning the metrics as results
    return accuracy, precision, recall


def Nfold(N, metaInputList, labelInputList, MLfunction):
    # Prepare the range of indexes for Cross Validation
    kfold = KFold(n_splits=N, shuffle=True)      #n_splits is the number of folds
    # Iterating through those ranges of indexes
    counter = 0
    totAcc = 0
    totPrec = 0
    totRecall = 0
    for train_index, test_index in kfold.split(metaInputList):
        # Increasing iteration counter
        counter = counter + 1
        print("KFOLD ::: We are in iteration " + str(counter))
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
    kfoldAcc = totAcc / N
    kfoldPrec = totPrec / N
    kfoldRecall = totRecall / N
    print("\n[N-FOLD] Avg accuracy is " + str(kfoldAcc))
    print("[N-FOLD] Avg precision is " + str(kfoldPrec))
    print("[N-FOLD]Avg recall is " + str(kfoldRecall))
    return kfoldAcc, kfoldPrec, kfoldRecall


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

    return kfoldAcc, kfoldPrec, kfoldRecall


# This function simply prints the confusion matrix for a given pair of labels
def printConfusionMatrix(y_test, y_pred):

    # Generating the confusion matrix
    confMat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=[1,2,3,4,5,6,7,8])
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

# Calling a Training function
# singleTraining(imageShape, metadataShape, numLabels)
# kNearestNeighbors( metaInputList, labelIntegerList)
randomForest(metaInputList, labelInputList)
# decisionTree(metaInputList, labelInputList)
# supportVectorMachine(metaInputList, labelIntegerList)
# naiveBayes(metaInputList, labelIntegerList)
# logisticRegression(metaInputList, labelIntegerList)

# # Calculating average values over a specified number of runs
# bestAcc = 0
# bestPrec = 0
# bestRecall = 0
# for i in range(0,30):
#
#     maxRuns = 20
#     totAcc = 0
#     totPrec = 0
#     totRecall = 0
#     for run in range(1,maxRuns+1):
#         acc, prec, recall = supportVectorMachine(metaInputList, labelIntegerList)
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
# bestAcc = 0
# bestPrec = 0
# bestRecall = 0
# for i in range(0,30):
#     acc, prec, rec = Nfold(11, metaInputList, labelIntegerList, supportVectorMachine)
#     # Finding Max
#     if acc > bestAcc:
#         bestAcc = acc
#     if prec > bestPrec:
#         bestPrec = prec
#     if rec > bestRecall:
#         bestRecall = rec
# print("\n\n---> Best 11-FOLD values found are ACC: " + str(bestAcc) + " || PREC: " + str(bestPrec) + " || REC: " + str(bestRecall))

# Leave One Out
# bestAcc = 0
# bestPrec = 0
# bestRecall = 0
# for i in range(0,10):
#     acc, prec, rec = leaveOneOut(metaInputList, labelIntegerList, randomForest)
#     # Finding Max
#     if acc > bestAcc:
#         bestAcc = acc
#     if prec > bestPrec:
#         bestPrec = prec
#     if rec > bestRecall:
#         bestRecall = rec
# print("\n\n---> Best LOO values found are ACC: " + str(bestAcc) + " || PREC: " + str(bestPrec) + " || REC: " + str(bestRecall))
