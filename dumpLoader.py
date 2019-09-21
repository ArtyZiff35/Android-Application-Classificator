import os
import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.client import device_lib
from keras import backend as K
from keras import optimizers
import random
from matplotlib import pyplot as plt
import keras_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing



# Constants
DUMP_DIRECTORY_PATH = "./dumpFiles"

# Definition of the SWISH activation function
def swish(x):
    return K.sigmoid(x) * x

# Setting up a Keras model of: 4 Conv and Pool + Flat + 5 Dense
def setupNNmodel(activation=swish, learn_rate=0.00001):
    model = Sequential()
    # Input - Layer
    model.add(Dense(finalArrayLen, activation="relu", input_shape=(finalArrayLen,)))
    # Hidden - Layers
    # model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(1000, activation=activation))
    model.add(Dropout(0.5, noise_shape=None, seed=None))
    model.add(Dense(500, activation="sigmoid"))
    # model.add(Dense(100, activation="sigmoid"))
    # Output- Layer
    model.add(Dense(numClasses, activation="softmax"))
    model.summary()

    model.compile(
        loss='kullback_leibler_divergence',
        optimizer=optimizers.Adam(lr=learn_rate),
        metrics=["accuracy",keras_metrics.precision(), keras_metrics.recall()]
        # metrics=["accuracy"]
    )

    return model


# This method measures the performance of the NN using K-Fold
def crossValidation(X, Y, nFold):

    # Setting up a label encoder, necessary to convert 2d label to 1d
    label_encoder = LabelEncoder()
    Y_1D = []
    for i in range(0, len(Y)):
        Y_1D.append(np.argmax(Y[i][:]))
    # Setting up the K-Fold splitting
    seed = 3
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=nFold, shuffle=True, random_state=seed)
    # Resetting the weights of the model
    initial_weights = NNmodel.get_weights()
    # Actually splitting into training and testing data
    accList = []
    precList = []
    recList = []
    iteration = 0
    for train, test in kfold.split(X, Y_1D):

        print("\n[[[[WE ARE IN ITERATION: " + str(iteration+1) + " ]]]]\n")
        iteration = iteration + 1

        # # Grid Search
        # model = KerasClassifier(build_fn=setupNNmodel, epochs=100, batch_size=10,)
        # # define the grid search parameters
        # activation=['tanh','sigmoid']
        # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        # param_grid = dict(activation=activation, learn_rate=learn_rate)
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        # grid_result = grid.fit(X[train], Y[train])
        # # summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # means = grid_result.cv_results_['mean_test_score']
        # stds = grid_result.cv_results_['std_test_score']
        # params = grid_result.cv_results_['params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("Mean Acc: %f (std: %f) with: %r" % (mean, stdev, param))

        # Training the model
        history = NNmodel.fit(
            X[train], Y[train],
            verbose=1,
            epochs=50,
            batch_size=128,
            shuffle=True,
            validation_split=0.1
        )
        # evaluate the model
        scores = NNmodel.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (NNmodel.metrics_names[1], scores[1] * 100))
        print("%s: %.2f%%" % (NNmodel.metrics_names[2], scores[2] * 100))
        print("%s: %.2f%%" % (NNmodel.metrics_names[3], scores[3] * 100))
        accList.append(scores[1] * 100)
        precList.append(scores[2] * 100)
        recList.append(scores[3] * 100)
        # Plotting Training and Validation curves for Accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # Plotting Training and Validation curves for Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # Setting back the weights
        NNmodel.set_weights(initial_weights)

    # Calculating final results
    mean = np.mean(accList)
    std = np.std(accList)
    print("KFOLD ACCURACY Results: " + str(mean) + "% with std(+/- " + str(std) + "%)")
    mean = np.mean(precList)
    std = np.std(precList)
    print("KFOLD PRECISION Results: " + str(mean) + "% with std(+/- " + str(std) + "%)")
    mean = np.mean(recList)
    std = np.std(recList)
    print("KFOLD RECALL Results: " + str(mean) + "% with std(+/- " + str(std) + "%)")


# Data structures
numApis = 0
numPermissions = 0
numStrings = 0
numClasses = 0              # Number of classes
classCounter = {}           # ClassName : NumOfEntriesForThisClass
classListDictionary = {}    # ClassName : listOfDataForThisClass
classProbability = {}       # ClassName : ClassProbability
classNameToIndex = {}       # ClassName : ClassIndex

totAppCounter = 0           # Total number of apps
validationPercentage = 0.08  # Percentage of data to be used as validation



files = [i for i in os.listdir(DUMP_DIRECTORY_PATH) if i.endswith("dat")]
for file in files:
    # The name of the current category is the same as the file name for now
    category = file
    category = os.path.splitext(category)[0]
    filePath = DUMP_DIRECTORY_PATH + "/" + file
    with open (filePath, 'rb') as fp:
        itemlist = pickle.load(fp)
        print("Loaded " + category + " with " + str(len(itemlist)) + " elements")
        # Update data structures
        classNameToIndex[category] = numClasses
        numClasses = numClasses + 1
        classCounter[category] = len(itemlist)
        totAppCounter = totAppCounter + len(itemlist)
        classListDictionary[category] = itemlist
        numApis = itemlist[-1].getNumOfMethods()
        numPermissions = itemlist[-1].getNumOfPermissions()
        numStrings = len(itemlist[-1].getStringsArray())


# Now that we have everything that we need from our files, we proceed...
# Calculating the class probability
# totEntries = 0
# for category in classCounter:
#     totEntries = totEntries + classCounter[category]
# for category in classCounter:
#     classProbability[category] = classCounter[category]/totEntries
# # Calculating the conditional probabilities
# apiCondProb = np.zeros((numApis,numClasses))
# for classIndex in range(0,numClasses):
#     for apiIndex in range(0, numApis):
#         # Let's count how many times this


# First, build the validation set by extracting some of the training data
# validationCounter = int(validationPercentage * totAppCounter)
# val_X = []
# val_Y = []
# while validationCounter > 0:
#     for category in classListDictionary:
#         categoryList = classListDictionary[category]
#         # Extract one element
#         app = categoryList.pop()
#         classListDictionary[category] = categoryList
#         # Merging the 3 arrays in just one
#         tmpArray = []
#         tmpArray.extend(app.getMehtodsArray())
#         tmpArray.extend(app.getPermissionsArray())
#         tmpArray.extend(app.getStringsArray())
#         # Adding to the validation set
#         val_X.append(tmpArray)
#         # Adding the corresponding numeric label
#         label = [0 for i in range(0, numClasses)]
#         label[classNameToIndex[app.getCategory()]] = 1
#         val_Y.append(label)
#         # Updating counter
#         validationCounter = validationCounter - 1
#         if validationCounter == 0:
#             break
#
# print("Validation size: " + str(len(val_X)))


# Building the complete features training array
X = []
Y = []
tmpArray = []
for category in classListDictionary:
    for app in classListDictionary[category]:
        # Putting apps from all categories together so that we can shuffle them
        tmpArray.append(app)

# Shuffle the array
random.shuffle(tmpArray)

for app in tmpArray:
        # Merging the 3 arrays in just one
        featuresArr = []
        featuresArr.extend(app.getMehtodsArray())
        featuresArr.extend(app.getPermissionsArray())
        featuresArr.extend(app.getStringsArray())
        # Adding to the training set
        X.append(featuresArr)
        # Adding the corresponding numeric label
        label = [0 for i in range(0,numClasses)]            # Transforming the label into the binary '00001000' format starting from decimal
        label[classNameToIndex[app.getCategory()]] = 1
        Y.append(label)

# Set up NN model
finalArrayLen = len(X[-1])
NNmodel = setupNNmodel()

# Converting to numpy arrays
X = np.array(X)
Y = np.array(Y)
# val_X = np.array(val_X)
# val_Y = np.array(val_Y)

# Normalizing Input
std_scale = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X)
X = std_scale.transform(X)
# minMax_scale = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(X)
# X = minMax_scale.transform(X)
# K-FOLD
crossValidation(X, Y, 10)


# # Training the model
# results = NNmodel.fit(
#                         X, Y,
#                         verbose=1,
#                         epochs=30,
#                         batch_size=128,
#                         shuffle=True,
#                   #      validation_data = (val_X, val_Y),
#                         validation_split = validationPercentage
#                         )
#
# # Plot metrics
# #  "Accuracy"
# plt.plot(results.history['acc'])
# plt.plot(results.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training set', 'Validation set'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(results.history['loss'])
# plt.plot(results.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Training set', 'Validation set'], loc='upper left')
# plt.show()
#
# # Printing the validation results
# print("\n\nAverage Validation accuracy: " + str(np.mean(results.history["val_acc"])))
# # Finding best validation accuracy
# print("Best validation accuracy: " + str(max(results.history['val_acc'])))
#
# print("Average validation Precision and Recall: " + str(np.mean(results.history["val_precision"])) + " " + str(np.mean(results.history["val_recall"])))
# print("Best validation Precision and Recall: " + str(max(results.history["val_precision"])) + " " + str(max(results.history["val_recall"])))