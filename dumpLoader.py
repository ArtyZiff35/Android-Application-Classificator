import pickle
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.client import device_lib
from keras import backend as K
import random


# Constants
DUMP_DIRECTORY_PATH = "./dumpFiles"

# Setting up a Keras model of: 4 Conv and Pool + Flat + 5 Dense
def setupNNmodel(inputShape, outputShape):
    model = Sequential()
    # Input - Layer
    model.add(Dense(50, activation="relu", input_shape=(inputShape,)))
    # Hidden - Layers
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(50, activation="relu"))
    # Output- Layer
    model.add(Dense(outputShape, activation="sigmoid"))
    model.summary()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model



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
validationPercentage = 0.25  # Percentage of data to be used as validation



files = [i for i in os.listdir(DUMP_DIRECTORY_PATH) if i.endswith("dat")]
for file in files:
    # The name of the current category is the same as the file name for now
    category = file
    category = os.path.splitext(category)[0]
    filePath = DUMP_DIRECTORY_PATH + "/" + file
    with open (filePath, 'rb') as fp:
        itemlist = pickle.load(fp)
        print("Loaded " + category)
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
validationCounter = int(validationPercentage * totAppCounter)
val_X = []
val_Y = []
while validationCounter > 0:
    for category in classListDictionary:
        categoryList = classListDictionary[category]
        # Extract one element
        app = categoryList.pop()
        classListDictionary[category] = categoryList
        # Merging the 3 arrays in just one
        tmpArray = []
        tmpArray.extend(app.getMehtodsArray())
        tmpArray.extend(app.getPermissionsArray())
        tmpArray.extend(app.getStringsArray())
        # Adding to the validation set
        val_X.append(tmpArray)
        # Adding the corresponding numeric label
        val_Y.append(classNameToIndex[category])
        # Updating counter
        validationCounter = validationCounter - 1
        if validationCounter == 0:
            break

print(str(len(val_X)))


# Building the complete features training array
X = []
Y = []
for category in classListDictionary:
    for app in classListDictionary[category]:
        # Merging the 3 arrays in just one
        tmpArray = []
        tmpArray.extend(app.getMehtodsArray())
        tmpArray.extend(app.getPermissionsArray())
        tmpArray.extend(app.getStringsArray())
        # Adding to the training set
        X.append(tmpArray)
        # Adding the corresponding numeric label
        Y.append(classNameToIndex[category])

# Set up NN model
finalArrayLen = len(X[-1])
NNmodel = setupNNmodel(finalArrayLen, numClasses)

# Converting to numpy arrays
X = np.array(X)
Y = np.array(Y)
val_X = np.array(val_X)
val_Y = np.array(val_Y)

# Training the model
results = NNmodel.fit(
                        X, Y,
                        verbose=1,
                        epochs=15,
                        batch_size=32,
                        shuffle=True,
                        validation_data = (val_X, val_Y)
                        )

# Printing the validation results
print("\n\nValidation accuracy:\n" + str(np.mean(results.history["val_acc"])))

