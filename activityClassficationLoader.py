import pickle
from activityDataClass import activityDataClass
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
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(5, 5),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3
    convLayer = Conv2D(filters=36,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 4
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 4
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
    merged = Dense(100, activation="relu")(merged)
    merged = Dropout(0.1, noise_shape=None, seed=None)(merged)
    merged = Dense(50, activation="relu")(merged)
    out = Dense(outputShape, activation="softmax")(merged)
    mergedModel = Model([metadataModel.input, imageModel.input], out)

    # Hidden Layers
    # mergedModel.add(Dense(200, activation="relu"))
    # mergedModel.add(Dropout(0.1, noise_shape=None, seed=None))
    # mergedModel.add(Dense(50, activation="relu"))
    # # Output- Layer
    # mergedModel.add(Dense(outputShape, activation="softmax"))  # Was using sigmoid, but it is only between 0 and 1


    mergedModel.compile(
        loss='mean_squared_error',
        optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()]
    )

    mergedModel.summary()

    return mergedModel
    

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

# PREPARING DATA TO BE IN THE CORRECT FORMAT FOR THE NN
# Creating Input Array
metaInputList = []      # Contains metadata || INPUT 1
imageInputList = []     # Contains screenshots || INPUT 2
labelInputList = []     # Contains labels || LABEL
for activity in labeledList:
    # Concatenating all metadata for the activity
    tmpArray = []
    tmpArray.append(activity.numClickableTop)
    tmpArray.append(activity.numClickableMid)
    tmpArray.append(activity.numClickableBot)
    tmpArray.append( int(activity.numClickableTop + activity.numClickableMid + activity.numClickableBot) ) # Adding the sum too

    tmpArray.append(activity.numSwipeableTop)
    tmpArray.append(activity.numSwipeableMid)
    tmpArray.append(activity.numSwipeableBot)
    tmpArray.append(int(activity.numSwipeableTop + activity.numSwipeableMid + activity.numSwipeableBot) ) # Adding the sum too

    tmpArray.append(activity.numEdittextTop)
    tmpArray.append(activity.numEdittextMid)
    tmpArray.append(activity.numEdittextBot)
    tmpArray.append(int(activity.numEdittextTop + activity.numEdittextMid + activity.numEdittextBot) ) # Adding the sum too

    tmpArray.append(activity.numLongclickTop)
    tmpArray.append(activity.numLongclickMid)
    tmpArray.append(activity.numLongclickBot)
    tmpArray.append(int(activity.numLongclickTop + activity.numLongclickMid + activity.numLongclickBot) ) # Adding the sum too

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

print("Data was successfully parsed!")

# Preparing the Neural Network model
imageShape = imageInputList[-1].shape
metadataShape = len(metaInputList[-1])
print("Images have a shape of " + str(imageShape) + " while metadata is made of " + str(metadataShape) + " elements")
model = setupFinalModel(imageShape, metadataShape, numLabels)


# Converting to numpy arrays
metaInputList = np.array(metaInputList)
imageInputList = np.array(imageInputList)
labelInputList = np.array(labelInputList)

# Training the model
results = model.fit(
                        x = [metaInputList, imageInputList],
                        y = labelInputList,
                        verbose=1,
                        epochs=1000,
                        batch_size=12,
                        shuffle=True,
                        validation_split = validationPercentage
                        )
