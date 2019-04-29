import os
import subprocess
import zipfile
import re
import shutil
import numpy as np
import pickle
from gensim.models import KeyedVectors
from appObject import appObject



# First of all, run script "apiListBuilder.py" in order to build file "apiMethodsList.txt" containing all Android APIs

#MAIN_APK_DIRECTORY_PATH = "./apkFiles"
MAIN_APK_DIRECTORY_PATH = "D:\\apkFiles"       # In external drive
DECOMPRESSED_DIRECTORY_PATH = "./decompressedApks"
BAKSMALI_PATH = "./supportFiles/baksmali-2.2.6.jar"
FULL_PERMISSIONS_LIST_FILE_PATH = "./fullPermissionsList.txt"
FULL_APIS_FILE_PATH = "./apiMethodsList.txt"
WORD2VEC_MODEL_PATH = "./word2vecModels/GoogleNews-vectors-negative300.bin"
DUMP_DIRECTORY_PATH = "./dumpFiles"
MAX_DELETION_BUFFER = 3

#######################################################################

# This method stores in a data structure all the permissions stored in the fullPermissionsList.txt file
# NOTE: from "READ_CALENDAR" going on, they are dangerous permissions
def getFullPermissionsList(fullPermissionsListPath):
    fullPermissionsList = []
    with open(fullPermissionsListPath, "r", encoding="utf8", errors='ignore') as inputFile:
        line = inputFile.readline()
        while(line):
            fullPermissionsList.append(line)
            line = inputFile.readline()
    return fullPermissionsList

def getFullApisList(fullApisListPath):
    fullAPIsDictionary = {}
    with open(fullApisListPath, "r", encoding="utf8", errors='ignore') as inputFile:
        line = inputFile.readline()
        counter = 0
        while(line):
            # Removing arguments and their parenthesis
            # cutIndex = line.find('(')
            # line = line[0:cutIndex]
            line = line.strip()
            fullAPIsDictionary[line] = counter      # Dictionary API String --> Index
            counter = counter + 1
            line = inputFile.readline()
    return fullAPIsDictionary

# This method uses BAKSMALI to parse the classes.dex file
def getAllAppMethods(decompressedAppPath):
    # Path to the classes.dex file for this app
    classesDexPath = decompressedAppPath + "/" + "classes.dex"
    # Write dump to file
    with open('./supportFiles/tempDexDump.txt', "w", encoding="utf8", errors='ignore') as tempDexDump:
        subprocess.call(['java', '-jar', BAKSMALI_PATH, 'dump', classesDexPath], shell=True, stdout=tempDexDump)
    # Read dump from file
    with open('./supportFiles/tempDexDump.txt', "r", encoding="utf8", errors='ignore') as tempDexDump:
        line = tempDexDump.readline()

        # First, find the section of the file concerning method signatures
        while line:
            if "method_id_item section" in line:  ## ---> class_def_item section
                break
            line = tempDexDump.readline()

        # Now, proceed to read struct of each method
        line = tempDexDump.readline()
        readingStatus = ""
        currentString = ""
        className = ""
        methodsList = []
        while line:
            firstThrough = False
            # Termination condition: we've reached annotation section
            if "class_def_item section" in line:
                break
            # First, we find the class name
            if "class_idx" in line:
                # Here we will have a complete name
                currentString = currentString.replace('|', '')
                methodName = re.findall(r':\s[A-Za-z<>]+\[', currentString)
                if len(methodName)>0:
                    methodName[0] = methodName[0].replace(':', '')
                    methodName[0] = methodName[0].replace('[', '')
                    methodName[0] = methodName[0].strip()
                    # Also, we will have a complete method name
                    fullMethodName = str(className) + '.' + methodName[0]
                    methodsList.append(fullMethodName)
                # Now, get back to normal job
                readingStatus = "class"
                currentString = line.strip()
                firstThrough = True
            # Then, the parameters
            if "proto_idx" in line:
                # Here, we will have a complete class
                currentString = currentString.replace('|', '')
                classSplit = currentString.split()
                if len(classSplit) > 0:
                    className = classSplit[-1]
                    className = className.replace(';', '')
                    className = className.replace('/','.')
                    className = className[1:]
                # Now, get back to normal job
                readingStatus = "proto"
            # Then, the name of the method
            if "name_idx" in line:
                readingStatus = "name"
                currentString = line.strip()
                firstThrough = True

            if (readingStatus == "class" or readingStatus == "name") and (firstThrough is False):
                currentString = currentString + line.strip()

            line = tempDexDump.readline()

        # Now we will have in methodsList all methods for this application.
        # Write methods to file
        methodsFile = decompressedAppPath + "/" + "methods.txt"
        with open(methodsFile, "w", encoding="utf8", errors='ignore') as outputFile:
            for meth in methodsList:
                outputFile.write(meth+'\n')


# This method uses APKTOOLS to get the manifest.xml in a readable format
def getAllAppPermissions(decompressedAppPath, apkPath):
    # Path to ApkTool
    apkToolsPath = '.\supportFiles"\"apktool'
    # Destination folder for debynarized files
    destFolder = decompressedAppPath+"/deBin"
    # Debynarize
    FNULL = open(os.devnull, 'w', encoding="utf8", errors='ignore')
    subprocess.call([apkToolsPath, 'd', '-f', '-o', destFolder, apkPath], shell=True, stdout=FNULL)
    # We will now have a readable AndroidManifest.xml
    manifestPath = destFolder+"/AndroidManifest.xml"
    # Try to read every possible permission
    permissionsList = []
    with open(manifestPath, "r", encoding="utf8", errors='ignore') as inputFile:
        line = inputFile.readline()
        while line:
            # We are interested only in permission lines
            if "permission" in line:
                line = line.strip()
                permission = re.findall('"([^"]*)"',line)
                if(len(permission)>0):
                    for perm in permission:
                        if "permission" in perm:
                            perm = perm.strip()
                            permissionsList.append(perm)
            line = inputFile.readline()
    # We now have a list of all the permissions required by the app
    # Write it to permissions.txt
    permissionsFile = decompressedAppPath + "/" + "permissions.txt"
    with open(permissionsFile, "w", encoding="utf8", errors='ignore') as outputFile:
        for permission in permissionsList:
            outputFile.write(permission+'\n')


# This method uses the debynarized files created by the getAllAppPermissions method via APKTOOL to retrieve all strings for this app
def getAllAppStrings(decompressedAppPath):
    # Path to the strings.xml file
    stringsFilePath = decompressedAppPath + "/deBin/res/values/strings.xml"
    # Read strings.xml
    stringsList = []
    with open(stringsFilePath, "r", encoding="utf8", errors='ignore') as inputFile:
        line = inputFile.readline()
        while line:
            # Select only lines containing string values
            if "string name" in line:
                matchResult = re.findall('>.+<',line)
                if len(matchResult) > 0:
                    string = matchResult[0]
                    string = string.replace('<', '')
                    string = string.replace('>', '')
                    string = string.strip()
                    stringsList.append(string)
            line = inputFile.readline()
    # Finally, write the list of strings to strings.txt
    stringsFile = decompressedAppPath + "/" + "strings.txt"
    with open(stringsFile, "w", encoding="utf8", errors='ignore') as outputFile:
        for string in stringsList:
            outputFile.write(string+'\n')



#######################################################################

# Instantiating the word2Vec model
print('Loading word2vec model...')
word2vecModel = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
# Store locally in a data structure the list of all APIs methods
fullAPIsDictionary = getFullApisList(FULL_APIS_FILE_PATH)
# Store locally in a data structure the list of all possible Android Permissions
fullPermissionsList = getFullPermissionsList(FULL_PERMISSIONS_LIST_FILE_PATH)
# Declaring the training array of application data
trainingList = []

# Parse through all APK files by category
categories = [i for i in os.listdir(MAIN_APK_DIRECTORY_PATH)]
deleteCounter = 0
# For each category
for category in categories:
    print("\n\n\nParsing category: " + category)
    # Craft the folder name for this category
    APK_DIRECTORY_PATH = MAIN_APK_DIRECTORY_PATH + "/" + category
    files = [i for i in os.listdir(APK_DIRECTORY_PATH) if i.endswith("apk")]
    # Resetting data that needs to be reset in each category
    fileCounter = 1
    trainingList = []
    # For each file of this category
    for file in files:
        print("Working on (" + category + " --> " + str(fileCounter) + "/" + str(len(files)) + ") " + str(file) + "...")
        fileCounter = fileCounter + 1
        try:
            # First, save some useful strings about this app
            appName = os.path.splitext(file)[0]  # Base name of the app
            relativeApkPath = APK_DIRECTORY_PATH + "/" + file  # Relative path for the Apk
            relativeDecompressedPath = DECOMPRESSED_DIRECTORY_PATH + "/" + appName  # Relative path for the decompressed Apk

            # Decompress the Apk into its specific folder
            with zipfile.ZipFile(relativeApkPath, "r") as zip_ref:
                zip_ref.extractall(relativeDecompressedPath)

            # Building methods.txt for this app (list of all used methods by the app)
            getAllAppMethods(relativeDecompressedPath)

            # Building permissions.txt for this app (list of all permissions required by this app)
            getAllAppPermissions(relativeDecompressedPath, relativeApkPath)

            # Building strings.txt for this app (list of all strings hardcoded in the app)
            getAllAppStrings(relativeDecompressedPath)

            ###############################################################################
            # Now that the app has been fully parsed, we can proceed to build its features vector
            ###############################################################################

            # Building the APIs method binary array
            methodsArray = [0 for i in range(0, len(fullAPIsDictionary))]
            methodsFile = relativeDecompressedPath + "/" + "methods.txt"
            with open(methodsFile, "r", encoding="utf8", errors='ignore') as inputFile:
                method = inputFile.readline()
                while (method):
                    # Convert specific method to just its class (to match API class list)
                    methodSplit = method.split('.')
                    method = ""
                    for i in range(0, len(methodSplit) - 1):
                        method = method + methodSplit[i] + "."
                    method = method[:-1]
                    method = method.strip()
                    # Find if this used method is also in the Android APIs
                    if method in fullAPIsDictionary:
                        methodsArray[fullAPIsDictionary[method]] = 1
                    # for index in range(0, len(fullAPIsDictionary)):
                    #     if method == fullAPIsDictionary[index] :
                    #         methodsArray[index] = 1
                    # Read next method
                    method = inputFile.readline()

            # Building the Permissions binary array
            permissionsArray = [0 for i in range(0, len(fullPermissionsList))]
            permissionsFile = relativeDecompressedPath + "/" + "permissions.txt"
            with open(permissionsFile, "r", encoding="utf8", errors='ignore') as inputFile:
                permission = inputFile.readline()
                while (permission):
                    # Get only the last part of the permission's name
                    permission = permission.split('.')[-1]
                    # Find if this permission is also present in the full permissions list
                    for index in range(0, len(fullPermissionsList)):
                        if permission == fullPermissionsList[index]:
                            permissionsArray[index] = 1
                    # Read next permissions
                    permission = inputFile.readline()

            # Build the word2vec array for the set of strings of this app
            stringsFile = relativeDecompressedPath + "/" + "strings.txt"
            with open(stringsFile, "r", encoding="utf8", errors='ignore') as inputFile:
                line = inputFile.readline()
                lineCounter = 0
                totMat = None
                while (line):
                    # Remove all non alphabetical chars
                    regex = re.compile('[^a-zA-Z]')
                    line = regex.sub('', line)
                    # Add up all the vectors for the same line
                    wordsInLine = line.split()
                    # Filter out all words that are not present in the model
                    wordsInLine = [k for k in wordsInLine if k in word2vecModel.vocab]
                    if (len(wordsInLine) > 0):
                        lineCounter = lineCounter + 1
                        currentMat = None
                        for word in wordsInLine:
                            if currentMat is None:
                                currentMat = word2vecModel[word]
                            else:
                                currentMat = currentMat + word2vecModel[word]
                        # Then average among all the lines
                        if totMat is None:
                            totMat = currentMat
                        else:
                            totMat = totMat + currentMat
                    line = inputFile.readline()
                # Finalize the count of the average for all the lines by diving by the line counter
                if lineCounter > 0:
                    totMat = totMat / lineCounter
                else:
                    totMat = word2vecModel[category]

            # Stats
            print("Methods matches: " + str(np.count_nonzero(methodsArray)))
            print("Permissions matches: " + str(np.count_nonzero(permissionsArray)))

            # Eventually create the appObject with all the due arrays
            appData = appObject(methodsArray, permissionsArray, totMat, category)
            trainingList.append(appData)
            print("OK: Data was successfully appended with size: " + str(len(trainingList)))
            print("\n")

            # Now that we have finished with this app, we might as well proceed to delete the decompressedAPK folder for this APK
            deleteCounter = deleteCounter + 1
            if deleteCounter == MAX_DELETION_BUFFER:
                deleteCounter = 0
                print("[[[Deleting decompressed files...]]]")
                deletionPath = os.path.abspath(DECOMPRESSED_DIRECTORY_PATH)
                subprocess.call(["rd", '/s', '/q', '-o', deletionPath], shell=True)
        except Exception as e:
            print("!!! An error occurred during operations: " + str(e))

    # Now dump the list obtained for the last parsed category
    print("\n-------------------------------\nNow dumping list\n-------------------------------")
    dumpFile = DUMP_DIRECTORY_PATH + "/" + category + ".dat"
    with open(dumpFile, 'wb') as fp:
        pickle.dump(trainingList, fp)










