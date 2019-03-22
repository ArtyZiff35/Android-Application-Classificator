import os
import subprocess
import zipfile
import re


# First of all, run script "apiListBuilder.py" in order to build file "apiMethodsList.txt" containing all Android APIs

APK_DIRECTORY_PATH = "./apkFiles"
DECOMPRESSED_DIRECTORY_PATH = "./decompressedApks"
BAKSMALI_PATH = "./supportFiles/baksmali-2.2.6.jar"

#######################################################################

# This method uses BAKSMALI to parse the classes.dex file
def getAllAppMethods(decompressedAppPath):
    # Path to the classes.dex file for this app
    classesDexPath = decompressedAppPath + "/" + "classes.dex"
    # Write dump to file
    with open('./supportFiles/tempDexDump.txt', "w") as tempDexDump:
        subprocess.call(['java', '-jar', BAKSMALI_PATH, 'dump', classesDexPath], shell=True, stdout=tempDexDump)
    # Read dump from file
    with open('./supportFiles/tempDexDump.txt', "r") as tempDexDump:
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
        with open(methodsFile, "w") as outputFile:
            for meth in methodsList:
                outputFile.write(meth+'\n')


# This method uses APKTOOLS to get the manifest.xml in a readable format
def getAllAppPermissions(decompressedAppPath, apkPath):
    # Path to ApkTool
    apkToolsPath = '.\supportFiles"\"apktool'
    # Destination folder for debynarized files
    destFolder = decompressedAppPath+"/deBin"
    # Debynarize
    FNULL = open(os.devnull, 'w')
    subprocess.call([apkToolsPath, 'd', '-f', '-o', destFolder, apkPath], shell=True, stdout=FNULL)
    # We will now have a readable AndroidManifest.xml
    manifestPath = destFolder+"/AndroidManifest.xml"
    # Try to read every possible permission
    permissionsList = []
    with open(manifestPath, "r") as inputFile:
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
    with open(permissionsFile, "w") as outputFile:
        for permission in permissionsList:
            outputFile.write(permission+'\n')


# This method uses the debynarized files created by the getAllAppPermissions method via APKTOOL to retrieve all strings for this app
def getAllAppStrings(decompressedAppPath):
    # Path to the strings.xml file
    stringsFilePath = decompressedAppPath + "/deBin/res/values/strings.xml"
    # Read strings.xml
    stringsList = []
    with open(stringsFilePath, "r") as inputFile:
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
    with open(stringsFile, "w") as outputFile:
        for string in stringsList:
            outputFile.write(string+'\n')

#######################################################################


# Parse through all APK files
files = [i for i in os.listdir(APK_DIRECTORY_PATH) if i.endswith("apk")]

for file in files:
    print("Working on " + str(file) + "...")
    # First, save some useful strings about this app
    appName = os.path.splitext(file)[0]                                     # Base name of the app
    relativeApkPath = APK_DIRECTORY_PATH + "/" + file                       # Relative path for the Apk
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







