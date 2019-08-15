import os
import subprocess
import re

# This script prints to file "./apiMethodsList.txt" the list of all Android.jar APIs

CLASSES_FILE_PATH = './supportFiles/classesList.txt'
TEMP_METHODS_FILE_PATH = './supportFiles/tempMethodsListForClass.txt'
ANDROID_JAR_PATH = './supportFiles/android.jar'
FINAL_API_METHODS_LIST = './apiMethodsList.txt'

# This is the file where we'll be writing the list of all final methods
apiListOutput = open(FINAL_API_METHODS_LIST,'w')

# List all classes of android.jar and save them in a txt file
with open(CLASSES_FILE_PATH, 'w') as fileOutput:
    subprocess.call(['C:\Program Files\Java\jdk1.8.0_181"\"bin\jar.exe', '-tf', ANDROID_JAR_PATH], shell=True, stdout=fileOutput)

# Now, for each class
with open(CLASSES_FILE_PATH, 'r') as fileInput:
    line = fileInput.readline()
    while line:
        # Consider only Java Classes
        line = line.strip()
        if line.endswith("class") is True:

            # Create correct class name
            className = line.replace("/",".")
            className = className.replace(".class","")

            print('Parsing class ' + className)
            apiListOutput.write(className + '\n')
            # List all methods for this class with Javap
            # with open(TEMP_METHODS_FILE_PATH, 'w') as tmpFile:
            #     subprocess.call(['C:\Program Files\Java\jdk1.8.0_181"\"bin\javap.exe','-classpath', ANDROID_JAR_PATH, className], shell=True, stdout=tmpFile)
            # with open(TEMP_METHODS_FILE_PATH, 'r') as tmpFile:
            #     methodLine = tmpFile.readline()
            #     while methodLine:
            #         # Retrieve only the name of the method, removing return type, visibility, etc...
            #         methodLine = methodLine.strip()
            #         foundMethod = re.findall(r'\s[A-Za-z]+\(.*\)', methodLine)
            #         if len(foundMethod)>0:
            #             # Compose the complete name of the method by appending its name to the class name
            #             foundMethod[0] = foundMethod[0].strip()
            #             finalMethodName = className + '.' + foundMethod[0]
            #             # Write it to the final file
            #             apiListOutput.write(finalMethodName+'\n')
            #         # Read next method name
            #         methodLine = tmpFile.readline()

        # Read next class name
        line = fileInput.readline()

apiListOutput.close()

