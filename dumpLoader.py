import pickle
import os


DUMP_DIRECTORY_PATH = "./dumpFiles"


files = [i for i in os.listdir(DUMP_DIRECTORY_PATH) if i.endswith("dat")]
for file in files:
    filePath = DUMP_DIRECTORY_PATH + "/" + file
    with open (filePath, 'rb') as fp:
        itemlist = pickle.load(fp)
        print("Loaded " + file)