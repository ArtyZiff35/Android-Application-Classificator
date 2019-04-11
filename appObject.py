

class appObject:

    def __init__(self, methodsArray, permissionsArray, stringsArray, category):
        # Binary array containing 1/0 for presence of a specific api method call in the app
        self.binaryMethodsArray = methodsArray
        # Binary array containing 1/0 for presence of specific permission in the manifest of the app
        self.binaryPermissionsArray = permissionsArray
        # Array containing word2vec values for all strings of the app in strings.xml
        self.stringsArray = stringsArray
        # This is the category of this app
        self.category = category

