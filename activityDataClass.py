import uuid

class activityDataClass:

    def __init__(self, applicationName):
        # Generating unique name for this object
        self.name = uuid.uuid4()
        # Saving the name of the application this metadata is coming from
        self.applicationName = applicationName

        # Initializing all the counters for this object
        self.numClickableTop = 0
        self.numClickableMid = 0
        self.numClickableBot = 0

        self.numSwipeableTop = 0
        self.numSwipeableMid = 0
        self.numSwipeableBot = 0

        self.numEdittextTop = 0
        self.numEdittextMid = 0
        self.numEdittextBot = 0

        self.numLongclickTop = 0
        self.numLongclickMid = 0
        self.numLongclickBot = 0

        self.numPassword = 0
        self.numCheckable = 0
        self.presentDrawer = 0
        self.numTotElements = 0

    def incrementnumClickableTop(self):
        self.numClickableTop = self.numClickableTop + 1

    def incrementnumClickableMid(self):
        self.numClickableMid = self.numClickableMid + 1

    def incrementnumClickableBot(self):
        self.numClickableBot = self.numClickableBot + 1

    def incrementnumSwipeableTop(self):
        self.numSwipeableTop = self.numSwipeableTop + 1

    def incrementnumSwipeableMid(self):
        self.numSwipeableMid = self.numSwipeableMid + 1

    def incrementnumSwipeableBot(self):
        self.numSwipeableBot = self.numSwipeableBot + 1

    def incrementnumEdittextTop(self):
        self.numEdittextTop = self.numEdittextTop + 1

    def incrementnumEdittextMid(self):
        self.numEdittextMid = self.numEdittextMid + 1

    def incrementnumEdittextBot(self):
        self.numEdittextBot = self.numEdittextBot + 1

    def incrementnumLongclickTop(self):
        self.numLongclickTop = self.numLongclickTop + 1

    def incrementnumLongclickMid(self):
        self.numLongclickMid = self.numLongclickMid + 1

    def incrementnumLongclickBot(self):
        self.numLongclickBot = self.numLongclickBot + 1

    def incrementnumPassword(self):
        self.numPassword = self.numPassword + 1

    def incrementnumCheckable(self):
        self.numCheckable = self.numCheckable + 1

    def setPresentDrawer(self):
        self.presentDrawer = 1

    def incrementnumTot(self):
        self.numTotElements = self.numTotElements + 1

    # This following method is used to set the screenshot for the current activity
    def setScreenshot(self, screenshot):
        self.screenshot = screenshot

    # This method is used to store the whole original UI attributes for this activity
    def setAllUIElements(self, elements):
        self.allElements = elements

    # This method sets the label for this activity
    def setLabel(self, label, labelNumeric):
        self.label = label
        self.labelNumeric = labelNumeric
