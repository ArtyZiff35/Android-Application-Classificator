from activityDataClass import activityDataClass


class activityDataRemodeler:

    # This method transforms a list of activityDataClass objects into a list of activityDataClass objects with customized screen splitting into two sections
    # We use the same activityDataClass as always, but we ignore the "middle section" of the screen
    @staticmethod
    def splitIntoTwoScreenSections(activityObjectList, splittingPercentage):

        outputActivityList = []
        originalScreenHeight = 1920     # Supposing a 1920x1080 resolution

        # Calculating the splitting point in terms of height
        splittingPoint = originalScreenHeight * splittingPercentage

        # Iterating through all saved activities
        for activity in activityObjectList:
            # Instantiation of the copy activityDataClass object and getting the activity name
            activityObject = activityDataClass("dummy")
            # Iterating through all UI elements of this activity
            for elementID in activity.allElements:
                elementAttributes = activity.allElements[elementID]
                # Finding the element's bounds
                topHeight = elementAttributes['bounds'][0][1]
                bottomHeight = elementAttributes['bounds'][1][1]
                # Understanding in which section of the screen the element is located
                if bottomHeight <= splittingPoint:
                    # Case of TOP SECTION of screen
                    if elementAttributes['clickable'] == 'true':
                        activityObject.incrementnumClickableTop()
                    if elementAttributes['scrollable'] == 'true':
                        activityObject.incrementnumSwipeableTop()
                    if elementAttributes['class'] == 'android.widget.EditText':
                        activityObject.incrementnumEdittextTop()
                    if elementAttributes['long-clickable'] == 'true':
                        activityObject.incrementnumLongclickTop()
                else:
                    # Case of BOTTOM SECTION of screen
                    if elementAttributes['clickable'] == 'true':
                        activityObject.incrementnumClickableBot()
                    if elementAttributes['scrollable'] == 'true':
                        activityObject.incrementnumSwipeableBot()
                    if elementAttributes['class'] == 'android.widget.EditText':
                        activityObject.incrementnumEdittextBot()
                    if elementAttributes['long-clickable'] == 'true':
                        activityObject.incrementnumLongclickBot()
                # Doing last checks
                if elementAttributes['password'] == 'true':
                    activityObject.incrementnumPassword()
                if elementAttributes['checkable'] == 'true':
                    activityObject.incrementnumCheckable()
                # Incrementing the total number of UI elements for this activity
                activityObject.incrementnumTot()
            # Set the label
            activityObject.setLabel(activity.label, activity.labelNumeric)
            # Eventually add the new object to the output list
            outputActivityList.append(activityObject)

        return outputActivityList