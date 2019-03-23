class CustomActorsStorage:
    def __init__(self, _visName=''):

        self.visName = _visName
        self.actorsDict = {}  # {nameOfActor:vtk actor object}
        self.actorsOrderDict = {}  # {actorNumberOfAppearance :  actorName}

        self.actorsOrderList = []  # [actorName,actorObject]

    def addActor(self, _actorName, _actorObject):
        self.actorsDict[_actorName] = _actorObject
        self.actorsOrderDict[len(list(self.actorsOrderDict.keys()))] = _actorObject

        self.actorsOrderList.append(_actorName)
        self.actorsOrderList.append(_actorObject)

    def getActor(self, _actorName):
        try:
            return self.actorsDict[_actorName]
        except LookupError as e:
            return None

    def getActorsInTheOrderOfAppearance(self):
        return self.actorsOrderList

    def getActorsDict(self):
        return self.actorsDict
