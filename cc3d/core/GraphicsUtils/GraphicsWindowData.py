class GraphicsWindowData:

    def __init__(self):
        self.sceneName = ''
        self.sceneType = ''
        self.planeName = ''
        self.planePosition = -1
        self.is3D = False
        self.winPosition = None
        self.winSize = None
        self.winType = None
        self.cameraClippingRange = []
        self.cameraFocalPoint = []
        self.cameraPosition = []
        self.cameraViewUp = []

    def toDict(self):
        outDict = {}
        attribs = ['sceneName', 'sceneType', 'planeName', 'planePosition', 'is3D', 'winPosition', 'winSize', 'winType',
                   'cameraClippingRange', 'cameraFocalPoint', 'cameraPosition', 'cameraViewUp']
        for attribName in attribs:
            outDict[attribName] = getattr(self, attribName)

        return outDict

    def fromDict(self, outDict):
        attribs = ['sceneName', 'sceneType', 'planeName', 'planePosition', 'is3D', 'winPosition', 'winSize', 'winType',
                   'cameraClippingRange', 'cameraFocalPoint', 'cameraPosition', 'cameraViewUp']
        for attribName in attribs:
            try:
                setattr(self, attribName, outDict[attribName])
            except KeyError:
                pass

    def __str__(self):

        out = '\n'
        out += '3D='+str(self.is3D)+'\n'
        out += 'sceneName='+str(self.sceneName)+'\n'
        out += 'sceneType='+str(self.sceneType)+'\n'
        out += 'planeName='+str(self.planeName)+'\n'
        out += 'planePosition='+str(self.planePosition)+'\n'
        out += 'winSize='+str(self.winSize)+'\n'
        out += 'winPosition='+str(self.winPosition)+'\n'
        out += 'winType='+str(self.winType)+'\n'
        out += 'cameraClippingRange='+str(self.cameraClippingRange)+'\n'
        out += 'cameraFocalPoint='+str(self.cameraFocalPoint)+'\n'
        out += 'cameraPosition='+str(self.cameraPosition)+'\n'
        out += 'cameraViewUp='+str(self.cameraViewUp)+'\n'

        return out
