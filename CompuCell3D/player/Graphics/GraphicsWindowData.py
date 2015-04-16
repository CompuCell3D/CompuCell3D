class GraphicsWindowData(object):

    def __init__(self):
        self.sceneName = ''
        self.sceneType = ''
        self.planeName = ''
        self.planePosition = -1        
        self.camera = None
        self.is3D = False
        self.winPosition = None
        self.winSize = None
        
    def toDict(self):
        outDict = {}
        attribs = ['sceneName','sceneType','planeName','planePosition','is3D','winPosition','winSize']
        for attribName in attribs:            
            outDict[attribName] = getattr(self,attribName)
            
        return outDict

    def fromDict(self, outDict):
        attribs = ['sceneName','sceneType','planeName','planePosition','is3D','winPosition','winSize']
        for attribName in attribs:            
            setattr(self,attribName,outDict[attribName]) 
                         
        

        
    def __str__(self):    
    
        out ='\n'
        out += '3D='+str(self.is3D)+'\n'
        out += 'sceneName='+str(self.sceneName)+'\n'
        out += 'sceneType='+str(self.sceneType)+'\n'
        out += 'planeName='+str(self.planeName)+'\n'
        out += 'planePosition='+str(self.planePosition)+'\n'    
        out += 'winSize='+str(self.winSize)+'\n'    
        out += 'winPosition='+str(self.winPosition)+'\n'    
        # out += 'camera='+str(self.camera)+'\n'
        
        return out