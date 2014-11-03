class CAMVCDrawView3D(object):
    def __init__(self,_masterDrawView=None):
        self.initMasterDrawView(_masterDrawView)
        
    def initMasterDrawView(self,_masterDrawView=None):
        if _masterDrawView is None:
            return
            
        from weakref import ref
        
        MDV = ref(_masterDrawView)
        self.mdv=MDV()
                
    def drawCellField(self, bsd, fieldType):    
        self.mdv.drawCellGlyphs3D()