import vtk

class CAMVCDrawView2D(object):
    def __init__(self,_masterDrawView=None):
        self.initMasterDrawView(_masterDrawView)
        
    def initMasterDrawView(self,_masterDrawView=None):
        if _masterDrawView is None:
            return
            
        from weakref import ref
        
        MDV = ref(_masterDrawView)
        self.mdv=MDV()
                
    def drawCellField(self, _bsd, fieldType):    
        self.mdv.drawCellGlyphs2D()
        
    def drawConField(self, sim, fieldType):    
        print 'Inside drawScalarField sim=',sim
        print 'fieldType=',fieldType
        self.mdv.graphicsFrameWidget.modelSpecificDrawModel2D.initScalarFieldCartesianActors(fieldType,(self.mdv.conActor,))
        

        
        if not self.mdv.currentActors.has_key("ConActor"):
            self.mdv.currentActors["ConActor"]=self.mdv.conActor  
            self.mdv.graphicsFrameWidget.ren.AddActor(self.mdv.conActor) 
            
            actorProperties=vtk.vtkProperty()
            actorProperties.SetOpacity(1.0)
            
            self.mdv.conActor.SetProperty(actorProperties)
            
        # # # if Configuration.getSetting("LegendEnable",self.mdv.currentDrawingParameters.fieldName):            
            # # # self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))            
            # # # self.showLegend(True)
        # # # else:
            # # # self.showLegend(False)

        # # if self.parentWidget.borderAct.isChecked():
            # # self.drawBorders2D() 
        # # else:
            # # self.hideBorder()

# # # #        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):                        
# # # #            self.showContours(True)
# # # #        else:
# # # #            self.showContours(False)
        # # # self.showContours(True)
            
        self.mdv.drawModel.prepareOutlineActors((self.mdv.outlineActor,))       
        self.mdv.showOutlineActor()    
        
        # FIXME: 
        # self.drawContourLines()
        # # # print "DRAW CON FIELD NUMBER OF ACTORS = ",self.ren.GetActors().GetNumberOfItems()
        # self.repaint()
        self.mdv.Render()        