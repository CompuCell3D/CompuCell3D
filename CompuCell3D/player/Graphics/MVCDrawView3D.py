
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Utilities.QVTKRenderWidget import QVTKRenderWidget

from Plugins.ViewManagerPlugins.SimpleTabView import FIELD_TYPES,PLANES

from MVCDrawViewBase import MVCDrawViewBase
import Configuration
import vtk, math
import sys, os


MODULENAME='==== MVCDrawView3D.py:  '

class MVCDrawView3D(MVCDrawViewBase):
    def __init__(self, _drawModel, qvtkWidget, parent=None):
        MVCDrawViewBase.__init__(self,_drawModel,qvtkWidget, parent)
        
        self.initArea()
        self.setParams()
        self.usedCellTypesList=None
        self.usedDraw3DFlag=False
        self.boundingBox = Configuration.getSetting("BoundingBoxOn")
        self.show3DAxes = Configuration.getSetting("ShowAxes")
        self.warnUserCellBorders = True

    def initArea(self):
        '''
        Sets up the VTK simulation area
        :return:None
        '''
        # Zoom items
        self.zitems = []
        
        self.cellTypeActors={}
        self.outlineActor = vtk.vtkActor()
        self.outlineDim=[0,0,0]
        
        self.invisibleCellTypes={}
        self.typesInvisibleStr=""
        self.set3DInvisibleTypes()
        
        self.axesActor = vtk.vtkCubeAxesActor2D()

        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0,1.0)
        self.clut.SetValueRange(1.0,1.0)
        self.clut.SetAlphaRange(1.0,1.0)
        self.clut.SetNumberOfColors(1024)
        self.clut.Build()

        ## Set up the mapper and actor (3D) for concentration field.
        # self.conMapper = vtk.vtkPolyDataMapper()
        self.conActor = vtk.vtkActor()

        self.glyphsActor=vtk.vtkActor()
        # self.glyphsMapper=vtk.vtkPolyDataMapper()
        
        self.cellGlyphsActor  = vtk.vtkActor()
        self.FPPLinksActor  = vtk.vtkActor()

        # Weird attributes
        self.typeActors             = {} # vtkActor
        # self.smootherFilters        = {} # vtkSmoothPolyDataFilter
        # self.polyDataNormals        = {} # vtkPolyDataNormals
        # self.typeExtractors         = {} # vtkDiscreteMarchingCubes
        # self.typeExtractorMappers   = {} # vtkPolyDataMapper
        
    def getPlane(self):
        return ("3D", 0)
    
    def setPlotData(self,_plotData):
        self.currentFieldType=_plotData  
        
    def drawFieldLocal(self, _bsd,_useFieldComboBox=True):
        fieldType=("Cell_Field", FIELD_TYPES[0])
#        print MODULENAME, "FIELD TYPES=",self.fieldTypes
        plane=self.getPlane()
        self.drawModel.setDrawingParameters(_bsd,plane[0],plane[1],fieldType)
        
        self.currentDrawingParameters.bsd=_bsd
        self.currentDrawingParameters.plane=plane[0]
        self.currentDrawingParameters.planePos=plane[1]
        self.currentDrawingParameters.fieldName=fieldType[0]
        self.currentDrawingParameters.fieldType=fieldType[1]        
        self.drawModel.setDrawingParametersObject(self.currentDrawingParameters)
        
        if self.fieldTypes is not None:
            currentPlane=""
            currentPlanePos=""
            if _useFieldComboBox:
                name = str(self.graphicsFrameWidget.fieldComboBox.currentText())
                
                # check that there is a meaningful selection in the fieldComboBox, otherwise it may break:
                if (name != "-- Field Type --"):
                    fieldType=(name, self.fieldTypes[name])
                    (currentPlane, currentPlanePos)=self.getPlane()
                else:
                    # these 4 lines are the same as the "else" 4 lines below:
                    name = self.currentFieldType[0]
                    fieldType=(name, self.fieldTypes[name])
                    self.currentFieldType=fieldType        # this assignment seems redundent but I keep it in order to make sure nothing breaks        
                    (currentPlane, currentPlanePos)=self.getPlane()

            else:
                name = self.currentFieldType[0]
                fieldType=(name, self.fieldTypes[name])
                self.currentFieldType=fieldType        # this assignment seems redundent but I keep it in order to make sure nothing breaks        
                (currentPlane, currentPlanePos)=self.getPlane()
                
            self.setDrawingFunctionName("draw"+fieldType[1]+currentPlane)
#            print MODULENAME, "DrawingFunctionName=","draw"+fieldType[1]+currentPlane
            
            # # self.parentWidget.setFieldType((name, self.parentWidget.fieldTypes[name])) 
            self.currentDrawingParameters.bsd=_bsd
            self.currentDrawingParameters.plane=plane[0]
            self.currentDrawingParameters.planePos=plane[1]
            self.currentDrawingParameters.fieldName=fieldType[0]
            self.currentDrawingParameters.fieldType=fieldType[1]        
            self.drawModel.setDrawingParametersObject(self.currentDrawingParameters)
            
        self.drawField(_bsd,fieldType)
        self.qvtkWidget().repaint()
        # self.Render()
        # if not self.usedDraw3DFlag and len(self.currentActors.keys()):
        if not self.usedDraw3DFlag and self.graphicsFrameWidget.ren.GetActors().GetNumberOfItems():        
            self.usedDraw3DFlag=True
            # MS resetting camera takes place in drawField fcn in MVCDrawViewBase.py
            # # # self.qvtkWidget.resetCamera()
            
    def set3DInvisibleTypes(self):
        '''
        Initializes a dictionary self.invisibleCellTypes of invisible cell types - reads settings "Types3DInvisible"
        :return:None
        '''
        self.colorMap = Configuration.getSetting("TypeColorMap")
        
        typesInvisibleStrTmp = str(Configuration.getSetting("Types3DInvisible"))
        # print "GOT ",typesInvisibleStrTmp
        if typesInvisibleStrTmp != self.typesInvisibleStr:
            self.typesInvisibleStr = str(Configuration.getSetting("Types3DInvisible"))
            
            import string
            typesInvisible = string.replace(self.typesInvisibleStr," ","")
            
            typesInvisible = string.split(typesInvisible,",")
            # print "typesInvisibleVec=",typesInvisibleVec
            #turning list into a dictionary
            self.invisibleCellTypes.clear()
            for type in typesInvisible:
                self.invisibleCellTypes[int(type)]=0        
            # print "\t\t\t self.invisibleCellTypes=",self.invisibleCellTypes
    
    def setCamera(self, fieldDim = None):
        '''
        Initializes default camera view for 3D scene
        :param fieldDim:field dimension (Dim3D C++ object)
        :return:
        '''
        camera = self.graphicsFrameWidget.ren.GetActiveCamera()

        self.setDim(fieldDim)
        # Should I specify these parameters explicitly? 
        # What if I change dimensions in XML file? 
        # The parameters should be set based on the configuration parameters!
        # Should it set depending on projection? (e.g. xy, xz, yz)
        
        distance = self.largestDim(self.dim)*2 # 200 #273.205 #
        
        # FIXME: Hardcoded numbers
        
        camera.SetPosition(self.dim[0]/2, self.dim[1]/2, distance)
        camera.SetFocalPoint(self.dim[0]/2, self.dim[1]/2, 0)
        camera.SetClippingRange(distance - 1, distance + 1)
        # self.GetCurrentRenderer().ResetCameraClippingRange()
        
        self.__initDist = distance #camera.GetDistance()
        self.qvtkWidget().repaint()

    def setDim(self, fieldDim):
        '''
        assigns field dimensions (x,y,z) to a vector self.dim
        :param fieldDim: field dimension - instance of Dim3D (CC3D ++ object)
        :return: None
        '''
        #self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    def hideAllActors(self):
        '''
        Hides all actors
        :return:None
        '''
        removedActors=[]
        for actorName in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
            removedActors.append(actorName)
        #cannot remove dictionary elements in  above loop
        for actorName in removedActors:
            del self.currentActors[actorName]

    def showCellTypeActors(self):
        '''
        Shows Actors representing cell types
        :return:None
        '''
        for actorNumber in self.usedCellTypesList:
            actorName="CellType_"+str(actorNumber)
            # print "Actor name=",actorName
            # print "self.invisibleCellTypes=",self.invisibleCellTypes
            
            if actorNumber in self.invisibleCellTypes:
#                print MODULENAME,"showCellTypeActors: cannot display ",actorName
                pass
            elif not actorName in self.currentActors:
#            if not actorName in self.currentActors:
#            if not actorName in self.currentActors and not actorNumber in self.invisibleCellTypes:
                self.currentActors[actorName]=self.cellTypeActors[actorNumber]
                self.graphicsFrameWidget.ren.AddActor(self.currentActors[actorName])

    def hideCellTypeActors(self):
        '''
        Hides actors representing cell types
        :return:None
        '''
        removedActors=[]
        for actorNumber in self.usedCellTypesList:
            actorName="CellType_"+str(actorNumber)
            # print "Actor name=",actorName
            if actorName in self.currentActors:
                self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
                removedActors.append(actorName)
        #cannot remove dictionary elements in  above loop
        for actorName in removedActors:
            del self.currentActors[actorName]

    def showOutlineActor(self, flag=True):
        '''
        Shoulws bounding box around the lattice
        :param flag: show On/Off flag
        :return:None
        '''
        if flag:
            if not self.currentActors.has_key("Outline"):
                self.currentActors["Outline"]=self.outlineActor
                self.graphicsFrameWidget.ren.AddActor(self.outlineActor)
            else:
                self.graphicsFrameWidget.ren.RemoveActor(self.outlineActor)
                # self.axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
                self.graphicsFrameWidget.ren.AddActor(self.outlineActor)
        else:
            if self.currentActors.has_key("Outline"):
                del self.currentActors["Outline"]
                self.graphicsFrameWidget.ren.RemoveActor(self.outlineActor)

    def drawPlotVisDecorations(self):
        '''
        Draws plot window decorations - outline and axes
        :return:None
        '''

        # if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):
        #     self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))
        #     self.showLegend(True)
        # else:
        #     self.showLegend(False)

        if Configuration.getSetting("BoundingBoxOn"):
            self.drawModel.prepareOutlineActors((self.outlineActor,))
            self.showOutlineActor(True)
        else:
            self.showOutlineActor(False)

        if Configuration.getSetting("ShowPlotAxes", self.currentDrawingParameters.fieldName):
            self.drawModel.prepareAxesActors((None,),(self.axesActor,))
            self.showAxes(True)
        else:
            self.showAxes(False)


    def drawCellVisDecorations(self):
        '''
        Draws cell view window decorations - outline and axes
        :return:None
        '''
        if Configuration.getSetting("BoundingBoxOn"):
            self.drawModel.prepareOutlineActors((self.outlineActor,))
            self.showOutlineActor(True)
        else:
            self.showOutlineActor(False)

        # if Configuration.getSetting("Show3DAxes"):
        if Configuration.getSetting("ShowAxes"):
            self.drawModel.prepareAxesActors((None,), (self.axesActor,))
            self.showAxes(True)
        else:
            self.showAxes(False)


    def showAxes(self, flag=True):
        '''
        Shows 3D axes
        :param flag:show On/Off flag
        :return:None
        '''
        # we do not support axes for vtk less than 5.10.0
        if self.vtk_version_identifier() < self.version_identifier(5,10,0):
            return
            
        if flag:
            if not self.currentActors.has_key("Axes3D"):
                # setting camera for the actor is vrey important to get axes working properly
                self.axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
                self.currentActors["Axes3D"] = self.axesActor
                # print 'self.graphicsFrameWidget.ren.GetActiveCamera()=',self.graphicsFrameWidget.ren.GetActiveCamera()
                self.graphicsFrameWidget.ren.AddActor(self.axesActor)
            else:
                self.graphicsFrameWidget.ren.RemoveActor(self.axesActor)
                self.axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
                self.graphicsFrameWidget.ren.AddActor(self.axesActor)
        else:
            if self.currentActors.has_key("Axes3D"):
                del self.currentActors["Axes3D"]
                self.graphicsFrameWidget.ren.RemoveActor(self.axesActor)



    def drawCellField(self, bsd, fieldType):
        '''
        Draws Cell Field
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
#        print MODULENAME, '  drawCellField():    calling drawModel.extractCellFieldData()'
        self.usedCellTypesList = self.drawModel.extractCellFieldData()
        numberOfActors = len(self.usedCellTypesList)
        
        self.hideAllActors()
        self.set3DInvisibleTypes()
        
        self.drawModel.prepareOutlineActors((self.outlineActor,))
        # remember to edit configs changed for actors to be visible or not. control variable are being changed there

        self.drawCellVisDecorations()

        dictKey = self.graphicsFrameWidget.winId().__int__()
        
        if (self.parentWidget.graphicsWindowVisDict[dictKey][0])  \
          and not (self.parentWidget.graphicsWindowVisDict[dictKey][1]):    # cells (= cell types)
#            print MODULENAME, '  drawCellField():    drawing Cells for this window'
#             self.prepareCellTypeActors()
            self.drawModel.prepareCellTypeActors(self.cellTypeActors, self.invisibleCellTypes)
            self.showCellTypeActors()
            self.drawModel.initCellFieldActors(self.currentActors)
            
        if self.parentWidget.graphicsWindowVisDict[dictKey][1]:    # cell borders (= individual cells)

            self.parentWidget.displayWarning ('3D Cell rendering with Vis->Borders "ON"  may be slow')

            # self.prepareCellTypeActors()
            self.drawModel.prepareCellTypeActors(self.cellTypeActors, self.invisibleCellTypes)
            self.showCellTypeActors()
            self.drawModel.initCellFieldBordersActors(self.currentActors)
        
        # Note: Cell borders and Cluster borders are not meaningful for 3D cells (= [dictKey][1], [2])
        
#        if self.parentWidget.cellGlyphsAct.isChecked():
        if self.parentWidget.graphicsWindowVisDict[dictKey][3]:    # glyphs
            self.drawCellGlyphs3D()
            
#        if self.parentWidget.FPPLinksAct.isChecked():
        if self.parentWidget.graphicsWindowVisDict[dictKey][4]:    # FPP links
            self.drawFPPLinks3D() 
        
        self.Render()

    def showConActors(self):
        '''
        Shows Contour Actors for Concentration field
        :return:None
        '''
        if not self.currentActors.has_key("ConActor"):
            self.currentActors["ConActor"]=self.conActor  
            self.graphicsFrameWidget.ren.AddActor(self.conActor) 
            # print "\n\n\n\n added CON ACTOR"        

    def hideConActors(self):
        '''
        hides Contour Actors for Concentration field
        :return:None
        '''
        if self.currentActors.has_key("ConActor"):
            self.graphicsFrameWidget.ren.RemoveActor(self.conActor) 
            del self.currentActors["ConActor"]  

    def drawConField(self, bsd, fieldType):
        '''
        Draws Concentration Field. Calls universal function drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''

#        print MODULENAME,'  -----  drawConField()'
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)

    def drawScalarField(self, bsd, fieldType):
        '''
        Draws Scalar Field. Calls universal function drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData3D") # this is simply a "pointer" to function        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)
    
    def drawScalarFieldCellLevel(self, bsd, fieldType):
        '''
        Draws Scalar Field -  Cell Level Coloring. Calls universal function drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData3D") # this is simply a "pointer" to function         
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)
    
    def drawScalarFieldData(self, bsd, fieldType,_fillScalarField):  # rf. draw*  functions preceding this
        '''
        Draws Scalar Field - this function actually performs drawing requested by other functions
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        import PlayerPython
        
        self.invisibleCellTypesVector=PlayerPython.vectorint()
        for type in self.invisibleCellTypes: 
            self.invisibleCellTypesVector.append(type)
        
        self.drawModel.initScalarFieldDataActors((self.conActor,), self.invisibleCellTypesVector, _fillScalarField)
        
        self.hideAllActors()
        
        self.set3DInvisibleTypes()
        
        self.drawModel.prepareOutlineActors((self.outlineActor,))

        self.drawPlotVisDecorations()

        self.showConActors()
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):   # rwh: need to speed this up w/ local flag
            self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
        
        self.Render()    

    def drawVectorField(self, bsd, fieldType):
        '''
        Draws Vector Field -  Vectors Are attached to arbitrary point in space. Calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldCellLevel(self, bsd, fieldType):
        '''
        Draws Vector Field -  Vectors Are attached to cell. Calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldCellLevelData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldData(self,bsd,fieldType,_fillVectorFieldFcn):
        '''
        Draws Vector Field - this function actually performs drawing requested by other functions
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initVectorFieldDataActors((self.glyphsActor,),_fillVectorFieldFcn)
                
        if not self.currentActors.has_key("Glyphs2DActor"):
            self.currentActors["Glyphs2DActor"]=self.glyphsActor  
            self.graphicsFrameWidget.ren.AddActor(self.glyphsActor)         
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.glyphsMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)

        self.drawModel.prepareOutlineActors((self.outlineActor,))
        # self.showOutlineActor()
        # self.showAxes()
        self.drawPlotVisDecorations()
        self.Render()

    #-------------------------------------------------------------------------
    def drawCellGlyphs3D(self):
#        print MODULENAME,' drawCellGlyphs3D ============='
        #self.setBorderColor()         
        self.drawModel.initCellGlyphsActor3D(self.cellGlyphsActor, self.invisibleCellTypes.keys() )

        if not self.currentActors.has_key("CellGlyphsActor"):
            self.currentActors["CellGlyphsActor"]=self.cellGlyphsActor
            self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor)
            # print "ADDING cellGlyphs ACTOR"
        else:
            # will ensure that borders is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem()!=self.borderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.cellGlyphsActor)
                self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys()  
    
    def showCellGlyphs(self):
#        print MODULENAME,'  showCellGlyphs'
        Configuration.setSetting("CellGlyphsOn",True)
        if not self.currentActors.has_key("CellGlyphsActor"):
            self.currentActors["CellGlyphsActor"]=self.cellGlyphsActor  
#            print '  showCellGlyphs, add cellGlyphsActor'
            self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        #self.Render()
        #self.graphicsFrameWidget.repaint()        
    
    def hideCellGlyphs(self):
#        print MODULENAME,'  hideCellGlyphs'
        Configuration.setSetting("CellGlyphsOn",False)
        if self.currentActors.has_key("CellGlyphsActor"):
            del self.currentActors["CellGlyphsActor"] 
#            print '  showCellGlyphs, remove cellGlyphsActor'
            self.graphicsFrameWidget.ren.RemoveActor(self.cellGlyphsActor)
        self.Render()
        self.graphicsFrameWidget.repaint()

    #-------------------------------------------------------------------------
    def drawFPPLinks3D(self):
#        print MODULENAME,' drawFPPLinks3D ============='
        #self.setBorderColor()         
        self.drawModel.initFPPLinksActor3D(self.FPPLinksActor, self.invisibleCellTypes.keys() )

        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"]=self.FPPLinksActor
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
            # print "ADDING FPPLinks ACTOR"
        else:
            # will ensure that links actor is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem()!=self.FPPLinksActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
                self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys() 

    def showFPPLinks(self):
#        print '============  MVCDrawView3D.py:  showFPPLinks'
        Configuration.setSetting("FPPLinksOn",True)
        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"]=self.FPPLinksActor  
#            print '============       MVCDrawView3D.py:  showFPPLinks, add FPPLinksActor'
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        #self.Render()
        #self.graphicsFrameWidget.repaint()
    
    def hideFPPLinks(self):
#        print '============  MVCDrawView3D.py:  hideFPPLinks'
        Configuration.setSetting("FPPLinksOn",False)
        if self.currentActors.has_key("FPPLinksActor"):
            del self.currentActors["FPPLinksActor"] 
#            print '============       MVCDrawView3D.py:  showFPPLinks, remove FPPLinksActor'
            self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
        self.Render()
        self.graphicsFrameWidget.repaint()

    def showClusterBorder(self):
        pass
    def hideClusterBorder(self):
        pass
    #-------------------------------------------------------------------------
    def takeSimShot(self, fileName):
        renderLarge = vtk.vtkRenderLargeImage()
    
        renderLarge.SetInput(self.graphicsFrameWidget.ren)         
        
        renderLarge.SetMagnification(1)

        # We write out the image which causes the rendering to occur. If you
        # watch your screen you might see the pieces being rendered right
        # after one another.
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        # # # print "GOT HERE fileName=",fileName
        writer.SetFileName(fileName)
        
        writer.Write()

    def initSizeDim(self, dataSet, x, y, z):
        (xloc, yloc, zloc) = (x, y, z)
        if x == 1:
            xloc += 2
        if y == 1:
            yloc += 2
        if z == 1:
            zloc += 2
            
        dataSet.SetDimensions(xloc, yloc, zloc)

    def configsChanged(self):
        self.populateLookupTable()
        #reassign which types are invisible        
        self.set3DInvisibleTypes()
        self.boundingBox = Configuration.getSetting("BoundingBoxOn")
        # self.show3DAxes = Configuration.getSetting("Show3DAxes")
        self.show3DAxes = Configuration.getSetting("ShowAxes")
#        print MODULENAME, '  configsChanged():  boundingBox=',self.boundingBox
#        self.legendEnable = Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName)  # what fieldName??
        self.parentWidget.requestRedraw()
