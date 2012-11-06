
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
#        self.legendEnable = Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName)  # what fieldName??
        self.warnUserCellBorders = True
    
    # Sets up the VTK simulation area 
    def initArea(self):
        # Zoom items
        self.zitems = []
        
        self.cellTypeActors={}
        self.outlineActor = vtk.vtkActor()
        self.outlineDim=[0,0,0]
        
        self.invisibleCellTypes={}
        self.typesInvisibleStr=""
        self.set3DInvisibleTypes()
        
        axesActor = vtk.vtkActor()
        axisTextActor = vtk.vtkFollower()
        
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
                fieldType=(name, self.fieldTypes[name])
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
        self.qvtkWidget.repaint()
        # self.Render()
        # if not self.usedDraw3DFlag and len(self.currentActors.keys()):
        if not self.usedDraw3DFlag and self.graphicsFrameWidget.ren.GetActors().GetNumberOfItems():        
            self.usedDraw3DFlag=True
            # MS resetting camera takes place in drawField fcn in MVCDrawViewBase.py
            # # # self.qvtkWidget.resetCamera()
            
    def set3DInvisibleTypes(self):
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
        self.qvtkWidget.repaint()

    def setDim(self, fieldDim):
        #self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    def hideAllActors(self):
        removedActors=[]
        for actorName in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
            removedActors.append(actorName)
        #cannot remove dictionary elements in  above loop
        for actorName in removedActors:
            del self.currentActors[actorName]

    def showCellTypeActors(self):    
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

    def prepareCellTypeActors(self):
        for actorNumber in self.usedCellTypesList:
            actorName="CellType_"+str(actorNumber)
            # print "Actor name=",actorName
            if not actorNumber in self.cellTypeActors and not actorNumber in self.invisibleCellTypes:
                self.cellTypeActors[actorNumber]=vtk.vtkActor()
                
    def prepareOutlineActor(self,_imageData):
#        print MODULENAME, '------------  prepareOutlineActor()'
        outlineDimTmp=_imageData.GetDimensions()
        # print "\n\n\n this is outlineDimTmp=",outlineDimTmp," self.outlineDim=",self.outlineDim
        if self.outlineDim[0] != outlineDimTmp[0] or self.outlineDim[1] != outlineDimTmp[1] or self.outlineDim[2] != outlineDimTmp[2]:
            self.outlineDim=outlineDimTmp
        
            outline = vtk.vtkOutlineFilter()
            outline.SetInput(_imageData)
            outlineMapper = vtk.vtkPolyDataMapper()
            outlineMapper.SetInputConnection(outline.GetOutputPort())
        
            self.outlineActor.SetMapper(outlineMapper)
            
            color = Configuration.getSetting("WindowColor")   # eventually do this smarter (only get/update when it changes)
            self.outlineActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
#            self.outlineActor.GetProperty().SetColor(1, 1, 1)        
            self.outlineDim = _imageData.GetDimensions()

    def showOutlineActor(self):
#        print MODULENAME, '------------  showOutlineActor()'
        self.currentActors["Outline"]=self.outlineActor
        if self.boundingBox:
            color = Configuration.getSetting("BoundingBoxColor")   # eventually do this smarter (only get/update when it changes)
            self.outlineActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
            self.graphicsFrameWidget.ren.AddActor(self.outlineActor)
    
    def hideOutlineActor(self):
        self.graphicsFrameWidget.ren.RemoveActor(self.outlineActor)
        del self.currentActors["Outline"]
        
    def showAxes(self):
        axes = vtk.vtkAxes()
        axes.SetOrigin(-1, -1, -1)
        axes.SetScaleFactor(20)
        axesMapper = vtk.vtkPolyDataMapper()
        axesMapper.SetInputConnection(axes.GetOutputPort())
        
        axesActor.SetMapper(axesMapper)
        self.ren.AddActor(axesActor)
        
        atext = vtk.vtkVectorText()
        atext.SetText("X-Axis")
        textMapper = vtk.vtkPolyDataMapper()
        textMapper.SetInputConnection(atext.GetOutputPort())
        
        axisTextActor.SetMapper(textMapper)
        #axisTextActor.SetScale(0.2, 0.2, 0.2)
        axisTextActor.SetScale(3, 3, 3)
        #axisTextActor.RotateY(90)
        axisTextActor.AddPosition(0, 0, 0)        
        
        self.graphicsFrameWidget.ren.AddActor(axisTextActor)    
        
    def drawCellField(self, bsd, fieldType):
#        print MODULENAME, '  drawCellField():    calling drawModel.extractCellFieldData()'
        self.usedCellTypesList = self.drawModel.extractCellFieldData()
        numberOfActors = len(self.usedCellTypesList)
        
        self.hideAllActors()
        self.set3DInvisibleTypes()
        
        self.drawModel.prepareOutlineActors((self.outlineActor,))
        self.showOutlineActor()

        dictKey = self.graphicsFrameWidget.winId().__int__()
        
#        print MODULENAME, '  drawCellField():  graphicsWindowVisDict[dictKey]=',self.parentWidget.graphicsWindowVisDict[dictKey]

#        if self.parentWidget.cellsAct.isChecked():
        if (self.parentWidget.graphicsWindowVisDict[dictKey][0])  \
          and not (self.parentWidget.graphicsWindowVisDict[dictKey][1]):    # cells (= cell types)
#            print MODULENAME, '  drawCellField():    drawing Cells for this window'
            self.prepareCellTypeActors()
            self.showCellTypeActors()
            self.drawModel.initCellFieldActors(self.currentActors)
            
        if self.parentWidget.graphicsWindowVisDict[dictKey][1]:    # cell borders (= individual cells)
            if self.warnUserCellBorders:
                reply = QMessageBox.warning(self.parentWidget, "Message",
                                        "Warning:  About to draw individual cells (Vis->Borders is on). If you cancel and 3D:BBox is off, you may need to press 'r' in window to re-center.",
                                        QMessageBox.Ok,QMessageBox.Cancel)
#                print '\n------------ reply,  QMessageBox.Cancel=',reply,QMessageBox.Cancel
                if reply == QMessageBox.Cancel:     # != 1024
                    return
                self.warnUserCellBorders = False
            self.prepareCellTypeActors()
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
        if not self.currentActors.has_key("ConActor"):
            self.currentActors["ConActor"]=self.conActor  
            self.graphicsFrameWidget.ren.AddActor(self.conActor) 
            # print "\n\n\n\n added CON ACTOR"        

    def hideConActors(self):
        if self.currentActors.has_key("ConActor"):
            self.graphicsFrameWidget.ren.RemoveActor(self.conActor) 
            del self.currentActors["ConActor"]  

    def drawConField(self, bsd, fieldType):
#        print MODULENAME,'  -----  drawConField()'
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)

    def drawScalarField(self, bsd, fieldType):
#        print MODULENAME,'  -----  drawScalarField()'
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData3D") # this is simply a "pointer" to function        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)
    
    def drawScalarFieldCellLevel(self, bsd, fieldType):
#        print MODULENAME,'  -----  drawScalarFieldCellLevel()'
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData3D") # this is simply a "pointer" to function         
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)
    
    def drawScalarFieldData(self, bsd, fieldType,_fillScalarField):  # rf. draw*  functions preceding this
        import PlayerPython
        
        self.invisibleCellTypesVector=PlayerPython.vectorint()
        for type in self.invisibleCellTypes: 
            self.invisibleCellTypesVector.append(type)
        
        self.drawModel.initScalarFieldDataActors((self.conActor,), self.invisibleCellTypesVector, _fillScalarField)
        
        self.hideAllActors()
        
        self.set3DInvisibleTypes()
        
        self.drawModel.prepareOutlineActors((self.outlineActor,))
        self.showOutlineActor()
        
        self.showConActors()
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):   # rwh: need to speed this up w/ local flag
#        if self.legendEnable:
            self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
        
        self.Render()    

    def drawVectorField(self, bsd, fieldType):
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldCellLevel(self, bsd, fieldType):        
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldCellLevelData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldData(self,bsd,fieldType,_fillVectorFieldFcn):
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
        self.showOutlineActor()
        
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
#        print MODULENAME, '  configsChanged():  boundingBox=',self.boundingBox
#        self.legendEnable = Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName)  # what fieldName??
        self.parentWidget.requestRedraw()
