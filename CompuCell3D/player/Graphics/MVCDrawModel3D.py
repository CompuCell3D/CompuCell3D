from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Utilities.QVTKRenderWidget import QVTKRenderWidget

from Plugins.ViewManagerPlugins.SimpleTabView import FIELD_TYPES,PLANES

from MVCDrawModelBase import MVCDrawModelBase
import Configuration
import vtk, math
import sys, os

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()

MODULENAME='------  MVCDrawModel3D.py'


class MVCDrawModel3D(MVCDrawModelBase):
    def __init__(self, qvtkWidget, parent=None):
        MVCDrawModelBase.__init__(self,qvtkWidget, parent)
        
        self.initArea()
        self.setParams()
        
        self.usedCellTypesList=None
        self.usedDraw3DFlag=False
    
    # Sets up the VTK simulation area 
    def initArea(self):
        # Zoom items
        self.zitems = []
        
        # self.cellTypeActors={}
        # self.outlineActor = vtk.vtkActor()
        self.outlineDim=[0,0,0]
        
        # self.invisibleCellTypes={}
        # self.typesInvisibleStr=""
        # self.set3DInvisibleTypes()
        
        # axesActor = vtk.vtkActor()
        # axisTextActor = vtk.vtkFollower()
        
        self.numberOfTableColors = 1024
        self.scalarLUT = vtk.vtkLookupTable()
        self.scalarLUT.SetHueRange(0.67, 0.0)
        self.scalarLUT.SetSaturationRange(1.0,1.0)
        self.scalarLUT.SetValueRange(1.0,1.0)
        self.scalarLUT.SetAlphaRange(1.0,1.0)
        self.scalarLUT.SetNumberOfColors(self.numberOfTableColors)
        self.scalarLUT.Build()
        
        self.lowTableValue = self.scalarLUT.GetTableValue(0)
        self.highTableValue = self.scalarLUT.GetTableValue(self.numberOfTableColors-1)

        ## Set up the mapper and actor (3D) for concentration field.
        self.conMapper = vtk.vtkPolyDataMapper()
        # self.conActor = vtk.vtkActor()

        # self.glyphsActor=vtk.vtkActor()
        self.glyphsMapper = vtk.vtkPolyDataMapper()

        self.cellGlyphsMapper  = vtk.vtkPolyDataMapper()
        self.FPPLinksMapper  = vtk.vtkPolyDataMapper()

        # Weird attributes
        # self.typeActors             = {} # vtkActor
        self.smootherFilters        = {} # vtkSmoothPolyDataFilter
        self.polyDataNormals        = {} # vtkPolyDataNormals
        self.typeExtractors         = {} # vtkDiscreteMarchingCubes
        self.typeExtractorMappers   = {} # vtkPolyDataMapper
        

    def setDim(self, fieldDim):
        # self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    def prepareCellTypeActors(self, _cellTypeActorsDict, _invisibleCellTypes):
        '''
        Scans list of invisible cell types and used cell types and creates those actors that user selected to be visible
        :return:None
        '''
        for actorNumber in self.usedCellTypesList:
            actorName="CellType_"+str(actorNumber)
            if not actorNumber in _cellTypeActorsDict and not actorNumber in _invisibleCellTypes:
                _cellTypeActorsDict[actorNumber] = vtk.vtkActor()

    def prepareOutlineActors(self, _actors):

        outlineData = vtk.vtkImageData()
        
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        
        outlineData.SetDimensions(fieldDim.x+1,fieldDim.y+1,fieldDim.z+1)
        
        # if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.currentDrawingParameters.plane=="XY":       
            # import math            
            # outlineData.SetDimensions(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
            # print "self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
        # else:            
            # outlineData.SetDimensions(self.dim[0]+1, self.dim[1]+1, 1)

        # outlineDimTmp=_imageData.GetDimensions()
        # # print "\n\n\n this is outlineDimTmp=",outlineDimTmp," self.outlineDim=",self.outlineDim
        # if self.outlineDim[0] != outlineDimTmp[0] or self.outlineDim[1] != outlineDimTmp[1] or self.outlineDim[2] != outlineDimTmp[2]:
            # self.outlineDim=outlineDimTmp
        
        outline = vtk.vtkOutlineFilter()
        
        if VTK_MAJOR_VERSION>=6:
            outline.SetInputData(outlineData)
        else:    
            outline.SetInput(outlineData)
        

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
    
        _actors[0].SetMapper(outlineMapper)
        if self.hexFlag:
            _actors[0].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
        _actors[0].GetProperty().SetColor(1, 1, 1)        
        # self.outlineDim=_imageData.GetDimensions()

        color = Configuration.getSetting("BoundingBoxColor")   # eventually do this smarter (only get/update when it changes)
        _actors[0].GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)

    def prepareAxesActors(self, _mappers, _actors):

        axesActor=_actors[0]
        color = Configuration.getSetting("AxesColor")   # eventually do this smarter (only get/update when it changes)
        color = (float(color.red())/255,float(color.green())/255,float(color.blue())/255)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(color)
        tprop.ShadowOn()
        dim = self.currentDrawingParameters.bsd.fieldDim

        axesActor.SetNumberOfLabels(4) # number of labels

        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"]:
            axesActor.SetBounds(0, dim.x, 0, dim.y*math.sqrt(3.0)/2.0, 0, dim.z*math.sqrt(6.0)/3.0)
        else:
            axesActor.SetBounds(0, dim.x, 0, dim.y, 0, dim.z)

        axesActor.SetLabelFormat("%6.4g")
        axesActor.SetFlyModeToOuterEdges()
        axesActor.SetFontFactor(1.5)

        # axesActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
        axesActor.GetProperty().SetColor(color)

        xAxisActor = axesActor.GetXAxisActor2D()
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetRulerDistance(40)
        # xAxisActor.SetRulerMode(20)
        # xAxisActor.RulerModeOn()
        xAxisActor.SetNumberOfMinorTicks(3)

        # setting camera fot he actor is vey important to get axes working properly
#         axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
#         self.graphicsFrameWidget.ren.AddActor(axesActor)

    def extractCellFieldData(self):   # called from MVCDrawView3D.py:drawCellField()
        import CompuCell
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        fieldDim   = self.currentDrawingParameters.bsd.fieldDim
        
        # self.usedCellTypesList=self.fillCellFieldData(cellField)
        
        self.cellType = vtk.vtkIntArray()
        self.cellType.SetName("celltype")
        self.cellTypeIntAddr = self.extractAddressIntFromVtkObject(self.cellType)
        
        # Also get the CellId
        self.cellId = vtk.vtkLongArray()
        self.cellId.SetName("cellid")
        self.cellIdIntAddr = self.extractAddressIntFromVtkObject(self.cellId)
        
#        print '\n\n'
#        print MODULENAME,'   extractCellFieldData():  calling fieldExtractor.fillCellFieldData3D...'
#        self.usedCellTypesList = self.parentWidget.fieldExtractor.fillCellFieldData3D(self.cellTypeIntAddr)
        self.usedCellTypesList = self.parentWidget.fieldExtractor.fillCellFieldData3D(self.cellTypeIntAddr, self.cellIdIntAddr)
#        print MODULENAME,'   extractCellFieldData():  self.cellType.GetSize()=',self.cellType.GetSize()
#        print MODULENAME,'   extractCellFieldData():  self.cellType.GetNumberOfTuples()=',self.cellType.GetNumberOfTuples()
        
#        print MODULENAME," INSIDE DRAW 3D"
#        print "    usedCellTypesList",self.usedCellTypesList
        
        # numberOfActors=len(self.usedCellTypesList)
        return self.usedCellTypesList
    

    def initCellFieldActors(self,_actors):  # original rendering technique (and still used if Vis->Cell Borders not checked) - vkDiscreteMarchingCubes on celltype
        import CompuCell
        fieldDim = self.currentDrawingParameters.bsd.fieldDim    
#        print MODULENAME,'  initCellFieldActors():  fieldDim.x,y,z=',fieldDim.x,fieldDim.y,fieldDim.z
#        numberOfActors = len(self.usedCellTypesList)
        cellTypeImageData = vtk.vtkImageData()
        
        # if hex lattice
#        allowedAreaMin.x=0.0;
#        allowedAreaMin.y=(fieldDim.z>=3? -sqrt(3.0)/6.0 : 0.0);
#        allowedAreaMin.z=0.0;
#
#        allowedAreaMax.x=fieldDim.x+0.5;
#        allowedAreaMax.y=fieldDim.y*sqrt(3.0)/2.0+(fieldDim.z>=3? sqrt(3.0)/6.0 : 0.0);
#        allowedAreaMax.z=fieldDim.z*sqrt(6.0)/3.0;
        
        cellTypeImageData.SetDimensions(fieldDim.x+2,fieldDim.y+2,fieldDim.z+2) # adding 1 pixel border around the lattice to make rendering smooth at lattice borders
        cellTypeImageData.GetPointData().SetScalars(self.cellType)
#        print MODULENAME,'  initCellFieldActors():  self.cellType.GetSize()=',self.cellType.GetSize()
#        print MODULENAME,'  initCellFieldActors(): self.cellType.GetNumberOfTuples()=',self.cellType.GetNumberOfTuples()

        voi = vtk.vtkExtractVOI()
##        voi.SetInputConnection(uGrid.GetOutputPort())
#        voi.SetInput(uGrid.GetOutput())

        if VTK_MAJOR_VERSION>=6:
            voi.SetInputData(cellTypeImageData)
        else:    
            voi.SetInput(cellTypeImageData)


        
#        voi.SetVOI(1,self.dim[0]-1, 1,self.dim[1]-1, 1,self.dim[2]-1 )  # crop out the artificial boundary layer that we created
        voi.SetVOI(0,249, 0,189, 0,170)
        
        numberOfActors = len(self.usedCellTypesList)
        
        # creating and initializing filters, smoothers and mappers - one for each cell type

        filterList = [vtk.vtkDiscreteMarchingCubes() for i in xrange(numberOfActors)]
        smootherList = [vtk.vtkSmoothPolyDataFilter() for i in xrange(numberOfActors)]
        normalsList = [vtk.vtkPolyDataNormals() for i in xrange(numberOfActors)]
        mapperList = [vtk.vtkPolyDataMapper() for i in xrange(numberOfActors)]
        
        # actorCounter=0
        # for i in usedCellTypesList:
        for actorCounter in xrange(len(self.usedCellTypesList)):
            
            if VTK_MAJOR_VERSION>=6:
                filterList[actorCounter].SetInputData(cellTypeImageData)
            else:    
                filterList[actorCounter].SetInput(cellTypeImageData)
            
            
            
#            filterList[actorCounter].SetInputConnection(voi.GetOutputPort())

            # filterList[actorCounter].SetValue(0, usedCellTypesList[actorCounter])
            filterList[actorCounter].SetValue(0, self.usedCellTypesList[actorCounter])
            smootherList[actorCounter].SetInputConnection(filterList[actorCounter].GetOutputPort())
#            print MODULENAME,' smooth iters=',smootherList[actorCounter].GetNumberOfIterations()
#            smootherList[actorCounter].SetNumberOfIterations(200)
            normalsList[actorCounter].SetInputConnection(smootherList[actorCounter].GetOutputPort())
            normalsList[actorCounter].SetFeatureAngle(45.0)
            mapperList[actorCounter].SetInputConnection(normalsList[actorCounter].GetOutputPort())
            mapperList[actorCounter].ScalarVisibilityOff()
            
            actorName = "CellType_" + str(self.usedCellTypesList[actorCounter])
#            print MODULENAME,' initCellFieldActors():  actorName=',actorName
            if actorName in _actors:
                _actors[actorName].SetMapper(mapperList[actorCounter])
                _actors[actorName].GetProperty().SetDiffuseColor(self.celltypeLUT.GetTableValue(self.usedCellTypesList[actorCounter])[0:3])
                if self.hexFlag:
                    _actors[actorName].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
#                _actors[actorName].GetProperty().SetOpacity(0.5)

                
                
    # new rendering technique - vkDiscreteMarchingCubes on cellId
    def initCellFieldBordersActors(self,_actors):   # rf. MVCDrawView3D:drawCellField()
#        print MODULENAME,'  initCellFieldBordersActors():  self.usedCellTypesList=',self.usedCellTypesList
        
        from vtk.util.numpy_support import vtk_to_numpy
        import CompuCell
        fieldDim   = self.currentDrawingParameters.bsd.fieldDim    
        numberOfActors = len(self.usedCellTypesList)
        cellTypeImageData = vtk.vtkImageData()
        cellTypeImageData.SetDimensions(fieldDim.x+2,fieldDim.y+2,fieldDim.z+2) # adding 1 pixel border around the lattice to make rendereing smooth at lattice borders
#        cellTypeImageData.GetPointData().SetScalars(self.cellType)
        cellTypeImageData.GetPointData().SetScalars(self.cellId)
        
        # create a different actor for each cell type
        numberOfActors = len(self.usedCellTypesList)
        
        # creating and initializing filters, smoothers and mappers - one for each cell type

        filterList = [vtk.vtkDiscreteMarchingCubes() for i in xrange(numberOfActors)]
        smootherList = [vtk.vtkSmoothPolyDataFilter() for i in xrange(numberOfActors)]
        normalsList = [vtk.vtkPolyDataNormals() for i in xrange(numberOfActors)]
        mapperList = [vtk.vtkPolyDataMapper() for i in xrange(numberOfActors)]
        
        # actorCounter=0
        # for i in usedCellTypesList:
#        print '-------------- NOTE:  drawing 3D Cell Borders is in beta ----------------'
        for actorCounter in xrange(len(self.usedCellTypesList)):
#            print MODULENAME,' initCellFieldBordersActors(): actorCounter=',actorCounter,

            if VTK_MAJOR_VERSION>=6:
                filterList[actorCounter].SetInputData(cellTypeImageData)
            else:    
                filterList[actorCounter].SetInput(cellTypeImageData)


            
            # filterList[actorCounter].SetValue(0, usedCellTypesList[actorCounter])
            
#            print MODULENAME,' initCellFieldBordersActors():  type(self.cellType)=',type(self.cellType)   # duh, 'vtkObject'
#            print MODULENAME,' initCellFieldBordersActors():  dir(self.cellType)=',dir(self.cellType)
            
            if self.usedCellTypesList[actorCounter] >= 1:
                ctAll = vtk_to_numpy(self.cellType)
#                print ', len(ctAll)=',len(ctAll),
                cidAll = vtk_to_numpy(self.cellId)
#                print ', len(cidAll)=',len(cidAll),
                
                cidUnique = []
                for idx in range(len(ctAll)):
#                    if ctAll[idx] == 1:
                    if ctAll[idx] == self.usedCellTypesList[actorCounter]:
                        cid = cidAll[idx]
                        if cid not in cidUnique:
                            cidUnique.append(cidAll[idx])
                        
#                print ', len(cidUnique)=',len(cidUnique),
#                print ', len(ctAll, cidAll, cidUnique)=',len(ctAll),len(cidAll),len(cidUnique),
                
                for idx in range(len(cidUnique)):
                    filterList[actorCounter].SetValue(idx, cidUnique[idx])
                    
            else:
#                cellTypeImageData.GetPointData().SetScalars(self.cellType)
#                filterList[actorCounter].SetValue(0, self.usedCellTypesList[actorCounter])
                filterList[actorCounter].SetValue(0, 13)   # rwh: what the??
            
#            filterList[actorCounter].SetValue(0, self.usedCellTypesList[actorCounter])
            smootherList[actorCounter].SetInputConnection(filterList[actorCounter].GetOutputPort())
            normalsList[actorCounter].SetInputConnection(smootherList[actorCounter].GetOutputPort())
            normalsList[actorCounter].SetFeatureAngle(45.0)
            mapperList[actorCounter].SetInputConnection(normalsList[actorCounter].GetOutputPort())
            mapperList[actorCounter].ScalarVisibilityOff()
            
            actorName = "CellType_" + str(self.usedCellTypesList[actorCounter])
#            print ', actorName=',actorName
            if actorName in _actors:
                _actors[actorName].SetMapper(mapperList[actorCounter])
                _actors[actorName].GetProperty().SetDiffuseColor(self.celltypeLUT.GetTableValue(self.usedCellTypesList[actorCounter])[0:3])
                if self.hexFlag:
                    _actors[actorName].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
        
        
#    def drawCellField_rwh_old(self, bsd, fieldType):
#        import CompuCell
#        # potts      = sim.getPotts()
#        # cellField  = potts.getCellFieldG()
#        fieldDim   = bsd.fieldDim
#        
#        # self.usedCellTypesList=self.fillCellFieldData(cellField)
#        
#        self.cellType = vtk.vtkIntArray()
#        self.cellType.SetName("celltype")
#        self.cellTypeIntAddr = self.extractAddressIntFromVtkObject(self.cellType)
#        
#        self.cellId = vtk.vtkIntArray()
#        self.cellId.SetName("cellid")
#        self.cellIdIntAddr = self.extractAddressIntFromVtkObject(self.cellId)
#        
##        self.usedCellTypesList = self.parentWidget.fieldExtractor.fillCellFieldData3D(self.cellTypeIntAddr)
#        self.usedCellTypesList = self.parentWidget.fieldExtractor.fillCellFieldData3D(self.cellTypeIntAddr, self.cellIdIntAddr)
#        
#        print MODULENAME,"drawCellField: usedCellTypesList",self.usedCellTypesList
#        
#        numberOfActors = len(self.usedCellTypesList)
#        # self.numberOfUsedCellTypes=len(self.usedCellTypesList)
#        #each cell type will be represented by one actor
#        # print "\n\n\n numberOfActors=",numberOfActors,"\n\n\n"
#                
#        # creating vtkImageData
#        cellTypeImageData = vtk.vtkImageData()
#        cellTypeImageData.SetDimensions(fieldDim.x+2,fieldDim.y+2,fieldDim.z+2) # adding 1 pixel border around the lattice to make rendereing smooth at lattice borders
#        cellTypeImageData.GetPointData().SetScalars(self.cellType)
#        
#        self.hideAllActors()
#        self.set3DInvisibleTypes()
#        self.prepareOutlineActor(cellTypeImageData)
#        self.showOutlineActor()
#        
#        self.prepareCellTypeActors()
#        self.showCellTypeActors()
#        
#        # creating and initializing filters, smoothers and mappers - one for each cell type
#        
#        filterList = [vtk.vtkDiscreteMarchingCubes() for i in xrange(numberOfActors)]
#        smootherList = [vtk.vtkSmoothPolyDataFilter() for i in xrange(numberOfActors)]
#        normalsList = [vtk.vtkPolyDataNormals() for i in xrange(numberOfActors)]
#        mapperList = [vtk.vtkPolyDataMapper() for i in xrange(numberOfActors)]
#        
#        # actorCounter=0
#        # for i in usedCellTypesList:
#        for actorCounter in xrange(len(self.usedCellTypesList)):
#            filterList[actorCounter].SetInput(cellTypeImageData)
#            # filterList[actorCounter].SetValue(0, usedCellTypesList[actorCounter])
#            
#            filterList[actorCounter].SetValue(0, self.usedCellTypesList[actorCounter])
#            smootherList[actorCounter].SetInputConnection(filterList[actorCounter].GetOutputPort())
#            normalsList[actorCounter].SetInputConnection(smootherList[actorCounter].GetOutputPort())
#            normalsList[actorCounter].SetFeatureAngle(45.0)
#            mapperList[actorCounter].SetInputConnection(normalsList[actorCounter].GetOutputPort())
#            mapperList[actorCounter].ScalarVisibilityOff()
#            
#            actorName = "CellType_" + str(self.usedCellTypesList[actorCounter])
#            print MODULENAME, '  drawCellField, actorName=',actorName
#            if actorName in self.currentActors:
#                self.currentActors[actorName].SetMapper(mapperList[actorCounter])
#                rgbVal = self.celltypeLUT.GetTableValue(self.usedCellTypesList[actorCounter])[0:3]
#                self.currentActors[actorName].GetProperty().SetDiffuseColor(rgbVal)
##                self.currentActors[actorName].GetProperty().SetDiffuseColor(self.celltypeLUT.GetTableValue(self.usedCellTypesList[actorCounter])[0:3])
#                # # # print "clut.GetTableValue(actorName)[0:3]=",self.lut.GetTableValue(self.usedCellTypesList[actorCounter])[0:3]
#
#        self.Render()
        
    def showConActors(self):
        if not self.currentActors.has_key("ConActor"):
            self.currentActors["ConActor"] = self.conActor  
            self.graphicsFrameWidget.ren.AddActor(self.conActor) 
            # print "\n\n\n\n added CON ACTOR"        

    def hideConActors(self):
        if self.currentActors.has_key("ConActor"):
            self.graphicsFrameWidget.ren.RemoveActor(self.conActor) 
            del self.currentActors["ConActor"]  
    
    def drawConField(self, bsd, fieldType):        
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)

    def drawScalarField(self, bsd, fieldType):
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData3D") # this is simply a "pointer" to function        
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)
    
    def drawScalarFieldCellLevel(self, bsd, fieldType):    
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData3D") # this is simply a "pointer" to function         
        self.drawScalarFieldData(bsd,fieldType,fillScalarField)

    def initScalarFieldDataActors(self, _actors, _invisibleCellTypesVector, _fillScalarField):  #  rf. MVCDrawView3D:drawScalarFieldData
        import CompuCell
        import PlayerPython
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print MODULENAME, '  initScalarFieldDataActors(): fieldDim=',fieldDim

        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        conFieldName = self.currentDrawingParameters.fieldName
        
        numIsos = Configuration.getSetting("NumberOfContourLines",conFieldName)
        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)
        if type(self.isovalStr) == QVariant:
#          isovalStr = isovalStr.toString()
#          print MODULENAME, ' self.isovalStr.toList()=',self.isovalStr.toList()
#          print MODULENAME, ' self.isovalStr.toString()=',self.isovalStr.toString()
          self.isovalStr = str(self.isovalStr.toString())
#          print MODULENAME, ' new type(self.isovalStr)=',type(self.isovalStr)
#        elif type(self.isovalStr) == QString:
        else:
          self.isovalStr = str(self.isovalStr)
          
#        print MODULENAME, '  initScalarFieldDataActors(): len(self.isovalStr), numIsos=',len(self.isovalStr),numIsos
#        if (len(self.isovalStr) == 0) and (numIsos == 0):
#            print MODULENAME, '  initScalarFieldDataActors(): nothing to do, returning'
##            self.clearDisplay()    
#            self.graphicsFrameWidget.ren.RemoveActor(_actors[0])
#            self.Render()    
#            return
        
        #print self._statusBar.currentMessage() 
        self.dim    = [fieldDim.x, fieldDim.y, fieldDim.z]
#        print MODULENAME, '  initScalarFieldDataActors(): self.dim=',self.dim
        
#        field       = vtk.vtkImageDataGeometryFilter()
#        contour     = vtk.vtkContourFilter()
        
        # self.fillCellFieldData(cellField)
        
        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr = self.extractAddressIntFromVtkObject(self.conArray)

        self.cellTypeCon = vtk.vtkIntArray()
        self.cellTypeCon.SetName("concelltype")
        self.cellTypeConIntAddr = self.extractAddressIntFromVtkObject(self.cellTypeCon)
        
        # self.invisibleCellTypesVector=PlayerPython.vectorint()
        # for type in self.invisibleCellTypes: 
            # self.invisibleCellTypesVector.append(type)
        # # print "self.invisibleCellTypesVector=",self.invisibleCellTypesVector
        # # print "self.invisibleCellTypesVector=",self.invisibleCellTypesVector.size()
        
#        print MODULENAME, '  initScalarFieldDataActors(): calling ',_fillScalarField
        fillSuccessful = _fillScalarField(self.conArrayIntAddr, self.cellTypeConIntAddr, conFieldName, _invisibleCellTypesVector)
        
        if not fillSuccessful:
            return
        
#        print MODULENAME, '  initScalarFieldDataActors(): conArray.GetNumberOfTuples()=',self.conArray.GetNumberOfTuples()
#        print MODULENAME, '  initScalarFieldDataActors(): conArray.GetSize()=',self.conArray.GetSize()
            
        range = self.conArray.GetRange()
        self.minCon = range[0]
        self.maxCon = range[1]
        fieldMax = range[1]
#        print MODULENAME, '  initScalarFieldDataActors(): min,maxCon=',self.minCon,self.maxCon
        
        if Configuration.getSetting("MinRangeFixed",conFieldName):
            self.minCon = Configuration.getSetting("MinRange",conFieldName)            
#            self.clut.SetTableValue(0,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(0,self.lowTableValue)
                        
        if Configuration.getSetting("MaxRangeFixed",conFieldName):
            self.maxCon = Configuration.getSetting("MaxRange",conFieldName)
#            self.clut.SetTableValue(self.numberOfTableColors-1,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(self.numberOfTableColors-1,self.highTableValue)
        
        # print "getting conField=",conField
        # (self.minCon, self.maxCon) = self.fillConFieldData( cellField, conField)
        # print "(self.minCon, self.maxCon) ",(self.minCon, self.maxCon)
        
#        conc_vol = vtk.vtkImageData()
#        conc_vol.GetPointData().SetScalars(self.conArray)
#        print MODULENAME, '  initScalarFieldDataActors(): setting conc_vol dims=',self.dim[0]+2,self.dim[1]+2,self.dim[2]+2
#        conc_vol.SetDimensions(self.dim[0]+2,self.dim[1]+2,self.dim[2]+2)
        
        uGrid = vtk.vtkStructuredPoints()
        uGrid.SetDimensions(self.dim[0]+2, self.dim[1]+2, self.dim[2]+2)  #  only add 2 if we're filling in an extra boundary (rf. FieldExtractor.cpp)
#        uGrid.SetDimensions(self.dim[0],self.dim[1],self.dim[2])  
#        uGrid.GetPointData().SetScalars(self.cellTypeCon)   # cellType scalar field
        uGrid.GetPointData().SetScalars(self.conArray)
#        uGrid.GetPointData().AddArray(self.conArray)        # additional scalar field

#        print 'dir(uGrid)=',dir(uGrid)
#        self.conArray.GetName("concentration")

        voi = vtk.vtkExtractVOI()
##        voi.SetInputConnection(uGrid.GetOutputPort())
#        voi.SetInput(uGrid.GetOutput())

        if VTK_MAJOR_VERSION>=6:
            voi.SetInputData(uGrid)
        else:    
            voi.SetInput(uGrid)


        
        voi.SetVOI(1,self.dim[0]-1, 1,self.dim[1]-1, 1,self.dim[2]-1 )  # crop out the artificial boundary layer that we created
        
        isoContour = vtk.vtkContourFilter()
        #skinExtractorColor = vtk.vtkDiscreteMarchingCubes()
        #skinExtractorColor = vtk.vtkMarchingCubes()
#        isoContour.SetInput(uGrid)
        isoContour.SetInputConnection(voi.GetOutputPort())
        
        isoVals = self.getIsoValues(conFieldName)
#        print MODULENAME, 'initScalarFieldDataActors():  getIsoValues=',isoVals

#        print MODULENAME, ' initScalarFieldDataActors():   getting ScalarIsoValues for field conFieldName=: ',conFieldName
#        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)
##        print MODULENAME, '  type(self.isovalStr)=',type(self.isovalStr)
##        print MODULENAME, '  self.isovalStr=',self.isovalStr
#        if type(self.isovalStr) == QVariant:
##          isovalStr = isovalStr.toString()
##          print MODULENAME, ' self.isovalStr.toList()=',self.isovalStr.toList()
##          print MODULENAME, ' self.isovalStr.toString()=',self.isovalStr.toString()
#          self.isovalStr = str(self.isovalStr.toString())
##          print MODULENAME, ' new type(self.isovalStr)=',type(self.isovalStr)
##        elif type(self.isovalStr) == QString:
#        else:
#          self.isovalStr = str(self.isovalStr)


#        print MODULENAME, '  pre-replace,split; initScalarFieldDataActors(): self.isovalStr=',self.isovalStr
#        import string
#        self.isovalStr = string.replace(self.isovalStr,","," ")
#        self.isovalStr = string.split(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final type(self.isovalStr)=',type(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final self.isovalStr=',self.isovalStr

#        print MODULENAME, '  initScalarFieldDataActors(): len(self.isovalStr)=',len(self.isovalStr)
        printIsoValues = True
#        if printIsoValues:  print MODULENAME, ' isovalues= ',
        isoNum = 0
        for idx in xrange(len(self.isovalStr)):
#            print MODULENAME, '  initScalarFieldDataActors(): idx= ',idx
            try:
                isoVal = float(self.isovalStr[idx])
                if printIsoValues:  print MODULENAME, '  initScalarFieldDataActors(): setting (specific) isoval= ',isoVal
                isoContour.SetValue(isoNum, isoVal)
                isoNum += 1
            except:
                print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ',self.isovalStr[idx]
        if isoNum > 0:  isoNum += 1
#        print MODULENAME, '  after specific isovalues, isoNum=',isoNum
#        numIsos = Configuration.getSetting("NumberOfContourLines")
#        print MODULENAME, '  Next, do range of isovalues: min,max, # isos=',self.minCon,self.maxCon,numIsos
        delIso = (self.maxCon - self.minCon)/(numIsos+1)  # exclude the min,max for isovalues
#        print MODULENAME, '  initScalarFieldDataActors(): delIso= ',delIso
        isoVal = self.minCon + delIso
        for idx in xrange(numIsos):
            if printIsoValues:  print MODULENAME, '  initScalarFieldDataActors(): isoNum, isoval= ',isoNum,isoVal
            isoContour.SetValue(isoNum, isoVal)
            isoNum += 1
            isoVal += delIso
        if printIsoValues:  print 
        
        # UGLY hack to NOT display anything since our attempt to RemoveActor (below) don't seem to work
        if isoNum == 0:
            isoVal = fieldMax + 1.0    # go just outside valid range
            isoContour.SetValue(isoNum, isoVal)
        
#        concLut = vtk.vtkLookupTable()
        # concLut.SetTableRange(conc_vol.GetScalarRange())
#        concLut.SetTableRange([self.minCon,self.maxCon])
        self.scalarLUT.SetTableRange([self.minCon,self.maxCon])
#        concLut.SetNumberOfColors(256)
#        concLut.Build()
        # concLut.SetTableValue(39,0,0,0,0)        
        
#        skinColorMapper = vtk.vtkPolyDataMapper()
        #skinColorMapper.SetInputConnection(skinNormals.GetOutputPort())
#        self.conMapper.SetInputConnection(skinExtractorColor.GetOutputPort())
        self.conMapper.SetInputConnection(isoContour.GetOutputPort())
        self.conMapper.ScalarVisibilityOn()
        self.conMapper.SetLookupTable(self.scalarLUT)
        # # # print " this is conc_vol.GetScalarRange()=",conc_vol.GetScalarRange()
        # self.conMapper.SetScalarRange(conc_vol.GetScalarRange())
        self.conMapper.SetScalarRange([self.minCon,self.maxCon])
        #self.conMapper.SetScalarRange(0,1500)
        
        # rwh - what does this do?
#        self.conMapper.SetScalarModeToUsePointFieldData()
#        self.conMapper.ColorByArrayComponent("concentration",0)
        
#        print MODULENAME,"initScalarFieldDataActors():  Plotting 3D Scalar field"
        # self.conMapper      = vtk.vtkPolyDataMapper()
        # self.conActor       = vtk.vtkActor()        

        _actors[0].SetMapper(self.conMapper)
        if self.hexFlag:
            _actors[0].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
        self.Render()    


    def drawVectorField(self, bsd, fieldType):
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldCellLevel(self, bsd, fieldType):        
#        print MODULENAME,"INSIDE drawVectorFieldCellLevel"
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldCellLevelData3D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData3D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)

    def initVectorFieldDataActors(self,_actors,_fillVectorFieldFcn):
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        fieldDim  = self.currentDrawingParameters.bsd.fieldDim
        
        conFieldName = self.currentDrawingParameters.fieldName
        
        #print self._statusBar.currentMessage() 
        self.dim    = [fieldDim.x, fieldDim.y, fieldDim.z]

        vectorGrid = vtk.vtkUnstructuredGrid()

        points = vtk.vtkPoints()
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        pointsIntAddr = self.extractAddressIntFromVtkObject(points)
        vectorsIntAddr = self.extractAddressIntFromVtkObject(vectors)
        
        fillSuccessful = _fillVectorFieldFcn(pointsIntAddr,vectorsIntAddr,conFieldName)
        if not fillSuccessful:
            return
        
        vectorGrid.SetPoints(points)
        vectorGrid.GetPointData().SetVectors(vectors)

        cone = vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        #cone.SetRadius(4)

        range = vectors.GetRange(-1)
        
        self.minMagnitude = range[0]
        self.maxMagnitude = range[1]
        
        if Configuration.getSetting("MinRangeFixed",conFieldName):
            self.minMagnitude = Configuration.getSetting("MinRange",conFieldName)
            
        if Configuration.getSetting("MaxRangeFixed",conFieldName):
            self.maxMagnitude = Configuration.getSetting("MaxRange",conFieldName)
            
        glyphs = vtk.vtkGlyph3D()
        
        if VTK_MAJOR_VERSION>=6:
            glyphs.SetInputData(vectorGrid)
        else:    
            glyphs.SetInput(vectorGrid)

        
        glyphs.SetSourceConnection(cone.GetOutputPort())
        #glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # glyphs.SetScaleFactor(Configuration.getSetting("ArrowLength")) # scaling arrows here ArrowLength indicates scaling factor not actual length
        
        arrowScalingFactor = Configuration.getSetting("ArrowLength",conFieldName) # scaling factor for an arrow - ArrowLength indicates scaling factor not actual length
        
        if Configuration.getSetting("ScaleArrowsOn",conFieldName):
            glyphs.SetScaleModeToScaleByVector()
            rangeSpan = self.maxMagnitude-self.minMagnitude
            dataScalingFactor = max(abs(self.minMagnitude),abs(self.maxMagnitude))
#            print MODULENAME,"self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude
            
            if dataScalingFactor==0.0:
                dataScalingFactor = 1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)
            #coloring arrows


            color = Configuration.getSetting("ArrowColor",conFieldName)
            r,g,b = color.red(), color.green(), color.blue()
#            print MODULENAME,"   initVectorFieldDataActors():  arrowColor=",arrowColor
#            r = arrowColor.red()
#            g = arrowColor.green()
#            b = arrowColor.blue()
#            _actors[0].GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))        
            _actors[0].GetProperty().SetColor(r, g, b)        
        else:
            glyphs.SetColorModeToColorByVector()
            glyphs.SetScaleFactor(arrowScalingFactor)         
        
        
        self.glyphsMapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphsMapper.SetLookupTable(self.scalarLUT)

        # # # print "vectors.GetNumberOfTuples()=",vectors.GetNumberOfTuples()    
        # self.glyphsMapper.SetScalarRange(vectors.GetRange(-1)) # this will return the range of magnitudes of all the vectors store int vtkFloatArray
        # self.glyphsMapper.SetScalarRange(range)

        self.glyphsMapper.SetScalarRange([self.minMagnitude,self.maxMagnitude])
        
        _actors[0].SetMapper(self.glyphsMapper)
        if self.hexFlag:
            _actors[0].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
        
        
    def __zoomStep(self, delta):
        # # # print "ZOOM STEP"
        if self.ren:
            # renderer = self.GetCurrentRenderer()
            camera = self.ren.GetActiveCamera()
            
            zoomFactor = math.pow(1.02,(0.5*(delta/8)))

            # I don't know why I might need the parallel projection
            if camera.GetParallelProjection(): 
                parallelScale = camera.GetParallelScale()/zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                self.ren.ResetCameraClippingRange()

            self.Render()
            
    # def zoomIn(self):
    #     delta = 2*120
    #     self.__zoomStep(delta)
    #
    # def zoomOut(self):
    #     delta = -2*120
    #     self.__zoomStep(delta)
    #
    # def zoomFixed(self, val):
    #     if self.ren:
    #         # renderer = self._CurrentRenderer
    #         camera = self.ren.GetActiveCamera()
    #         self.__curDist = camera.GetDistance()
    #
    #         # To zoom fixed, dolly should be set to initial position
    #         # and then moved to a new specified position!
    #         if (self.__initDist != 0):
    #             # You might need to rewrite the fixed zoom in case if there
    #             # will be flickering
    #             camera.Dolly(self.__curDist/self.__initDist)
    #
    #         camera.Dolly(self.zitems[val])
    #         self.ren.ResetCameraClippingRange()
    #
    #         self.Render()

    def takeSimShot(self, fileName):
        renderLarge = vtk.vtkRenderLargeImage()
        if VTK_MAJOR_VERSION>=6:
            renderLarge.SetInputData(self.graphicsFrameWidget.ren)
        else:    
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

    # this function is used during prototyping. in production code it is replaced by C++ counterpart    
    def fillCellFieldData_old(self,_cellFieldG):
        import CompuCell
        
        pt = CompuCell.Point3D() 
        cell = CompuCell.CellG() 
        fieldDim = _cellFieldG.getDim()

        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]
        # # # print "FILLCELLFIELDDATA 3D"
        # # # print "self.dim=",self.dim
        offset=0

        #will add 1 pixel border to celltype vtkImage data so that rendering will look smooth at the borders
        self.cellType = vtk.vtkIntArray()
        self.cellType.SetName("celltype")
        self.cellType.SetNumberOfValues((self.dim[2]+2)*(self.dim[1]+2)*(self.dim[0]+2))
        self.cellId=[[[0 for k in range(self.dim[2])] for j in range(self.dim[1])] for i in range(self.dim[0])]
        
        usedCellTypes={}
        
        # For some reasons the points x=0 are eaten up (don't know why).
        # So we just populate empty cellIds.
        # for i in range(self.dim[0]+1):
            # self.cellType.SetValue(offset, 0)
            # offset += 1
                
        for k in range(self.dim[2]+2):
            for j in range(self.dim[1]+2):
                for i in range(self.dim[0]+2):                
                    if i==0 or i ==self.dim[0]+1 or j==0 or j ==self.dim[1]+1 or k==0 or k ==self.dim[2]+1:
                        self.cellType.InsertValue(offset, 0)
                        offset+=1
                    else:
                        pt.x = i-1
                        pt.y = j-1
                        pt.z = k-1
                        cell = _cellFieldG.get(pt)
                        if cell is not None:
                            type    = int(cell.type)
                            id      = int(cell.id)
                            if not type in usedCellTypes:
                                usedCellTypes[type]=0
                        else:
                            type    = 0
                            id      = 0
                        self.cellType.InsertValue(offset, type)
                        # print "inserting type ",type," offset ",offset
                        # print "pt=",pt," type=",type
                        
                        offset += 1
                        
                        self.cellId[pt.x][pt.y][pt.z] = id  

        usedCellTypesList=usedCellTypes.keys()
        usedCellTypesList.sort()
        return usedCellTypesList
        
    # this function is used during prototyping. in production code it is replaced by C++ counterpart            
    def fillConFieldData(self,_cellFieldG,_conField):
        import CompuCell
        
        pt = CompuCell.Point3D(0,0,0) 
        cell = CompuCell.CellG() 
        fieldDim = _cellFieldG.getDim()

        self.dim  = [fieldDim.x, fieldDim.y, fieldDim.z]         

        # # # print "FILL CONFIELDDATA 3D"
        # # # print "self.dim=",self.dim

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        zdim = self.dim[2]+2
        ydim = self.dim[1]+2
        xdim = self.dim[0]+2
        self.conArray.SetNumberOfValues(xdim*ydim*zdim)

        self.cellTypeCon = vtk.vtkIntArray()
        self.cellTypeCon.SetName("concelltype")
        self.cellTypeCon.SetNumberOfValues((self.dim[2]+2)*(self.dim[1]+2)*(self.dim[0]+2))
        
        offset=0        
        # # For some reasons the points x=0 are eaten up (don't know why).
        # # So we just populate empty cellIds.
        # for i in range(self.dim[0]+1):
            # self.conArray.SetValue(offset, 0.0)    
            # offset += 1
        
        
        maxCon = float(_conField.get(pt)) # concentration at pt=0,0,0
        minCon = float(_conField.get(pt)) # concentration at pt=0,0,0
        
        con=0.0
        for k in range(zdim):
            for j in range(ydim):
                for i in range(xdim):                
                    # if padding bogus boundary
#                    if i==0 or i ==self.dim[0]+1 or j==0 or j ==self.dim[1]+1 or k==0 or k ==self.dim[2]+1:
#                        con=0.0
#                        self.conArray.SetValue(offset, con)
#                        type=0
#                        self.cellTypeCon.SetValue(offset,type)
#                    else:
#                        pt.x = i-1
#                        pt.y = j-1
#                        pt.z = k-1
                        pt.x = i
                        pt.y = j
                        pt.z = k
                        
                        # con = float(_conField.get(pt))
                        # con = float((self.dim[1]-pt.y)*(self.dim[0]-pt.x))
                        con = float(pt.y*pt.x)
                        self.conArray.SetValue(offset, con)
                        
                        cell = _cellFieldG.get(pt)
                        
                        if cell is not None:
                            type = int(cell.type)
                            if type in self.invisibleCellTypes:
                                type = 0
                        else:
                            type = 0
                     
                        self.cellTypeCon.SetValue(offset,type)
                        
                        if maxCon < con:
                            maxCon = con
                        
                        if minCon > con:
                            minCon = con
                        
                        
                        offset += 1
#                    offset += 1
        
        return (minCon, maxCon)

    def initCellGlyphsActor3D(self, _glyphActor, _invisibleCellTypes):
#        print MODULENAME,'  ---initCellGlyphsActor3D'
#        print MODULENAME,'    _invisibleCellTypes=', _invisibleCellTypes

        from PySteppables import CellList

        fieldDim=self.currentDrawingParameters.bsd.fieldDim
        sim = self.currentDrawingParameters.bsd.sim
        if (sim == None):
          print 'MVCDrawModel3D.py: initCellGlyphsActor3D(),  sim is empty'
          return
      
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList=CellList(inventory)
        centroidPoints = vtk.vtkPoints()
        cellTypes = vtk.vtkIntArray()
        cellTypes.SetName("CellTypes")

#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")

#        if self.scaleGlyphsByVolume:
        cellScalars = vtk.vtkFloatArray()
        cellScalars.SetName("CellScalars")
        
        cellCount = 0
        
#        if self.hexFlag:
#          print MODULENAME,'   initCellGlyphsActor3D(): doing hex'
#          for cell in cellList:
#              if cell.type in _invisibleCellTypes: continue   # skip invisible cell types
#
#              #print 'cell.id=',cell.id  # = 2,3,4,...
#              #print 'cell.type=',cell.type
#              #print 'cell.volume=',cell.volume
#              xmid = cell.xCOM/1.122
#              ymid = cell.yCOM/1.122
##              zmid = cell.zCOM/1.07457
#              zmid = cell.zCOM
#    #          if cellCount < 50:  print cellCount,' glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
#    #          if cell.volume > 1: print cellCount,' ** glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
#    #          cellCount += 1
#              centroidPoints.InsertNextPoint(xmid,ymid,zmid)
#              cellTypes.InsertNextValue(cell.type)
#    
#    #          if self.scaleGlyphsByVolume:
#              if Configuration.getSetting("CellGlyphScaleByVolumeOn"):       # todo: make class attrib; update only when changes
#                cellScalars.InsertNextValue(cell.volume ** 0.333)   # take cube root of V, to get ~radius
#              else:
#                cellScalars.InsertNextValue(1.0)      # lame way of doing this
#        else:
#        print MODULENAME,'   initCellGlyphsActor3D(): self.offset=',self.offset
        for cell in cellList:
              if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

              #print 'cell.id=',cell.id  # = 2,3,4,...
              #print 'cell.type=',cell.type
              #print 'cell.volume=',cell.volume
              xmid = cell.xCOM     # + self.offset
              ymid = cell.yCOM
              zmid = cell.zCOM
    #          if cellCount < 50:  print cellCount,' glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
    #          if cell.volume > 1: print cellCount,' ** glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
    #          cellCount += 1
              centroidPoints.InsertNextPoint(xmid,ymid,zmid)
              cellTypes.InsertNextValue(cell.type)
    
    #          if self.scaleGlyphsByVolume:
              if Configuration.getSetting("CellGlyphScaleByVolumeOn"):       # todo: make class attrib; update only when changes
                cellScalars.InsertNextValue(cell.volume ** 0.333)   # take cube root of V, to get ~radius
              else:
                cellScalars.InsertNextValue(1.0)      # lame way of doing this


        centroidsPD = vtk.vtkPolyData()
        centroidsPD.SetPoints(centroidPoints)
        centroidsPD.GetPointData().SetScalars(cellTypes)

#        if self.scaleGlyphsByVolume:
        centroidsPD.GetPointData().AddArray(cellScalars)

        centroidGS = vtk.vtkSphereSource()
        thetaRes = Configuration.getSetting("CellGlyphThetaRes")     # todo: make class attrib; update only when changes
        phiRes = Configuration.getSetting("CellGlyphPhiRes")            
        centroidGS.SetThetaResolution(thetaRes)  # increase these values for a higher-res sphere glyph
        centroidGS.SetPhiResolution(phiRes)

        centroidGlyph = vtk.vtkGlyph3D()
        
        if VTK_MAJOR_VERSION>=6:
            centroidGlyph.SetInputData(centroidsPD)
        else:    
            centroidGlyph.SetInput(centroidsPD)        
        
        centroidGlyph.SetSource(centroidGS.GetOutput())
        
        glyphScale = Configuration.getSetting("CellGlyphScale")            
        centroidGlyph.SetScaleFactor( glyphScale )
        centroidGlyph.SetIndexModeToScalar()
        centroidGlyph.SetRange(0,self.celltypeLUTMax)

        centroidGlyph.SetColorModeToColorByScalar()
#        if self.scaleGlyphsByVolume:
        centroidGlyph.SetScaleModeToScaleByScalar()
        
#        centroidGlyph.SetScaleModeToDataScalingOff()  # call this to disable scaling by scalar value
#        centroidGlyph.SetScaleModeToDataScalingOn()   # method doesn't even exist?!

        centroidGlyph.SetInputArrayToProcess(3,0,0,0,"CellTypes")
        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellScalars")

        if VTK_MAJOR_VERSION>=6:
            self.cellGlyphsMapper.SetInputData(centroidGlyph.GetOutput())
        else:    
            self.cellGlyphsMapper.SetInput(centroidGlyph.GetOutput())


        
        self.cellGlyphsMapper.SetScalarRange(0,self.celltypeLUTMax)
        self.cellGlyphsMapper.ScalarVisibilityOn()
        
        self.cellGlyphsMapper.SetLookupTable(self.celltypeLUT)   # defined in parent class
#        print MODULENAME,' usedCellTypesList=' ,self.usedCellTypesList

        _glyphActor.SetMapper(self.cellGlyphsMapper)  # Note: we don't need to scale actor for hex lattice here since using cell info

#---------------------------------------------------------------------------
    def initFPPLinksActor3D(self, _fppActor, _invisibleCellTypes):
#        print MODULENAME,'  initFPPLinksActor3D'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell
        
        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor3D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):  # bogus check
          print MODULENAME,'    fppPlugin is null, returning'
          return

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print 'fieldDim, fieldDim.x =',fieldDim,fieldDim.x
        xdim = fieldDim.x
        ydim = fieldDim.y
        zdim = fieldDim.z
        
        # To test if links should be stubs (for wraparound on periodic BCs)
        xdim_delta = xdim/2
        ydim_delta = ydim/2
        zdim_delta = zdim/2
        
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList = CellList(inventory)
        
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0

        for cell in cellList:
          if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue
          xmid0 = cell.xCOM    # + self.offset
          ymid0 = cell.yCOM
          zmid0 = cell.zCOM
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,zmid0)
          
          endPt = beginPt + 1
          
#2345678901234
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):  # First pass (Internal list)
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM    # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))  
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
#                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
                numStubs = 0
                if abs(xdiff) > xdim_delta:   # wraps around in x-direction
                    numStubs += 1
#                    print '>>>>>> wraparound X'
                    ymid0end = ymid0
                    zmid0end = zmid0
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1
#                else:   # wraps around in y-direction
                if abs(ydiff) > ydim_delta:   # wraps around in y-direction
                    numStubs += 1
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    zmid0end = zmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1
                    
                if abs(zdiff) > zdim_delta:   # wraps around in z-direction
                    numStubs += 1
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    ymid0end = ymid0
                    if zdiff < 0:
                      zmid0end = zmid0 + self.stubSize
                    else:
                      zmid0end = zmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = zdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)

                if numStubs > 1: print MODULENAME,"  --------------  numStubs = ",numStubs

            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' (internal link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#         ---------------------------------------
#2345678901234 
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):   # Second pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   #  + self.offset   # used to do: float(fppd.neighborAddress.xCM) / vol + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))  
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
#                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
                numStubs = 0
                if abs(xdiff) > xdim_delta:   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    numStubs += 1
                    ymid0end = ymid0
                    zmid0end = zmid0
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1
                    
                if abs(ydiff) > ydim_delta:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    numStubs += 1
                    xmid0end = xmid0
                    zmid0end = zmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1
                    
                if abs(zdiff) > zdim_delta:   # wraps around in z-direction
#                    print '>>>>>> wraparound Z'
                    numStubs += 1
                    xmid0end = xmid0
                    ymid0end = ymid0
                    if zdiff < 0:
                      zmid0end = zmid0 + self.stubSize
                    else:
                      zmid0end = zmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' ----- (external link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#2345678901234
#          print 'after external links: beginPt, endPt=',beginPt,endPt
          beginPt = endPt  # update point index 

        #-----------------------
        if lineNum == 0:  return
#        print '---------- # links=',lineNum
        
        # create Blue-Red LUT
#        lutBlueRed = vtk.vtkLookupTable()
#        lutBlueRed.SetHueRange(0.667,0.0)
#        lutBlueRed.Build()

#        print '---------- # links,scalarValMin,Max =',lineNum,scalarValMin,scalarValMax
        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)

        
        
        if VTK_MAJOR_VERSION>=6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:    
            FPPLinksPD.Update()
            self.FPPLinksMapper.SetInput(FPPLinksPD)
        
        
        

#        self.FPPLinksMapper.SetScalarModeToUseCellFieldData()
        
        _fppActor.SetMapper(self.FPPLinksMapper)  # Note: we don't need to scale actor for hex lattice here since using cell info

        
#---------------------------------------------------------------------------
    def initFPPLinksColorActor3D(self, _fppActor, _invisibleCellTypes):
#        print MODULENAME,'  initFPPLinksActor3D_color'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell
        
        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor3D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):  # bogus check
          print '    fppPlugin is null, returning'
          return

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print 'fieldDim, fieldDim.x =',fieldDim,fieldDim.x
        xdim = fieldDim.x
        ydim = fieldDim.y
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList = CellList(inventory)
        
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colorScalars = vtk.vtkFloatArray()
        colorScalars.SetName("fpp_scalar")

#        cellTypes = vtk.vtkIntArray()
#        cellTypes.SetName("CellTypes")
#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0
        scalarValMin = 1000.0
        scalarValMax = -scalarValMin

        for cell in cellList:
          if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue
          xmid0 = cell.xCOM    # + self.offset
          ymid0 = cell.yCOM
          zmid0 = cell.zCOM
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,zmid0)
          
          endPt = beginPt + 1
          
#2345678901
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):  # First pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
#            d2 = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff)  # compute dist^2 and avoid sqrt
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))  
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                zmid0end = zmid0 
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                scalarVal = d2/targetDist2    # actual^2/target^2
#                scalarVal = actualDist / fppd.targetDistance    # actual/target
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist 
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)
                
                lineNum += 1
                endPt += 1

        
#         ---------------------------------------
#2345678901
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):   # Second pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
#            d2 = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff)  # compute dist^2 and avoid sqrt
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                zmid0end = zmid0 
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                scalarVal = d2/targetDist2    # actual^2/target^2
#                scalarVal = actualDist / fppd.targetDistance    # actual/target
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist 
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)
                
                lineNum += 1
                endPt += 1

          beginPt = endPt  # update point index 
          
          #--------------------------------------

        if lineNum == 0:  return
        
        # create Blue-Red LUT
#        lutBlueRed = vtk.vtkLookupTable()
#        lutBlueRed.SetHueRange(0.667,0.0)
#        lutBlueRed.Build()

#        print '---------- # links,scalarValMin,Max =',lineNum,scalarValMin,scalarValMax
        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)

        
        
        if VTK_MAJOR_VERSION>=6:
            pass
        else:    
            FPPLinksPD.Update()
        
        FPPLinksPD.GetCellData().SetScalars(colorScalars)
        
        if VTK_MAJOR_VERSION>=6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:    
            self.FPPLinksMapper.SetInput(FPPLinksPD)
        
        
        

        self.FPPLinksMapper.SetScalarModeToUseCellFieldData()
        self.FPPLinksMapper.SelectColorArray("fpp_scalar")
        self.FPPLinksMapper.SetScalarRange(scalarValMin,scalarValMax)

        self.FPPLinksMapper.SetLookupTable(self.lutBlueRed)
        
        _fppActor.SetMapper(self.FPPLinksMapper)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(self.lutBlueRed)
        #scalarBar.SetTitle("Stress")
        scalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        #scalarBar.GetPositionCoordinate().SetValue(0.8,0.05)
        scalarBar.SetOrientationToVertical()
        scalarBar.SetWidth(0.1)
        scalarBar.SetHeight(0.9)
        scalarBar.SetPosition(0.88,0.1)
        #scalarBar.SetLabelFormat("%-#6.3f")
        scalarBar.SetLabelFormat("%-#3.1f")
        scalarBar.GetLabelTextProperty().SetColor(1,1,1)
        #scalarBar.GetTitleTextProperty().SetColor(1,0,0)

#        self.graphicsFrameWidget.ren.AddActor2D(scalarBar)    

    def configsChanged(self):
        self.populateLookupTable()
        #reassign which types are invisible        
        self.set3DInvisibleTypes()
        self.parentWidget.requestRedraw()
