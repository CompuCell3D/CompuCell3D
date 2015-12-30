import sys, os

def setVTKPaths():
   #import sys
   #from os import environ
   import string
   #import sys
   platform=sys.platform
   if platform=='win32':
      sys.path.insert(0,environ["PYTHON_DEPS_PATH"])
      # sys.path.append(environ["VTKPATH"])
   
      # sys.path.append(os.environ["VTKPATH"])
      # sys.path.append(os.environ["VTKPATH1"])
      # sys.path.append(os.environ["PYQT_PATH"])
      # sys.path.append(os.environ["SIP_PATH"])
      # sys.path.append(os.environ["SIP_UTILS_PATH"])
#   else:
#      swig_path_list=string.split(environ["VTKPATH"])
#      for swig_path in swig_path_list:
#         sys.path.append(swig_path)


# setVTKPaths()

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import Graphics

from PyQt4 import QtCore, QtGui,QtOpenGL
#import vtk
import Configuration
import vtk, math
#import sys, os

from Plugins.ViewManagerPlugins.SimpleTabView import FIELD_TYPES,PLANES
from DrawingParameters import DrawingParameters
from CustomActorsStorage import CustomActorsStorage
from CameraSettings import CameraSettings

MODULENAME='----- MVCDrawViewBase.py: '

XZ_Z_SCALE=math.sqrt(6.0)/3.0
YZ_Y_SCALE=math.sqrt(3.0)/2.0
YZ_Z_SCALE=math.sqrt(6.0)/3.0


from  Messaging import dbgMsg

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()
VTK_MINOR_VERSION=vtk.vtkVersion.GetVTKMinorVersion()
VTK_BUILD_VERSION=vtk.vtkVersion.GetVTKBuildVersion()


class MVCDrawViewBase:
    def __init__(self, _drawModel , graphicsFrameWidget, parent=None):
        self.legendActor    = vtk.vtkScalarBarActor()
        self.legendActor.SetNumberOfLabels(8)
        (self.minCon, self.maxCon) = (0, 0)
#        print MODULENAME,"graphicsFrameWidget=",graphicsFrameWidget
#        print MODULENAME,"parent=",parent

        self.plane = 'XY'
        self.planePos = 0

        # # # self.drawModel = _drawModel
        self.parentWidget = parent
#        print MODULENAME,'  __init__: parentWidget=',self.parentWidget
        # from weakref import ref
        # self.graphicsFrameWidget = ref(graphicsFrameWidget)
        # gfw=self.graphicsFrameWidget()
        # self.qvtkWidget = self.graphicsFrameWidget.qvtkWidget
        
        
        
        
        from weakref import ref
        
        dM = ref(_drawModel)
        self.drawModel=dM()
        
        
        gfw=ref(graphicsFrameWidget)
        self.graphicsFrameWidget = gfw()
        
        
        # qvtk=ref(self.graphicsFrameWidget.qvtkWidget)
        
        
        # self.qvtkWidget = qvtk()
        
        self.qvtkWidget = ref(self.graphicsFrameWidget.qvtkWidget)
        
        
        
        # # # self.graphicsFrameWidget = graphicsFrameWidget
        # # # self.qvtkWidget = self.graphicsFrameWidget.qvtkWidget
        
        
        self.currentDrawingFunction = None
        self.currentActors = {} # dictionary of current actors
        self.drawingFcnName = "" # holds a string describing name of the drawing fcn . Used to determine if current actors need to be removed before next drawing
        self.drawingFcnHasChanged = True
        self.fieldTypes = None 
        self.currentDrawingParameters = DrawingParameters()
        self.currentFieldType = ("Cell_Field", FIELD_TYPES[0])
        self.__initDist = 0 # initial camera distance - used in zoom functions
        
        
        #CUSTOM ACTORS
        self.customActors = {} #{visName: CustomActorsStorage() }
        self.currentCustomVisName = '' #stores name of the current custom visualization
        self.currentVisName = '' #stores name of the current visualization         
        self.cameraSettingsDict = {} # {fieldName:CameraSettings()}
        
    # def __del__(self):
        # print '\n\n\n\n CLEANING UP ',MODULENAME
        
    def  version_identifier(self, major, minor, build):
        return major*10**6+minor*10**3+build

    def vtk_version_identifier(self):
        return self.version_identifier(VTK_MAJOR_VERSION,VTK_MINOR_VERSION,VTK_BUILD_VERSION)
        
    def setDrawingFunctionName(self,_fcnName):
        # print "\n\n\n THIS IS _fcnName=",_fcnName," self.drawingFcnName=",self.drawingFcnName
        
        if self.drawingFcnName != _fcnName:
            self.drawingFcnHasChanged = True
        else:
            self.drawingFcnHasChanged = False
        self.drawingFcnName = _fcnName
     
    def clearEntireDisplay(self):
        
        actorsCollection=self.graphicsFrameWidget.ren.GetActors()
        # print 'actorsCollection=',actorsCollection
        actorsList=[]
        numberOfActors=actorsCollection.GetNumberOfItems()
        print 'numberOfActors=',numberOfActors
        for i in range(numberOfActors):
            actor=actorsCollection.GetItemAsObject(i)
            actorsList.append(actor)
            
        for actor in actorsList:    
            self.graphicsFrameWidget.ren.RemoveActor(actor)            
            
        actorsList =[]   
        print 'actorsList=',actorsList    
        
    def clearDisplay(self):   # called whenever user selects a different field to render; beware NOT doing this because, for example, it wouldn't redraw a dynamic scalarbar
        
#        print MODULENAME,"   ---------  clearDisplay()"

        # # # actorsCollection=self.graphicsFrameWidget.ren.GetActors()
        # # # print 'actorsCollection=',actorsCollection

        # # # print 'self.currentActors=',self.currentActors
        for actor in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actor])
            
        self.currentActors.clear()
    
    def Render(self):
#        print MODULENAME,"   ---------  Render()"
        self.graphicsFrameWidget.Render()
        
    #this is an ugly solution that seems to work on 32 bit machines. We will see if it will work on other machines        
    def extractAddressIntFromVtkObject(self,_vtkObj):
        # pointer_ia=ia.__this__
        # print "pointer_ia=",pointer_ia
        # address=pointer_ia[1:9]
        # print "address=",address," int(address)=",int(address,16)
        return self.parentWidget.fieldExtractor.unmangleSWIGVktPtrAsLong(_vtkObj.__this__)
        # return int(_vtkObj.__this__[1:9],16)
                        
    def setFieldTypes(self,_fieldTypes):
        self.fieldTypes = _fieldTypes        

    def setPlotData(self,_plotData):
        self.currentFieldType = _plotData        
        
    def drawFieldLocal(self, _bsd,_useFieldComboBox=True):
        # print 'resetting camera in drawFieldLocal View base'
        # self.resetAllCameras()    
        # import time
        # print 'BEFORE INSIDEE graphicsFrame.drawFieldLocal'    
        # time.sleep(5)    
        # return
        fieldType = ("Cell_Field", FIELD_TYPES[0])
#        print MODULENAME,  "drawFieldLocal():  fieldType=",fieldType
        # print "DrawLocal"
        plane = self.getPlane()        
#        print MODULENAME,  "drawFieldLocal():  plane=",plane
        self.drawModel.setDrawingParameters(_bsd,plane[0],plane[1],fieldType)
        
        
        self.currentDrawingParameters.bsd = _bsd
        self.currentDrawingParameters.plane = plane[0]
        self.currentDrawingParameters.planePos = plane[1]
        self.currentDrawingParameters.fieldName = fieldType[0]
        self.currentDrawingParameters.fieldType = fieldType[1]        
        self.drawModel.setDrawingParametersObject(self.currentDrawingParameters)
        

            
            
        if self.fieldTypes is not None:
            if _useFieldComboBox:
                name = str(self.graphicsFrameWidget.fieldComboBox.currentText())
                # print "name=",name
                # print "self.fieldTypes=",self.fieldTypes
                fieldType = (name, self.fieldTypes[name])                                    
                self.currentFieldType = fieldType
            else:                
                name = self.currentFieldType[0]
                # print "name=",name
                # print "self.fieldTypes=",self.fieldTypes
                fieldType=(name, self.fieldTypes[name])                    
                # print "fieldType=",fieldType
                self.currentFieldType = fieldType     # this assignment seems redundent but I keep it in order to make sure nothing breaks
                
            (currentPlane, currentPlanePos) = self.getPlane()
#            print MODULENAME,  "drawFieldLocal():  setDrawingFnName=","draw"+fieldType[1]+currentPlane
            self.setDrawingFunctionName("draw"+fieldType[1]+currentPlane)
#            print "---- MVCDrawViewBase.py: DrawingFunctionName=","draw"+fieldType[1]+currentPlane
            
            # # self.parentWidget.setFieldType((name, self.parentWidget.fieldTypes[name])) 
            plane = (currentPlane, currentPlanePos)            
            self.drawModel.setDrawingParameters(_bsd,plane[0],plane[1],fieldType)
            
            self.currentDrawingParameters.bsd = _bsd
            self.currentDrawingParameters.plane = plane[0]
            self.currentDrawingParameters.planePos = plane[1]
            self.currentDrawingParameters.fieldName = fieldType[0]
            self.currentDrawingParameters.fieldType = fieldType[1]        
            self.drawModel.setDrawingParametersObject(self.currentDrawingParameters)
            
        
       
        self.drawField(_bsd,fieldType)        
        
        # print 'AFTER INSIDEE graphicsFrame.drawFieldLocal'    
        # time.sleep(5)           
  
        
#         print 'plane ',plane 
#         print self.parentWidget.latticeType
#         print self.currentActors
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.getSim3DFlag():
            for actorName, actor in self.currentActors.iteritems():
                if not hasattr(actor,'SetScale'): # skipping all the actors that cannot be scaled
                    continue
                if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"]:
                    if self.plane=="XY":
                        actor.SetScale(1.0,1.0,1.0)
#                         print 'XY actorName=',actorName," scale=",actor.GetScale()     
                    
                    elif self.plane=="XZ":
                        actor.SetScale(1.0,XZ_Z_SCALE,1.0)     
#                         print 'XZ actorName=',actorName," scale=",actor.GetScale()                       

                    elif self.plane=="YZ":
                        actor.SetScale(YZ_Y_SCALE,YZ_Z_SCALE,1.0)
#                         print 'YZ actorName=',actorName," scale=",actor.GetScale()     


        self.qvtkWidget().repaint()
        

        
        
        
    def drawField(self, _bsd, fieldType):   

        # print 'drawField ', fieldType
        dbgMsg('drawField ', fieldType)
        resetCamera = False # we reset camera only for visualizations for which camera settings are not in the dictionary and users have not requested custom cameras
        
        if self.drawingFcnHasChanged:            
            self.clearDisplay()
            
        drawField = getattr(self, "draw" + fieldType[1])   # e.g. "drawCellField"
        
        
        cs = None #camera settings
        
        if self.currentDrawingFunction != drawField: # changing type of drawing function e.g. from drawCellField to drawConField- need to remove actors that are currently displayed            

            for actorName in self.currentActors.keys():                                                
                self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
                del self.currentActors[actorName]
                
            # to prevent cyclic reference we user weakre
            from weakref import ref    
            self.currentDrawingFunction = ref(drawField)
            
            # # # self.currentDrawingFunction = drawField   
        
        
        
        currentDrawingFunction=self.currentDrawingFunction() # obtaining object from weakref
        if not currentDrawingFunction:return
        # print 'currentDrawingFunction=',currentDrawingFunction
        # import time
        # time.sleep(2)
        # return
        
        # here we handle actors for custom visualization when the name of the function does not change (it is drawCustomVis) but the name of the plot changes (hence actors have to be replaced with new actors)
        drawFieldCustomVis = getattr(self, "drawCustomVis")
        if currentDrawingFunction==drawFieldCustomVis:
            #check if actors the name of the custom vis has changed            
            if self.currentCustomVisName != self.currentDrawingParameters.fieldName:
                self.currentCustomVisName = self.currentDrawingParameters.fieldName

                for actorName in self.currentActors.keys():
                    self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
                    del self.currentActors[actorName]
                
        try:
        
            fieldName = self.currentDrawingParameters.fieldName                                   
            cs = self.cameraSettingsDict[fieldName]
            
            if self.checkIfCameraSettingsHaveChanged(cs):
                if self.currentVisName==self.currentDrawingParameters.fieldName: # this section is called when camera setings have changed between calls to this fcn (e.g. when screen refreshes with new MCS data) and the visualzation field was not change by the user                    
                    cs = self.getCurrentCameraSettings()
                    self.cameraSettingsDict[fieldName] = cs
                    
                    
                    self.setCurrentCameraSettings(cs)
                else:                    # this section is called when camera settings have changed between calls to this function and visualization field changes. Before initializing camera with cs for new vis field setting we store cs for  previous vis field
                    self.cameraSettingsDict[self.currentVisName] = self.getCurrentCameraSettings()         
                    
                    self.setCurrentCameraSettings(cs)
        except LookupError,e:
            resetCamera=True
            
            if self.currentVisName!=self.currentDrawingParameters.fieldName and self.currentVisName!='': # this is called when user modifies camera in one vis and then changes vis to another  for which camera has not been set up                 
                self.cameraSettingsDict[self.currentVisName] = self.getCurrentCameraSettings()

        self.currentVisName = self.currentDrawingParameters.fieldName    # updating current vis name        
        
        

        
        
        drawField(_bsd, fieldType)        




        
        if resetCamera:
            self.qvtkWidget().resetCamera() 
            self.cameraSettingsDict[fieldName] = self.getCurrentCameraSettings()
        
        # # # # here we handle actors for custom visualization when the name of the function does not change (it is drawCustomVis) but the name of the plot changes (hence actors have to be replaced with new actors)
        # drawFieldCustomVis = getattr(self, "drawCustomVis")
        # if self.currentDrawingFunction==drawFieldCustomVis:
        #     #check if actors the name of the custom vis has changed
        #     if self.currentCustomVisName != self.currentDrawingParameters.fieldName:
        #         self.currentCustomVisName = self.currentDrawingParameters.fieldName
        #
        #         for actorName in self.currentActors.keys():
        #             self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
        #             del self.currentActors[actorName]
        #
        # try:
        #
        #     fieldName = self.currentDrawingParameters.fieldName
        #     cs = self.cameraSettingsDict[fieldName]
        #
        #     if self.checkIfCameraSettingsHaveChanged(cs):
        #         if self.currentVisName==self.currentDrawingParameters.fieldName: # this section is called when camera setings have changed between calls to this fcn (e.g. when screen refreshes with new MCS data) and the visualzation field was not change by the user
        #             cs = self.getCurrentCameraSettings()
        #             self.cameraSettingsDict[fieldName] = cs
        #
        #
        #             self.setCurrentCameraSettings(cs)
        #         else:                    # this section is called when camera settings have changed between calls to this function and visualization field changes. Before initializing camera with cs for new vis field setting we store cs for  previous vis field
        #             self.cameraSettingsDict[self.currentVisName] = self.getCurrentCameraSettings()
        #
        #             self.setCurrentCameraSettings(cs)
        # except LookupError,e:
        #     resetCamera=True
        #
        #     if self.currentVisName!=self.currentDrawingParameters.fieldName and self.currentVisName!='': # this is called when user modifies camera in one vis and then changes vis to another  for which camera has not been set up
        #         self.cameraSettingsDict[self.currentVisName] = self.getCurrentCameraSettings()
        #
        # self.currentVisName = self.currentDrawingParameters.fieldName    # updating current vis name
        #
        #
        #
        #
        #
        # drawField(_bsd, fieldType)
        #
        #
        #
        #
        #
        # if resetCamera:
        #     qvtkWidget_obj = self.qvtkWidget()
        #     if qvtkWidget_obj:
        #         qvtkWidget_obj.resetCamera()
        #         self.cameraSettingsDict[fieldName] = self.getCurrentCameraSettings()
        
       
    def resetAllCameras(self) :             
            self.cameraSettingsDict={}
            # return
            # print 'THIS IS VIEWBASE = ',self
            # self.qvtkWidget.resetCamera() 
            # for fieldName in self.cameraSettingsDict:        
                # print 'fieldName=',fieldName
                # self.cameraSettingsDict[fieldName] = self.getCurrentCameraSettings()
                # # self.cameraSettingsDict[fieldName] = None
                
            # set camera seettings for the current scene    
            # print 'self.graphicsFrameWidget=',self.graphicsFrameWidget
            # self.graphicsFrameWidget.clearDisplayOnDemand()
            
    
    def drawCellField(self, _bsd, fieldType): pass
    
    def drawConField(self, _bsd, fieldType): pass
    
    def drawVectorFieldCellLevel(self, _bsd, fieldType): pass
    
    def drawVectorField(self, _bsd, fieldType): pass
    
    def drawScalarFieldCellLevel(self, _bsd, fieldType): pass     
    
    def drawScalarField(self, _bsd, fieldType): pass     

    def drawCustomVis(self,bsd,fieldType):
        import CompuCellSetup
        
        visName = self.currentDrawingParameters.fieldName
        visData = CompuCellSetup.customVisStorage.getVisData(visName)
                
        callbackFcn = visData.getVisCallbackFunction()
        actorsDict = visData.getActorDict()        
        
        if not self.customActors.has_key(visName):
            actorsStorage = CustomActorsStorage(visName)
            self.customActors[visName] = actorsStorage
            
            for actorName in actorsDict.keys():
                actorType = actorsDict[actorName]
                if actorType[0:3] == 'vtk':  # check, since we now piggyback off the actorsDict to allow a 'customVTKScript' for .dml 
                    exec("from vtk import "+actorType+"\n")
                    actor = eval(actorType+"()") 
                    actorsStorage.addActor(actorName,actor)    
                
        actorsStorage = self.customActors[visName]        
        # print "actorsStorage=",actorsStorage.actorsDict        
        actorsOrderList = actorsStorage.getActorsInTheOrderOfAppearance()

        for i in range(0,len(actorsOrderList),2):
            actorName = actorsOrderList[i]
            actorObj = actorsOrderList[i+1]
            if not self.currentActors.has_key(actorName):
                self.currentActors[actorName] = actorObj
                self.graphicsFrameWidget.ren.AddActor(actorObj)
        # print "self.currentActors=",self.currentActors    
        # print "actorsStorage.getActorsDict()=",actorsStorage.getActorsDict()
                
        #have to construct dictionary of actors  used in the specific visualization - actorsDict=visDataDict[visName] has list of names of actors used in the visualization visNAme
        if callbackFcn is not None:
            callbackFcn(actorsStorage.getActorsDict())
            
        self.Render()    
    
    def showContours(self, enable): pass
    
    def setPlane(self, plane, pos): pass
    
    def getPlane(self):
        return ("",0)
    
    def getCamera(self):
        return self.ren.GetActiveCamera()
        
    def initSimArea(self, _bsd):
        
        fieldDim   = _bsd.fieldDim
        # sim.getPotts().getCellFieldG().getDim()
        self.setCamera(fieldDim)
       
    def configsChanged(self): pass
        
    # Transforms interval [0, 255] to [0, 1]
    def toVTKColor(self, val):
        return float(val)/255

    def largestDim(self, dim):
        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]
                
        return ldim
    
    def getSim3DFlag(self):
        zdim = self.currentDrawingParameters.bsd.fieldDim.z
#        print MODULENAME,'  getSim3DFlag, zdim=',zdim
        if zdim > 1:
            return True
        else:
            return False
    
    def setParams(self):
        # You can use either Build() method (256 color by default) or
        # SetNumberOfTableValues() to allocate much more colors!
        self.lut = vtk.vtkLookupTable()
        # You need to explicitly call Build() when constructing the LUT by hand     
        self.lut.Build()
        self.populateLookupTable()
        # self.dim = [100, 100, 1] # Default values
    
    def populateLookupTable(self):  # rwh: why is this method in both View & Model?
#        print MODULENAME,' populateLookupTable()'
        self.drawModel.populateLookupTable() # have to update colors in model objects
        colorMap = Configuration.getSetting("TypeColorMap")
        for key in colorMap.keys():
            r = colorMap[key].red()
            g = colorMap[key].green()
            b = colorMap[key].blue()
            self.lut.SetTableValue(key, self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b), 1.0)
        # had to get rid of Render/repaint statements from this fcns because one of these is called elswhere and apparently calling them multiple times may cause software crash    
        # self.qvtkWidget.repaint() 
        
        # self.graphicsFrameWidget.Render()
        
    # Do I need this method?
    # Calculates min and max concentration
    def findMinMax(self, conField, dim):
        import CompuCell
        pt = CompuCell.Point3D() 

        maxCon = 0
        minCon = 0
        for k in range(dim[2]):
            for j in range(dim[1]):
                for i in range(dim[0]):
                    pt.x = i
                    pt.y = j
                    pt.z = k
                    
                    con = float(conField.get(pt))
                    
                    if maxCon < con:
                        maxCon = con
                    
                    if minCon > con:
                        minCon = con

        # Make sure that the concentration is positive
        if minCon < 0:
            minCon = 0

        return (minCon, maxCon)

    # Just returns min and max concentration
    def conMinMax(self):        
        return (self.drawModel.minCon, self.drawModel.maxCon)
    
    def frac(self, con, minCon, maxCon):
        if maxCon == minCon:
            return 0.0
        else:
            frac = (con - minCon)/(maxCon - minCon)
            
        if frac > 1.0:
            frac = 1.0
            
        if frac < 0.0:
            frac = 0.0

        return frac

    def showLegend(self, enable):
        if enable:
            if not self.currentActors.has_key("LegendActor"):
                self.currentActors["LegendActor"]=self.legendActor
                self.graphicsFrameWidget.ren.AddActor(self.legendActor)
        else:
            if self.currentActors.has_key("LegendActor"):
                del self.currentActors["LegendActor"]
                self.graphicsFrameWidget.ren.RemoveActor(self.legendActor)
        self.Render()
        self.graphicsFrameWidget.repaint()
                
        # self.repaint()

    def showAxes(self,flag=True):
        pass

    def setZoomItems(self, zitems):
        self.zitems = zitems

    def showBorder(self): pass
    def hideBorder(self): pass
    
    def showCells(self): pass
    def hideCells(self): pass
    
    def __zoomStep(self, delta):
        # # # print "ZOOM STEP"
        if self.graphicsFrameWidget.ren:
            # renderer = self.GetCurrentRenderer()
            camera = self.graphicsFrameWidget.ren.GetActiveCamera()
            
            zoomFactor = math.pow(1.02,(0.5*(delta/8)))

            # I don't know why I might need the parallel projection
            if camera.GetParallelProjection(): 
                parallelScale = camera.GetParallelScale()/zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                self.graphicsFrameWidget.ren.ResetCameraClippingRange()

            self.Render()
    
#     def zoomIn(self):
#         delta = 2*120
#         self.__zoomStep(delta)
#
#     def zoomOut(self):
#         delta = -2*120
#         self.__zoomStep(delta)
#
#     def zoomFixed(self, val):
# #        print MODULENAME,"zitems=",self.zitems
#         if self.graphicsFrameWidget.ren:
#             # renderer = self._CurrentRenderer
#             camera = self.graphicsFrameWidget.ren.GetActiveCamera()
#             self.__curDist = camera.GetDistance()
#
#             # To zoom fixed, dolly should be set to initial position
#             # and then moved to a new specified position!
#
#             if not self.__initDist:
#                 # fieldDim=self.currentDrawingParameters.bsd.fieldDim
#                 fieldDim=self.currentDrawingParameters.bsd.fieldDim
#
#                 self.dim=[fieldDim.x,fieldDim.y,fieldDim.z]
#
#                 self.__initDist=self.largestDim(self.dim)*2
#
#             if (self.__initDist != 0):
#                 # You might need to rewrite the fixed zoom in case if there
#                 # will be flickering
#                 camera.Dolly(self.__curDist/self.__initDist)
#
#             camera.Dolly(self.zitems[val])
#             self.graphicsFrameWidget.ren.ResetCameraClippingRange()
#
#             self.Render()
    
    def takeShot(self): pass
    
    def getCurrentCameraSettings(self):
        cs=CameraSettings()        
        cam = self.graphicsFrameWidget.ren.GetActiveCamera()        
        # print "GET cam=",cam
        cs.position=cam.GetPosition()
        cs.focalPoint=cam.GetFocalPoint()
        cs.viewUp=cam.GetViewUp()
        # cs.viewPlaneNormal=        
        cs.clippingRange=cam.GetClippingRange()
        cs.distance=cam.GetDistance()
        cs.viewAngle=cam.GetViewAngle()
        cs.parallelScale=cam.GetParallelScale()
        return cs
        
    def setCurrentCameraSettings(self,_cs):
        cam = self.graphicsFrameWidget.ren.GetActiveCamera()        
        cam.SetClippingRange(_cs.clippingRange)        
        cam.SetFocalPoint(_cs.focalPoint)
        cam.SetPosition(_cs.position)                
        cam.SetViewUp(_cs.viewUp)
        cam.SetDistance(_cs.distance)
        cam.SetViewAngle(_cs.viewAngle)
        cam.SetParallelScale(_cs.parallelScale)
        
        # print "SET cam=",cam        
        
    def checkIfCameraSettingsHaveChanged(self,_cs):
        cam = self.graphicsFrameWidget.ren.GetActiveCamera()        
        
        position=cam.GetPosition()        
        if position[0]!=_cs.position[0] or position[1]!=_cs.position[1] or position[2]!=_cs.position[2]:
            return True
            
        clippingRange=cam.GetClippingRange()                 
        if clippingRange[0]!=_cs.clippingRange[0] or clippingRange[1]!=_cs.clippingRange[1]:
            return True

        focalPoint=cam.GetFocalPoint()        
        if focalPoint[0]!=_cs.focalPoint[0] or focalPoint[1]!=_cs.focalPoint[1] or focalPoint[2]!=_cs.focalPoint[2]:
            return True
            
        return False       

    def cloneCamera(self,_camera):
        cam = self.ren.GetActiveCamera()
        # cam.ApplyTransform(_camera.GetViewTransformObject())
        cam.SetClippingRange(_camera.GetClippingRange())
        
        cam.SetFocalPoint(_camera.GetFocalPoint())
        cam.SetPosition(_camera.GetPosition())
        
        # cam.SetViewPlaneNormal(_camera.GetViewPlaneNormal())
        cam.SetViewUp(_camera.GetViewUp())
        
        # cam.SetRoll(_camera.GetRoll())
        
        # cam.SetDistance(_camera.GetDistance())
        # cam.SetViewShear(_camera.GetViewShear())
        
        
        # cam.SetDistance(_camera.GetDistance())
        
        # cam.SetViewAngle(_camera.GetViewAngle())
        
    def setCamera(self, fieldDim): 
        camera = self.GetCurrentRenderer().GetActiveCamera()
        
        #self.setDim(dim)
        # Should I specify these parameters explicitly? 
        # What if I change dimensions in XML file? 
        # The parameters should be set based on the configuration parameters!
        # Should it set depending on projection? (e.g. xy, xz, yz)
        
        distance = 200 #self.largestDim()*2 # 273.205 #
        # FIXME: Hardcoded numbers
        
        # camera.SetPosition(50, 50, distance)
        # camera.SetFocalPoint(50, 50, 0)
        # camera.SetClippingRange(distance - 100, distance + 100)
        # # self.GetCurrentRenderer().ResetCameraClippingRange()
        # self.ren.ResetCameraClippingRange()

    def startMovie(self):
        self.w = vtk.vtkWindowToImageFilter()
        self.w.SetInput(self.GetRenderWindow())
        
        self.movie = vtk.vtkMPEG2Writer() # Check is it is available.
        self.movie.SetInput(self.w.GetOutput())
        self.movie.SetFileName("cellsort2D.mpg")
        self.movie.Start()

    def writeMovie(self):
        self.movie.Write()
        self.w = vtk.vtkWindowToImageFilter()
        self.w.SetInput(self.GetRenderWindow())
        self.movie.SetInput(self.w.GetOutput())

    def endMovie(self):
        self.movie.End()

    def setStatusBar(self, statusBar):
        self._statusBar = statusBar

    # Break the settings read into groups?
    def readSettings(self):   # only performed at startup
#        print MODULENAME,'----- readSettings()'
        self.readColorsSets()
        self.readViewSets()
#        self.readColormapSets()
        self.readOutputSets()
#        self.readVectorSets()
        self.readVisualSets()
        # simDefaults?

    def readColorsSets(self):
#        print MODULENAME,'----- readColorsSets()'
        #colorsDefaults
        self._colorMap     = Configuration.getSetting("TypeColorMap")
        self._borderColor  = Configuration.getSetting("BorderColor")
        self._contourColor = Configuration.getSetting("ContourColor")
        self._brushColor   = Configuration.getSetting("BrushColor")
        self._penColor     = Configuration.getSetting("PenColor")
        self._windowColor     = Configuration.getSetting("WindowColor")
        self._boundingBoxColor     = Configuration.getSetting("BoundingBoxColor")

    def readViewSets(self):
        # For 3D only?
        # viewDefaults
        self._types3D      = Configuration.getSetting("Types3DInvisible")

    # this method is assuming these attributes are global (not per-field)
    def readColormapSets_old(self):
        print
        print MODULENAME,' >>>>>>>>>>>>>>>>>>>>>>>  readColormapSets():  doing Config-.getSetting...'
        print

        # colormapDefaults
#        self._minCon       = Configuration.getSetting("MinRange")
#        self._minConFixed  = Configuration.getSetting("MinRangeFixed")
#        self._maxCon       = Configuration.getSetting("MaxRange")
#        self._maxConFixed  = Configuration.getSetting("MaxRangeFixed")
#        self._accuracy     = Configuration.getSetting("NumberAccuracy")
#        self._numLegend    = Configuration.getSetting("NumberOfLegendBoxes")
#        self._enableLegend = Configuration.getSetting("LegendEnable")
#        self._contoursOn   = Configuration.getSetting("ContoursOn")
#        self._numberOfContourLines   = Configuration.getSetting("NumberOfContourLines")
        
    def readOutputSets(self):
        # Should I read the settings here?
        # outputDefaults
        self._updateScreen     = Configuration.getSetting("ScreenUpdateFrequency")
        self._imageOutput      = Configuration.getSetting("ImageOutputOn")
        self._shotFrequency    = Configuration.getSetting("ScreenUpdateFrequency")

    # this method is assuming these attributes are global (not per-field)
    def readVectorSets_old(self):
        print
        print MODULENAME,' >>>>>>>>>>>>>>>>>>>>>>>  readVectorSets():  doing Config-.getSetting...'
        # vectorDefaults
#        self._arrowColor   = Configuration.getSetting("ArrowColor")
#        self._arrowLength  = Configuration.getSetting("ArrowLength")
#        self._arrowColorFixed  = Configuration.getSetting("FixedArrowColorOn")
#        self._scaleArrows  = Configuration.getSetting("ScaleArrowsOn")
#        self._overlayVec   = Configuration.getSetting("OverlayVectorsOn")
        
#        self._enableLegendVec  = Configuration.getSetting("LegendEnableVector")
#        self._accuracyVec  = Configuration.getSetting("NumberAccuracyVector")
#        self._numLegendVec = Configuration.getSetting("NumberOfLegendBoxesVector")
#        self._minMag       = Configuration.getSetting("MinMagnitude")
#        self._minMagFixed  = Configuration.getSetting("MinMagnitudeFixed")
#        self._maxMag       = Configuration.getSetting("MaxMagnitude")
#        self._maxMagFixed  = Configuration.getSetting("MaxMagnitudeFixed")

    def readVisualSets(self):
        # visualDefaults
        self._cellBordersOn    = Configuration.getSetting("CellBordersOn")
        self._clusterBordersOn    = Configuration.getSetting("ClusterBordersOn")      
#        print MODULENAME,'   readVisualSets():  cellBordersOn, clusterBordersOn = ',self._cellBordersOn, self._clusterBordersOn 
        self._conLimitsOn  = Configuration.getSetting("ConcentrationLimitsOn")
        self._zoomFactor   = Configuration.getSetting("ZoomFactor")
        

    def setLatticeType(self, latticeType):
        self.latticeType = latticeType

