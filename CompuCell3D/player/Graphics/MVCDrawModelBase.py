import sys, os
import vtk
VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()

MODULENAME='----- MVCDrawModelBase.py: '


def setVTKPaths():
   import sys
   from os import environ
   import string
   import sys
   platform=sys.platform
   if platform=='win32':
      sys.path.insert(0,environ["PYTHON_DEPS_PATH"])
      # sys.path.append(environ["VTKPATH"])
   
      # sys.path.append(environ["VTKPATH"])
      # sys.path.append(environ["VTKPATH1"])
      # sys.path.append(environ["PYQT_PATH"])
      # sys.path.append(environ["SIP_PATH"])
      # sys.path.append(environ["SIP_UTILS_PATH"])
#   else:
#      swig_path_list=string.split(environ["VTKPATH"])
#      for swig_path in swig_path_list:
#         sys.path.append(swig_path)


setVTKPaths()
# print "GRAPHICS PATH=",sys.path  


from PyQt4.QtCore import *
from PyQt4.QtGui import *

# from Utilities.QVTKRenderWidget import QVTKRenderWidget
# from FrameQVTK import FrameQVTK
# import FrameQVTK
import Graphics

from PyQt4 import QtCore, QtGui,QtOpenGL
import vtk

import Configuration
import vtk, math
#import sys, os
import string

from Plugins.ViewManagerPlugins.SimpleTabView import FIELD_TYPES,PLANES
        
class MVCDrawModelBase:
    def __init__(self, graphicsFrameWidget, parent=None):
        
        (self.minCon, self.maxCon) = (0, 0)
        
        self.parentWidget=parent
        
        
        # self.graphicsFrameWidget = graphicsFrameWidget()        
        # self.qvtkWidget = self.graphicsFrameWidget.qvtkWidget
        
        from weakref import ref
        gfw=ref(graphicsFrameWidget)
        self.graphicsFrameWidget = gfw()
        
        # qvtk=ref(self.graphicsFrameWidget.qvtkWidget)
        
        self.qvtkWidget =ref(self.graphicsFrameWidget.qvtkWidget)       
        # self.qvtkWidget = qvtk()        
        
        # # # self.graphicsFrameWidget=graphicsFrameWidget
        # # # self.qvtkWidget=self.graphicsFrameWidget.qvtkWidget
        self.currentDrawingFunction=None       
        self.fieldTypes=None 
        self.currentDrawingParameters=None
#        self.scaleGlyphsByVolume = False
        
        self.hexFlag = self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"]
#        print MODULENAME,' __init__:   self.hexFlag=', self.hexFlag
        
        # should also set "periodic" boundary condition flag(s) (e.g. for drawing FPP links that wraparound)
        
    def setDrawingParametersObject(self,_drawingParams):
        self.currentDrawingParameters=_drawingParams
        
    def setDrawingParameters(self,_bsd,_plane,_planePos,_fieldType):   
        self.bsd=_bsd
        self.plane=_plane
        self.planePos=_planePos
        self.fieldtype=_fieldType
        
    def setDrawingFunctionName(self,_fcnName):
        # print "\n\n\n THIS IS _fcnName=",_fcnName," self.drawingFcnName=",self.drawingFcnName
        
        if self.drawingFcnName != _fcnName:
            self.drawingFcnHasChanged=True
        else:
            self.drawingFcnHasChanged=False
        self.drawingFcnName=_fcnName
        
    def clearDisplay(self):
        print MODULENAME,"     clearDisplay() "
        for actor in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actor])
            
        self.currentActors.clear()
    
    def Render(self):   # never called?!
#        print MODULENAME,"     --------- Render() "
        self.graphicsFrameWidget.Render()
        
    #this is an ugly solution that seems to work on 32 bit machines. We will see if it will work on other machines        
    def extractAddressIntFromVtkObject(self,_vtkObj):
        # pointer_ia=ia.__this__
        # print "pointer_ia=",pointer_ia
        # address=pointer_ia[1:9]
        # print "address=",address," int(address)=",int(address,16)
        return self.parentWidget.fieldExtractor.unmangleSWIGVktPtrAsLong(_vtkObj.__this__)
        # return int(_vtkObj.__this__[1:9],16)
                
        
    def initCellFieldActors(self, _actors): pass
    
    def initConFieldActors(self, _actors): pass
    
    def initVectorFieldCellLevelActors(self, _fillVectorFieldFcn, _actors): pass
    
    def initVectorFieldActors(self, _actors): pass
    
    def initScalarFieldCellLevelActors(self, _actors): pass     
    
    def initScalarFieldActors(self, _fillScalarField, _actors): pass   
    
    def prepareOutlineActors(self,_actors):pass        
    
    def setContourColor(self):
#        foo=1/0
        color = Configuration.getSetting("ContourColor")
        r = color.red()
        g = color.green()
        b = color.blue()
#        print MODULENAME,'  setBorderColor():   r,g,b=',r,g,b
        self.contourActor.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
#        self.contourActorHex.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))

    def getIsoValues(self,conFieldName):
        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)
#        print MODULENAME, '  type(self.isovalStr)=',type(self.isovalStr)
#        print MODULENAME, '  self.isovalStr=',self.isovalStr
        if type(self.isovalStr) == QVariant:
#          isovalStr = isovalStr.toString()
#          print MODULENAME, ' self.isovalStr.toList()=',self.isovalStr.toList()
#          print MODULENAME, ' self.isovalStr.toString()=',self.isovalStr.toString()
          self.isovalStr = str(self.isovalStr.toString())
#          print MODULENAME, ' new type(self.isovalStr)=',type(self.isovalStr)
#        elif type(self.isovalStr) == QString:
        else:
          self.isovalStr = str(self.isovalStr)


#        print MODULENAME, '  pre-replace,split; initScalarFieldDataActors(): self.isovalStr=',self.isovalStr
#        import string
        self.isovalStr = string.replace(self.isovalStr,","," ")
        self.isovalStr = string.split(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final type(self.isovalStr)=',type(self.isovalStr)
#        print MODULENAME, '  initScalarFieldDataActors(): final self.isovalStr=',self.isovalStr

#        print MODULENAME, '  initScalarFieldDataActors(): len(self.isovalStr)=',len(self.isovalStr)
        printIsoValues = False
#        if printIsoValues:  print MODULENAME, ' isovalues= ',
        isoNum = 0
        self.isoValList = []
        for idx in xrange(len(self.isovalStr)):
#            print MODULENAME, '  initScalarFieldDataActors(): idx= ',idx
            try:
                isoVal = float(self.isovalStr[idx])
                if printIsoValues:  print MODULENAME, '  initScalarFieldDataActors(): setting (specific) isoval= ',isoVal
                self.isoValList.append(isoVal)
#                isoContour.SetValue(isoNum, isoVal)
                isoNum += 1
            except:
                print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ',self.isovalStr[idx]
                
#        return [1.1,2.2,3.3]
#        print MODULENAME, '  returning self.isoValList=',self.isoValList
        return self.isoValList
    
    # def showContours(self, enable): pass
    
    # def setPlane(self, plane, pos): pass
    
    # def getPlane(self):
        # return ("",0)
    
    def getCamera(self):
        return self.ren.GetActiveCamera()
        
    # def initSimArea(self, _bsd):
        # fieldDim   = _bsd.fieldDim
        # # sim.getPotts().getCellFieldG().getDim()
        # self.setCamera(fieldDim)
       
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
    
    def setParams(self):
        # You can use either Build() method (256 color by default) or
        # SetNumberOfTableValues() to allocate much more colors!
        self.celltypeLUT = vtk.vtkLookupTable()
        # You need to explicitly call Build() when constructing the LUT by hand     
        self.celltypeLUT.Build()
        self.populateLookupTable()
        # self.dim = [100, 100, 1] # Default values
        
        # for FPP links (and offset also for cell glyphs)
        self.eps = 1.e-4     # not sure how small this should be (checking to see if cell volume -> 0)
        self.stubSize = 3.0  # dangling line stub size for lines that wraparound periodic BCs
#        self.offset = 1.0    # account for fact that COM of cell is offset from visualized lattice
#        self.offset = 0.0    # account for fact that COM of cell is offset from visualized lattice

        # scaling factors to map square lattice to hex lattice (rf. CC3D Manual)
        self.xScaleHex = 1.0
        self.yScaleHex =  0.866
        self.zScaleHex =  0.816
        
        self.lutBlueRed = vtk.vtkLookupTable()
        self.lutBlueRed.SetHueRange(0.667,0.0)
        self.lutBlueRed.Build()
    
    def populateLookupTable(self):
#        print MODULENAME,' populateLookupTable()'
        colorMap = Configuration.getSetting("TypeColorMap")
#        print MODULENAME,' populateLookupTable():  len(colorMap)=',len(colorMap)
        self.celltypeLUT.SetNumberOfTableValues(len(colorMap))
        self.celltypeLUT.SetNumberOfColors(len(colorMap))
#        lutGlyph.SetTableValue(5, 1,0,0, 1.0)     # SetTableValue (vtkIdType indx, double r, double g, double b, double a=1.0)
#        lutGlyph.SetTableValue(8, 0,1,1, 1.0)     # SetTableValue (vtkIdType indx, double r, double g, double b, double a=1.0)
        for key in colorMap.keys():
            r = colorMap[key].red()
            g = colorMap[key].green()
            b = colorMap[key].blue()
            self.celltypeLUT.SetTableValue(key, self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b), 1.0)
#            print "       type=",key," red=",r," green=",g," blue=",b
#            print "       type=",key," (VTK) red=",self.toVTKColor(r)," green=",self.toVTKColor(g)," blue=",self.toVTKColor(b)
        # self.qvtkWidget.repaint()
        self.celltypeLUT.Build()
        self.celltypeLUTMax = self.celltypeLUT.GetNumberOfTableValues() - 1   # cell types = [0,max]
        self.celltypeLUT.SetTableRange(0,self.celltypeLUTMax)
#        print "       celltypeLUTMax=",self.celltypeLUTMax
        
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
        return (self.minCon, self.maxCon)
    
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

    def prepareAxesActors(self, _mappers, _actors):
        pass

    def prepareLegendActors(self, _mappers, _actors):
        legendActor=_actors[0]
        mapper=_mappers[0]
            
        legendActor.SetLookupTable(mapper.GetLookupTable())    
        legendActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legendActor.GetPositionCoordinate().SetValue(0.01, 0.1)
        legendActor.SetOrientationToHorizontal()
        
        legendActor.SetOrientationToVertical()
        # self.legendActor.SetWidth(0.8)
        # self.legendActor.SetHeight(0.10)

        legendActor.SetWidth(0.1)
        legendActor.SetHeight(0.9)
        
        if VTK_MAJOR_VERSION>=6:
            legendActor.SetTitle('')

        # You don't actually need to make contrast for the text as
        # it has shadow!
        text_property = legendActor.GetLabelTextProperty()
        text_property.SetFontSize(12) # For some reason it doesn't make effect
        # text.BoldOff()
        text_property.SetColor(1.0, 1.0, 1.0)

        legendActor.SetLabelTextProperty(text_property)


    # Break the settings read into groups?
#    def readSettings_old(self):   # not ever called?!  (rf. MVCDrawViewBase)
#        self.readColorsSets()
#        self.readViewSets()
#        self.readColormapSets()
#        self.readOutputSets()
#        self.readVectorSets()
#        self.readVisualSets()
        # simDefaults?

    def readColorsSets(self):
        #colorsDefaults
        self._colorMap     = Configuration.getSetting("TypeColorMap")
        self._borderColor  = Configuration.getSetting("BorderColor")
        self._contourColor = Configuration.getSetting("ContourColor")
        self._brushColor   = Configuration.getSetting("BrushColor")
        self._penColor     = Configuration.getSetting("PenColor")

    def readViewSets(self):
        # For 3D only?
        # viewDefaults
        self._types3D      = Configuration.getSetting("Types3DInvisible")

#    def readColormapSets(self):   # don't think this is ever called
#        print MODULENAME,' readColormapSets():  doing Config-.getSetting...'
#        # colormapDefaults
#        self._minCon       = Configuration.getSetting("minRange")
#        self._minConFixed  = Configuration.getSetting("minRangeFixed")
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
        self._shotFrequency    = Configuration.getSetting("ScreenshotFrequency")

    def readVectorSets(self):
        # vectorDefaults
        self._arrowColor   = Configuration.getSetting("ArrowColor")
        self._arrowLength  = Configuration.getSetting("ArrowLength")
        self._arrowColorFixed  = Configuration.getSetting("FixedArrowColorOn")
        self._enableLegendVec  = Configuration.getSetting("LegendEnableVector")
        self._scaleArrows  = Configuration.getSetting("ScaleArrowsOn")
        self._accuracyVec  = Configuration.getSetting("NumberAccuracyVector")
        self._numLegendVec = Configuration.getSetting("NumberOfLegendBoxesVector")
        self._overlayVec   = Configuration.getSetting("OverlayVectorsOn")
        self._maxMag       = Configuration.getSetting("MaxMagnitude")
        self._maxMagFixed  = Configuration.getSetting("MaxMagnitudeFixed")
        self._minMag       = Configuration.getSetting("MinMagnitude")
        self._minMagFixed  = Configuration.getSetting("MinMagnitudeFixed")

    def readVisualSets(self):
        # visualDefaults
        self._cellBordersOn    = Configuration.getSetting("CellBordersOn")
        self._clusterBordersOn = Configuration.getSetting("ClusterBordersOn")       
        self._conLimitsOn  = Configuration.getSetting("ConcentrationLimitsOn")
        self._zoomFactor   = Configuration.getSetting("ZoomFactor")

    def setLatticeType(self, latticeType):
        self.latticeType=latticeType

    
