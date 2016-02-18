# -*- coding: utf-8 -*-
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Utilities.QVTKRenderWidget import QVTKRenderWidget
# from GraphicsNew import GraphicsNew
from MVCDrawViewBase import MVCDrawViewBase
from Plugins.ViewManagerPlugins.SimpleTabView import FIELD_TYPES
import Configuration
import vtk, math
import sys, os
import string

CONTOUR_ALLOWED_FIELD_TYPES=[FIELD_TYPES[1],FIELD_TYPES[2],FIELD_TYPES[3]]

MODULENAME='----MVCDrawView2D.py: '

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()
VTK_MINOR_VERSION=vtk.vtkVersion.GetVTKMinorVersion()


from Messaging import dbgMsg, setDebugging
# setDebugging(1)

class MVCDrawView2D(MVCDrawViewBase):
    def __init__(self, _drawModel, qvtkWidget, parent=None):
        MVCDrawViewBase.__init__(self,_drawModel,qvtkWidget, parent)        
        
        self.initArea()
        self.setParams()
        
        self.pixelizedScalarField=Configuration.getSetting("PixelizedScalarField")

    def initArea(self):
        '''
        Sets up the VTK simulation area
        :return:None
        '''
        self.actorCollection=vtk.vtkActorCollection()
        self.borderActor    = vtk.vtkActor()
        self.borderActorHex = vtk.vtkActor()
        self.clusterBorderActor    = vtk.vtkActor()
        self.clusterBorderActorHex = vtk.vtkActor()
        self.cellGlyphsActor  = vtk.vtkActor()
        self.FPPLinksActor  = vtk.vtkActor()  # used for both white and colored links
        self.outlineActor = vtk.vtkActor()
        # self.axesActor = vtk.vtkCubeAxesActor2D()
        self.axesActor = vtk.vtkCubeAxesActor()


        self.outlineDim=[0,0,0]
        
        self.cellsActor     = vtk.vtkActor()
        self.cellsActor.GetProperty().SetInterpolationToFlat() # ensures that pixels are drawn exactly not with interpolations/antialiasing
        
        self.hexCellsActor     = vtk.vtkActor()
        self.hexCellsActor.GetProperty().SetInterpolationToFlat() # ensures that pixels are drawn exactly not with interpolations/antialiasing
        
        self.conActor       = vtk.vtkActor()
        self.conActor.GetProperty().SetInterpolationToFlat()

        self.hexConActor       = vtk.vtkActor()
        self.hexConActor.GetProperty().SetInterpolationToFlat()
        
        self.contourActor   = vtk.vtkActor()      

        self.glyphsActor=vtk.vtkActor()
        #self.linksActor=vtk.vtkActor()

        # # Concentration lookup table
        
        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0,1.0)
        self.clut.SetValueRange(1.0,1.0)
        self.clut.SetAlphaRange(1.0,1.0)
        self.clut.SetNumberOfColors(1024)
        self.clut.Build()

        # Contour lookup table
        # Do I need lookup table? May be just one color?
        self.ctlut = vtk.vtkLookupTable()
        self.ctlut.SetHueRange(0.6, 0.6)
        self.ctlut.SetSaturationRange(0,1.0)
        self.ctlut.SetValueRange(1.0,1.0)
        self.ctlut.SetAlphaRange(1.0,1.0)
        self.ctlut.SetNumberOfColors(1024)
        self.ctlut.Build()
        
    def setPlane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)
#        print MODULENAME,"  got this plane ",(self.plane, self.planePos)
#        print (self.plane, self.planePos)

        
    def getPlane(self):
        return (self.plane, self.planePos)

    #----------------------------------------------------------------------------
    def showBorder(self):
        '''
        Shows cell border actor
        :return:None
        '''
        # print " SHOW BORDERS self.parentWidget.borderAct.isEnabled()=",self.parentWidget.borderAct.isEnabled()
        Configuration.setSetting("CellBordersOn",True)
#        print MODULENAME,' showBorder ============'
        if not self.currentActors.has_key("BorderActor"):
            self.currentActors["BorderActor"]=self.borderActor  
            self.graphicsFrameWidget.ren.AddActor(self.borderActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        # Don't re-render until next calc step since it could show previous/incorrect actor
        #self.Render()
        #self.graphicsFrameWidget.repaint()        
    
    def hideBorder(self):
        '''
        Hides cell border actor
        :return:None
        '''

        Configuration.setSetting("CellBordersOn",False)
        if self.currentActors.has_key("BorderActor"):
            del self.currentActors["BorderActor"] 
            self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
#            self.parentWidget.lastActiveWindow.ren.RemoveActor(self.borderActor)
        
        self.Render()
        self.graphicsFrameWidget.repaint()
        # self.parentWidget.lastActiveWindow.repaint()

    def showClusterBorder(self):
        '''
        Shows Cluster Border Actor
        :return:None
        '''
        Configuration.setSetting("ClusterBordersOn",True)
        if not self.currentActors.has_key("ClusterBorderActor"):
            self.currentActors["ClusterBorderActor"] = self.clusterBorderActor
            self.graphicsFrameWidget.ren.AddActor(self.clusterBorderActor)

    def hideClusterBorder(self):
        '''
        Hides Cluster Border Actor
        :return:None
        '''
        Configuration.setSetting("ClusterBordersOn",False)
        if self.currentActors.has_key("ClusterBorderActor"):
            del self.currentActors["ClusterBorderActor"] 
            self.graphicsFrameWidget.ren.RemoveActor(self.clusterBorderActor)
        self.Render()
        self.graphicsFrameWidget.repaint()

    #----------------------------------------------------------------------------
    def showCells(self):
        '''
        Shows Cell Type Actor
        :return:None
        '''
        if str(self.graphicsFrameWidget.fieldComboBox.currentText()) != 'Cell_Field':return

        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            if not self.currentActors.has_key("HexCellsActor"):
                self.currentActors["HexCellsActor"] = self.hexCellsActor  
                self.graphicsFrameWidget.ren.AddActor(self.hexCellsActor)
        else:
            if not self.currentActors.has_key("CellsActor"):                
                self.currentActors["CellsActor"]=self.cellsActor  
                self.graphicsFrameWidget.ren.AddActor(self.cellsActor)

        if self.currentActors.has_key("BorderActor"): # ensuring borders are the last actor added
            self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
            self.graphicsFrameWidget.ren.AddActor(self.borderActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        dbgMsg("self.currentActors.keys()=",self.currentActors.keys())

        # print
        # Don't re-render until next calc step since it could show previous/incorrect actor
        # self.Render()
        # self.graphicsFrameWidget.repaint()
        
    def hideCells(self):
        '''
        Hides Cell Type Actor
        :return:None
        '''

        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane == "XY": # drawing in other planes will be done on a rectangular lattice
            if self.currentActors.has_key("HexCellsActor"):
                del self.currentActors["HexCellsActor"]
                self.graphicsFrameWidget.ren.RemoveActor(self.hexCellsActor)
#            self.parentWidget.lastActiveWindow.ren.RemoveActor(self.hexCellsActor)
        if self.currentActors.has_key("CellsActor"):
            del self.currentActors["CellsActor"] 
            self.graphicsFrameWidget.ren.RemoveActor(self.cellsActor)
            # self.parentWidget.lastActiveWindow.ren.RemoveActor(self.cellsActor)
        self.Render()
        self.graphicsFrameWidget.repaint()
        # self.parentWidget.lastActiveWindow.repaint()

    def hideAllActors(self):   # never used?
        '''
        Hides all actors
        :return:None
        '''
        removedActors=[]
        for actorName in self.currentActors:
#            print '   hideAllActors:  removing actorName=',actorName
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actorName])
            removedActors.append(actorName)
        #cannot remove dictionary elements in  above loop
        for actorName in removedActors:
            del self.currentActors[actorName]

    #----------------------------------------------------------------------------
    def showCellGlyphs(self):
#        print MODULENAME,'  showCellGlyphs'
        Configuration.setSetting("CellGlyphsOn",True)
        if not self.currentActors.has_key("CellGlyphsActor"):
            self.currentActors["CellGlyphsActor"]=self.cellGlyphsActor  
#            print '============       MVCDrawView2D.py:  showCellGlyphs, add cellGlyphsActor'
            self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        #self.Render()
        #self.graphicsFrameWidget.repaint()        
    
    def hideCellGlyphs(self):
#        print MODULENAME,'  hideCellGlyphs'
        Configuration.setSetting("CellGlyphsOn",False)
        if self.currentActors.has_key("CellGlyphsActor"):
            del self.currentActors["CellGlyphsActor"] 
#            print '============       MVCDrawView2D.py:  hideCellGlyphs, remove cellGlyphsActor'
            self.graphicsFrameWidget.ren.RemoveActor(self.cellGlyphsActor)
        self.Render()
        self.graphicsFrameWidget.repaint()
        
    #----------------------------------------------------------------------------
    def showFPPLinks(self):
#        print MODULENAME,'   showFPPLinks'
        Configuration.setSetting("FPPLinksOn",True)
        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"] = self.FPPLinksActor  
#            print '============       MVCDrawView2D.py:  showFPPLinks, add FPPLinksActor'
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        #self.Render()
        #self.graphicsFrameWidget.repaint()        
    
    def hideFPPLinks(self):
#        print MODULENAME,'   hideFPPLinks'
        Configuration.setSetting("FPPLinksOn",False)
        if self.currentActors.has_key("FPPLinksActor"):
            del self.currentActors["FPPLinksActor"] 
#            print '============       MVCDrawView2D.py:  hideFPPLinks, remove FPPLinksActor'
            self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
        self.Render()
        self.graphicsFrameWidget.repaint()

    #----------------------------------------------------------------------------
    def showFPPLinksColor(self):
#        print MODULENAME,'   showFPPLinksColor'
        Configuration.setSetting("FPPLinksColorOn",True)
        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"] = self.FPPLinksActor  
#            print '============       MVCDrawView2D.py:  showFPPLinksColor, add FPPLinksActor'
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()
        #self.Render()
        #self.graphicsFrameWidget.repaint()        
    
    def hideFPPLinksColor(self):
#        print MODULENAME,'   hideFPPLinksColor'
        Configuration.setSetting("FPPLinksColorOn",False)
        if self.currentActors.has_key("FPPLinksActor"):
            del self.currentActors["FPPLinksActor"] 
#            print '============       MVCDrawView2D.py:  hideFPPLinksColor, remove FPPLinksActor'
            self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
        self.Render()
        self.graphicsFrameWidget.repaint()
    
    #----------------------------------------------------------------------------
    def removeContourActors(self):        
        if self.currentActors.has_key("ContourActor"):        
            del self.currentActors["ContourActor"]             
            self.graphicsFrameWidget.ren.RemoveActor(self.contourActor)
    
    def showContours(self, enable):
        if enable and self.currentFieldType[1] in CONTOUR_ALLOWED_FIELD_TYPES:
#            Configuration.setSetting("ContoursOn",True)
            if not self.currentActors.has_key("ContourActor"):
                self.currentActors["ContourActor"] = self.contourActor              
                self.graphicsFrameWidget.ren.AddActor(self.contourActor)        
            # will ensure that borders is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.contourActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.contourActor)
                self.graphicsFrameWidget.ren.AddActor(self.contourActor) 
                # print "ADDED CONTOUR ACTOR TO THE TOP"
        else:
#            Configuration.setSetting("ContoursOn",False)
            if self.currentActors.has_key("ContourActor"):
                del self.currentActors["ContourActor"]             
                self.graphicsFrameWidget.ren.RemoveActor(self.contourActor)
        # had to get rid of Render/repaint statements from this fcns because one of these is called elswhere and apparently calling them multiple times may cause software crash                    
        # self.Render() 
        # self.graphicsFrameWidget.repaint()
    
    def setBorderColor(self):  # called from drawBorders2D[Hex]  (below)
        color = Configuration.getSetting("BorderColor")
        r = color.red()
        g = color.green()
        b = color.blue()
#        print MODULENAME,'  setBorderColor():   r,g,b=',r,g,b
        self.borderActor.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
        self.borderActorHex.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
        # self.Render() 
        # self.qvtkWidget.repaint()
        
    def setClusterBorderColor(self):
        color = Configuration.getSetting("ClusterBorderColor")
        r = color.red()
        g = color.green()
        b = color.blue()
        self.clusterBorderActor.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
        self.clusterBorderActorHex.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
        # self.Render() 
        # self.qvtkWidget.repaint()
            
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
        # self.qvtkWidget.ren.ResetCameraClippingRange()
        self.graphicsFrameWidget.ren.ResetCameraClippingRange()
        self.__initDist = distance #camera.GetDistance()
        self.Render() 
        self.qvtkWidget().repaint()

    def setDim(self, fieldDim):       
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]
#        print MODULENAME,' setDim(), self.dim=',self.dim
#        print MODULENAME,' setDim(), self.dim[2]=',self.dim[2]

        
    def prepareOutlineActor(self,_dim):
        outlineData = vtk.vtkImageData()
        outlineData.SetDimensions(_dim[0], _dim[1], 1)

        outline = vtk.vtkOutlineFilter()
        
        if VTK_MAJOR_VERSION>=6:
            outline.SetInputData(outlineData)
        else:    
            outline.SetInput(outlineData)
                
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
    
        self.outlineActor.SetMapper(outlineMapper)
        self.outlineActor.GetProperty().SetColor(1, 1, 1)

    # def showOutlineActor(self, flag=True):
    #
    #     self.currentActors["Outline"]=self.outlineActor
    #     self.graphicsFrameWidget.ren.AddActor(self.outlineActor)

    def showOutlineActor(self, flag=True):
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

        # self.Render()
        # self.graphicsFrameWidget.repaint()


    def showAxes(self, flag=True):
    
        # we do not support axes for vtk less than 5.10.0
        if self.vtk_version_identifier() < self.version_identifier(5,10,0):
            return
        
        if flag:
            if not self.currentActors.has_key("Axes2D"):
                # setting camera for the actor is vrey important to get axes working properly
                self.axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
                self.currentActors["Axes2D"] = self.axesActor
                # print 'self.graphicsFrameWidget.ren.GetActiveCamera()=',self.graphicsFrameWidget.ren.GetActiveCamera()
                self.graphicsFrameWidget.ren.AddActor(self.axesActor)
            else:
                self.graphicsFrameWidget.ren.RemoveActor(self.axesActor)
                self.axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
                self.graphicsFrameWidget.ren.AddActor(self.axesActor)
        else:
            if self.currentActors.has_key("Axes2D"):
                del self.currentActors["Axes2D"]
                self.graphicsFrameWidget.ren.RemoveActor(self.axesActor)

        self.Render()
        self.graphicsFrameWidget.repaint()

    def hideOutlineActor(self):
        self.graphicsFrameWidget.ren.RemoveActor(self.outlineActor)
        del self.currentActors["Outline"]

    def drawCellField(self, _bsd, fieldType):
        '''
        Draws Cell Field
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        dictKey = self.graphicsFrameWidget.winId().__int__()  # get key (addr) for this window

        if self.parentWidget.graphicsWindowVisDict[dictKey][0] or self.getSim3DFlag():  # rwh: for multi-window bug fix;  rwh: hack for FPP
            if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
                self.drawCellFieldHex(_bsd,fieldType)
                return

            self.drawModel.initCellFieldActors((self.cellsActor,))

            if not self.currentActors.has_key("CellsActor"):
                self.currentActors["CellsActor"] = self.cellsActor
                self.graphicsFrameWidget.ren.AddActor(self.cellsActor)

        if self.parentWidget.graphicsWindowVisDict[dictKey][1]:  # rwh: for multi-window bug fix
            if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
                self.drawBorders2DHex()
            else:
                self.drawBorders2D()

        if self.parentWidget.graphicsWindowVisDict[dictKey][3] and not self.getSim3DFlag():  # rwh: for multi-window bug fix
            self.drawCellGlyphs2D()       

        if self.parentWidget.graphicsWindowVisDict[dictKey][4] and not self.getSim3DFlag():  # rwh: for multi-window bug fix
            self.drawFPPLinks2D()

        if self.parentWidget.graphicsWindowVisDict[dictKey][2]:  # rwh: for multi-window bug fix
            if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
                self.drawClusterBorders2DHex()    
            else:
                self.drawClusterBorders2D()
        
        if not Configuration.getSetting('CellsOn'):
            dbgMsg('HIDING CELLS')
            self.hideCells()
        else:
            dbgMsg('SHOWING CELLS')
            self.showCells()

        self.drawCellVisDecorations()

        self.Render()

    def drawCellVisDecorations(self):
        '''
        Draws Cell visualuzation window decorations - outline ana axes
        :return:None
        '''
        if Configuration.getSetting("BoundingBoxOn"):

            self.drawModel.prepareOutlineActors((self.outlineActor,))
            self.showOutlineActor(True)
        else:
            self.showOutlineActor(False)
            
        if self.vtk_version_identifier() >= self.version_identifier(5,10,0):
            
            if Configuration.getSetting("ShowAxes"):
                self.drawModel.prepareAxesActors((None,), (self.axesActor,))
                self.showAxes(True)
            else:
                self.showAxes(False)


    # FIXME: Draw contour lines: drawContourLines()
    def drawCellFieldHex(self, bsd, fieldType):
        '''
        Draws Cell Field For Hex Lattice
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initCellFieldHexActors((self.hexCellsActor,))
        
        if self.currentActors.has_key("CellsActor"):
            if self.ren:
                self.ren.RemoveActor(self.currentActors["CellsActor"])
            del self.currentActors["CellsActor"]        

        if not self.currentActors.has_key("HexCellsActor"):
            self.currentActors["HexCellsActor"]=self.hexCellsActor  
            self.graphicsFrameWidget.ren.AddActor(self.hexCellsActor)         
        
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()    

        if self.parentWidget.cellGlyphsAct.isChecked() and not self.getSim3DFlag():
            self.drawCellGlyphs2D()       

        if self.parentWidget.FPPLinksAct.isChecked() and not self.getSim3DFlag():
            self.drawFPPLinks2D()
            
        if self.parentWidget.clusterBorderAct.isChecked():
            self.drawClusterBorders2DHex()
        self.drawCellVisDecorations()

        self.Render()

    def drawConFieldHex(self,bsd,fieldType):
        '''
        Draws Concentration Field For Hex Lattice
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initConFieldHexActors((self.hexConActor,self.contourActor))    

        if not self.currentActors.has_key("HexConActor"):
            self.currentActors["HexConActor"]=self.hexConActor         
            self.graphicsFrameWidget.ren.AddActor(self.hexConActor)         
        
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()    
        else:
            self.hideBorder()

        # Draw legend!
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.hexConMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
        
        # print 'Configuration.getSetting("ContoursOn",%s)'%self.currentDrawingParameters.fieldName, ' = '  ,Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName)  
        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):            
            self.showContours(True)            
        else:
            self.showContours(False)

        if self.parentWidget.clusterBorderAct.isChecked():
            self.drawClusterBorders2DHex()    
        else:
            self.hideClusterBorder()

        self.drawPlotVisDecorations()

        self.Render()
        
    def drawScalarFieldCellLevelHex(self,bsd,fieldType):
        '''
        Draws Scalar Field at the Cell level - coloring entire cells -  for hex lattice
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initScalarFieldCellLevelHexActors((self.hexConActor,self.contourActor))
            
        if not self.currentActors.has_key("HexConActor"):
            self.currentActors["HexConActor"]=self.hexConActor  
            self.graphicsFrameWidget.ren.AddActor(self.hexConActor)         
        
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()    
        else:
            self.hideBorder()
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.hexConMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
    
        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):            
            self.showContours(True)            
        else:
            self.showContours(False)
        # self.showContours(True)            
            
        if self.parentWidget.clusterBorderAct.isChecked():
            self.drawClusterBorders2DHex()
        else:
            self.hideClusterBorder()

        self.drawPlotVisDecorations()

        self.Render()

    def drawScalarFieldHex(self,bsd,fieldType):
        '''
        Draws Scalar Field  - coloring individual pixels -  for hex lattice
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initScalarFieldHexActors((self.hexConActor,self.contourActor))
        
        if not self.currentActors.has_key("HexConActor"):
            self.currentActors["HexConActor"]=self.hexConActor  
            self.graphicsFrameWidget.ren.AddActor(self.hexConActor)         
        
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()    
        else:
            self.hideBorder()
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.hexConMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
    
        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):
            self.showContours(True)
        else:
            self.showContours(False)
            
        if self.parentWidget.clusterBorderAct.isChecked():
            self.drawClusterBorders2DHex()    
        else:
            self.hideClusterBorder()

        self.drawPlotVisDecorations()

        self.Render()
        
    def drawConField(self, sim, fieldType):
        '''
        Draws Concentration Field For Cartesian Lattice. Calls universal drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''

        # drawing in other planes will be done on a rectangular lattice
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY":
            self.drawConFieldHex(sim,fieldType)
            return

        # this is simply a "pointer" to function
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData2D")

        # when user requests we draw cartesian scalar field using exact pixels  not smoothed out regions as given
        # by simple vtkImageData scalar visualization.
        if self.pixelizedScalarField:
            # Perhaps there is switch in vtkImageDataGeometryFilter or related vtk object that will draw nice pixels
            # but for now we are sticking with this somewhat repetitive code
            # this is simply a "pointer" to function
            fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData2DCartesian")
            
        self.drawScalarFieldData(sim,fieldType,fillScalarField)
        
    def drawScalarFieldCellLevel(self, sim, fieldType):
        '''
        Draws Scalar Field at the Cell level - coloring entire cells -  for Cartesian lattice.
        Calls universal drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        # drawing in other planes will be done on a rectangular lattice
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY":
            self.drawScalarFieldCellLevelHex(sim,fieldType)
            return    
            
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData2D") # this is simply a "pointer" to function        
        
        if self.pixelizedScalarField:  # when user requests we draw cartesian scalar field using exact pixels  not smopothed out regions as given by sinple vtkImageData scalar visualization. 
        # Perhaps there is switch in vtkImageDataGeometryFilter or related vtk object that will draw nice pixels but for now we are sticking with this somewhat repetitive code     
            fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData2DCartesian") # this is simply a "pointer" to function        
        self.drawScalarFieldData(sim,fieldType,fillScalarField)
        
    def drawScalarField(self, sim, fieldType):
        '''
        Draws Scalar Field  - coloring individual pixels -  for Cartesian lattice. Calls universal drawScalarFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        # drawing in other planes will be done on a rectangular lattice
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane == "XY":
            self.drawScalarFieldHex(sim,fieldType)
            return

        # this is simply a "pointer" to function
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData2D")

        # when user requests we draw cartesian scalar field using exact pixels  not smopothed out regions as given
        # by sinple vtkImageData scalar visualization.
        if self.pixelizedScalarField:
            # Perhaps there is switch in vtkImageDataGeometryFilter or related vtk object that will draw nice pixels
            #  but for now we are sticking with this somewhat repetitive code
            # this is simply a "pointer" to function
            fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData2DCartesian")
            
        self.drawScalarFieldData(sim,fieldType,fillScalarField)

    def drawScalarFieldData(self, _bsd, fieldType,_fillScalarField):
        '''
        Draws Scalar Field  - universal function called by otther "drawScalar..." functions
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        # when user requests we draw cartesian scalar field using exact pixels  not smooothed out regions as given
        # by sinple vtkImageData scalar visualization.
        if self.pixelizedScalarField:
            # Perhaps there is switch in vtkImageDataGeometryFilter or related vtk object that will draw nice pixels
            # but for now we are sticking with this somewhat repetitive code
            self.drawModel.initScalarFieldCartesianActors(_fillScalarField,(self.conActor,self.contourActor,))
        else:
            dbgMsg('DRAWING SCALAR FIELD DATA')
            self.drawModel.initScalarFieldActors(_fillScalarField,(self.conActor,self.contourActor,))
        
        if not self.currentActors.has_key("ConActor"):
            self.currentActors["ConActor"]=self.conActor  
            self.graphicsFrameWidget.ren.AddActor(self.conActor) 
            
            actorProperties=vtk.vtkProperty()
            actorProperties.SetOpacity(1.0)
            
            self.conActor.SetProperty(actorProperties)
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):
            self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)

        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2D() 
        else:
            self.hideBorder()

        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):                        
            self.showContours(True)
        else:
            self.showContours(False)

        self.drawPlotVisDecorations()

        self.Render()

    def drawPlotVisDecorations(self):
        '''
        Draws Plot Window Decorations such as outline or axes
        :return:None
        '''
        # if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):
        #     self.drawModel.prepareLegendActors((self.drawModel.conMapper,),(self.legendActor,))
        #     self.showLegend(True)
        # else:
        #     self.showLegend(False)

        if Configuration.getSetting("BoundingBoxOn"):
            self.drawModel.prepareOutlineActors((self.outlineActor,))
            self.showOutlineActor()
        if self.vtk_version_identifier() >= self.version_identifier(5,10,0):

            if Configuration.getSetting("ShowPlotAxes", self.currentDrawingParameters.fieldName):
                self.drawModel.prepareAxesActors((None,),(self.axesActor,))
                self.showAxes(True)
            else:
                self.showAxes(False)

    def drawVectorField(self, bsd, fieldType):
        '''
        Draws Vector Field (vectors are at arbitrary positions) for cartesian lattice
        - calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.removeContourActors()
        # drawing in other planes will be done on a rectangular lattice
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane == "XY":
            self.drawVectorFieldDataHex(bsd,fieldType)
            return

        # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData2D
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldData2D")
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)
        
    def drawVectorFieldCellLevel(self, bsd, fieldType):
        '''
        Draws Vector Field (vectors are associated with cells) for cartesian lattice
        - calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''

        self.removeContourActors()
        
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawVectorFieldCellLevelDataHex(bsd,fieldType)
            return            
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldCellLevelData2D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData2D        
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)

    def drawVectorFieldDataHex(self,bsd,fieldType):
        '''
        Draws Vector Field (vectors are at arbitrary) for hex lattice
        - calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.removeContourActors()
        
        self.drawModel.initVectorFieldDataHexActors((self.glyphsActor,))
        
        if not self.currentActors.has_key("Glyphs2DActor"):
            self.currentActors["Glyphs2DActor"]=self.glyphsActor  
            self.graphicsFrameWidget.ren.AddActor(self.glyphsActor)         
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.glyphsMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
            
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()         
        else:
            self.hideBorder()


        self.drawPlotVisDecorations()

        self.Render()
        
        
    def drawVectorFieldCellLevelDataHex(self,bsd,fieldType):
        '''
        Draws Vector Field (vectors are associated with cells) for hex lattice
        - calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.removeContourActors()
        
        self.drawModel.initVectorFieldCellLevelDataHexActors((self.glyphsActor,))
        
        if not self.currentActors.has_key("Glyphs2DActor"):
            self.currentActors["Glyphs2DActor"]=self.glyphsActor  
            self.graphicsFrameWidget.ren.AddActor(self.glyphsActor)         
            
        if Configuration.getSetting("LegendEnable",self.currentDrawingParameters.fieldName):            
            self.drawModel.prepareLegendActors((self.drawModel.glyphsMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
            
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2DHex()         
        else:
            self.hideBorder()

        self.drawPlotVisDecorations()
           
        self.Render()


        
    def drawVectorFieldData(self,bsd,fieldType,_fillVectorFieldFcn):
        '''
        Draws Vector Field for cartesian lattice - universal function called by other "drawVector..." functions
        - calls universal function drawVectorFieldData
        :param bsd: BasicSimulationData - contains field dim etc
        :param fieldType: field type - e.g. cellfield, concentration field etc...
        :return:None
        '''
        self.drawModel.initVectorFieldCellLevelActors(_fillVectorFieldFcn, (self.glyphsActor,))
        
        if Configuration.getSetting("OverlayVectorsOn",self.currentDrawingParameters.fieldName):
            self.drawCellField(bsd, fieldType)
        else:
            self.hideCells()
        
        if not self.currentActors.has_key("Glyphs2DActor"):
            self.currentActors["Glyphs2DActor"]=self.glyphsActor  
            self.graphicsFrameWidget.ren.AddActor(self.glyphsActor)         

        fieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME, '------ drawVectorFieldData:   fieldName =',fieldName
        legendEnabled = Configuration.getSetting("LegendEnable",fieldName)
#        print MODULENAME, '------ drawVectorFieldData:   legendEnabled =',legendEnabled
#        foorwh=1/0
        if legendEnabled:
            self.drawModel.prepareLegendActors((self.drawModel.glyphsMapper,),(self.legendActor,))
            self.showLegend(True)
        else:
            self.showLegend(False)
            
        if self.parentWidget.borderAct.isChecked():
            self.drawBorders2D()         
        else:
            self.hideBorder()

        self.drawPlotVisDecorations()

        self.Render()
        
    # Optimize code?
    def dimOrder(self, plane):
        plane=string.lower(plane)
        order = (0, 1, 2)
        if plane == "xy":
            order = (0, 1, 2)
        elif plane == "xz":
            order = (0, 2, 1)
        elif plane == "yz": 
            order = (1, 2, 0)
            
        return order

    # Optimize code?
    def pointOrder(self, plane):
        plane=string.lower(plane)
        order = (0, 1, 2)
        if plane == "xy":
            order = (0, 1, 2)
        elif plane == "xz":
            order = (0, 2, 1)
        elif plane == "yz": 
            order = (2, 0, 1)
        return order

    def planeMapper(self, order, tuple):
        return [tuple[order[0]], tuple[order[1]], tuple[order[2]]]
    
    # ?
    def drawContourLines(self): 
        pass        
        
    def wheelEvent(self, ev): 
        print "wheelEvent \n\n\n\n"
        self.__zoomStep(ev.delta())
    
    # Overrides the mousePressEvent() method from QVTKRenderWidget
    def mousePressEvent(self,ev):
        if (ev.button() == 1):
            self._Mode = "Pan"
            self._ActiveButton = ev.button()
            #self.PickActor(ev.x(), ev.y())
            #print self.GetPicker()
            #self.showTip(ev.x(), ev.y())
            
        elif (ev.button() == 2):
            self._Mode = "Zoom"
            self._ActiveButton = ev.button()

        self.UpdateRenderer(ev.x(),ev.y())

    def event(self, ev):
        if ev.type() == QEvent.ToolTip:
            self.showTip(ev)
        return QWidget.event(self, ev)

    def showTip(self, ev):
        # toll tips are not enabled in this release
        return
        import CompuCell
        pt = CompuCell.Point3D() 

        self.PickActor(ev.x(), ev.y())
        id = self.GetPicker().GetCellId()
        if id != -1:
            pos = self.GetPicker().GetPickPosition()
            pt.x, pt.y, pt.z = int(pos[0]), int(pos[1]), 0 
            
            if  self.cellField.get(pt) is not None and self.cellField.get(pt).id != 0:
                QToolTip.hideText()
                QToolTip.showText(ev.globalPos(), self.toolTip(self.cellField.get(pt)))
                    
    def takeShot(self):
        filter = "PNG files (*.png)"
        fileName = QFileDialog.getSaveFileName(\
            self,
            "Save Screenshot",
            os.getcwd(), 
            filter
            )

        # Other way to get the correct file name: fileName.toAscii().data())
#        print MODULENAME,'  takeShot:  fileName = ',fileName
        if fileName is not None and fileName != "":
            self.takeSimShot(str(fileName))
    
    # fileName - full file name (e.g. "/home/user/shot.png")        
    def takeSimShot(self, fileName):
        # print MODULENAME,' takeSimShot:  fileName=',fileName

        # DON'T REMOVE!
        # Better quality
        # Passes vtkRenderer. Takes actual screenshot of the region within the widget window
        # If other application are present within this region it will shoot them also
        
        
        renderLarge = vtk.vtkRenderLargeImage()
        renderLarge.SetInput(self.graphicsFrameWidget.ren)
        renderLarge.SetMagnification(1)

        
        # We write out the image which causes the rendering to occur. If you
        # watch your screen you might see the pieces being rendered right
        # after one another.
        # writer = vtk.vtkPNGWriter()
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(renderLarge.GetOutputPort())
        print MODULENAME,"takeSimShot():  vtkPNGWriter, fileName=",fileName
        
        writer.SetFileName(fileName)
        print 'TRYING TO WRITE ',fileName
        writer.Write()
        print 'WROTE ',fileName
        
            
    def toolTip(self, cellG):
        return "Id:             %s\nType:       %s\nVolume:  %s" % (cellG.id, cellG.type, cellG.volume)

    def configsChanged(self):   # this method is invoked; rf. SIGNAL in SimpleTabView.py
#        print MODULENAME,'  configsChanged()'
        self.populateLookupTable()
        self.setBorderColor()
        # # Doesn't work, gives error: 
        # # vtkScalarBarActor (0x8854218): Need a mapper to render a scalar bar
        # #self.showLegend(Configuration.getSetting("LegendEnable"))
#        print MODULENAME,'  configsChanged():   type(self.currentDrawingParameters.fieldName)=',type(self.currentDrawingParameters.fieldName)
#        print MODULENAME,'  configsChanged():   self.currentDrawingParameters.fieldName=',self.currentDrawingParameters.fieldName

#        self.showContours(Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName))
        self.showContours(True)
        self.parentWidget.requestRedraw()

    # these drawBorders* fns called from drawCellField[Hex] (above)
    def drawBorders2D(self):
#        print '============ MVCDrawView2D.py: drawBorders2D ============='
        self.setBorderColor()
        self.drawModel.initBordersActors2D((self.borderActor,))

        if not self.currentActors.has_key("BorderActor"):
            self.currentActors["BorderActor"]=self.borderActor
            self.graphicsFrameWidget.ren.AddActor(self.borderActor)
            # print "ADDING BORDER ACTOR"
        else:
            # will ensure that borders is the last item to draw
            actorsCollection = self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.borderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
                self.graphicsFrameWidget.ren.AddActor(self.borderActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys()    
        
    def drawBorders2DHex(self):
        self.setBorderColor() 
        self.drawModel.initBordersActors2DHex((self.borderActor,))
        if not self.currentActors.has_key("BorderActor"):
            self.currentActors["BorderActor"]=self.borderActor
            self.graphicsFrameWidget.ren.AddActor(self.borderActor) 
        else:
            # will ensure that borders is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem()!=self.borderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
                self.graphicsFrameWidget.ren.AddActor(self.borderActor)
                
    def drawClusterBorders2D(self):
#        print MODULENAME,'  drawClusterBorders2D ============='
#        self.clusterBorderActor.GetProperty().SetColor(1.0,1.0,1.0)
        self.setClusterBorderColor() 
#        print MODULENAME,'    calling initClusterBordersActor2D...'     
        self.drawModel.initClusterBordersActors2D((self.clusterBorderActor))
#        print MODULENAME,'    back from initClusterBordersActor2D' 

#        print MODULENAME,"   adding cluster border actor; self.currentActors.keys()=",self.currentActors.keys()
        if not self.currentActors.has_key("ClusterBorderActor"):
            self.currentActors["ClusterBorderActor"] = self.clusterBorderActor
            self.graphicsFrameWidget.ren.AddActor(self.clusterBorderActor)
#            print "ADDING cluster BORDER ACTOR"
        else:
            # will ensure that borders is the last item to draw
            actorsCollection = self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.clusterBorderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.clusterBorderActor)
                self.graphicsFrameWidget.ren.AddActor(self.clusterBorderActor) 
    
    def drawClusterBorders2DHex(self):
#        print MODULENAME,'  drawClusterBorders2DHex ============='
#        self.clusterBorderActor.GetProperty().SetColor(1.0,1.0,1.0)
        self.setClusterBorderColor() 
#        print MODULENAME,'    calling initClusterBordersActor2D...'     
        self.drawModel.initClusterBordersActors2DHex((self.clusterBorderActor))
#        print MODULENAME,'    back from initClusterBordersActor2D' 

#        print MODULENAME,"   adding cluster border actor; self.currentActors.keys()=",self.currentActors.keys()
        if not self.currentActors.has_key("ClusterBorderActor"):
            self.currentActors["ClusterBorderActor"] = self.clusterBorderActor
            self.graphicsFrameWidget.ren.AddActor(self.clusterBorderActor)
#            print "ADDING cluster BORDER ACTOR"
        else:
            # will ensure that borders is the last item to draw
            actorsCollection = self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.clusterBorderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.clusterBorderActor)
                self.graphicsFrameWidget.ren.AddActor(self.clusterBorderActor)
                
    def drawCellGlyphs2D(self):
#        print MODULENAME,' drawCellGlyphs2D()'
        #self.setBorderColor()         
        self.drawModel.initCellGlyphsActor2D(self.cellGlyphsActor)

        if not self.currentActors.has_key("CellGlyphsActor"):
            self.currentActors["CellGlyphsActor"] = self.cellGlyphsActor
            self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor)
            # print "ADDING cellGlyphs ACTOR"
        else:
            # will ensure that borders is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.borderActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.cellGlyphsActor)
                self.graphicsFrameWidget.ren.AddActor(self.cellGlyphsActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys()    

    def drawFPPLinks2D(self):
#        print MODULENAME,' drawFPPLinks2D()'
        #self.setBorderColor()         
        self.drawModel.initFPPLinksActor2D(self.FPPLinksActor)

        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"] = self.FPPLinksActor
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
            # print "ADDING FPPLinks ACTOR"
        else:
            # will ensure that links actor is the last item to draw
            actorsCollection=self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.FPPLinksActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
                self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys()   
        
    def drawFPPLinksColor2D(self):
#        print MODULENAME,' drawFPPLinksColor2D()'
        #self.setBorderColor()         
        self.drawModel.initFPPLinksColorActor2D(self.FPPLinksActor)

        if not self.currentActors.has_key("FPPLinksActor"):
            self.currentActors["FPPLinksActor"] = self.FPPLinksActor
            self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor)
            # print "ADDING FPPLinks ACTOR"
        else:
            # will ensure that links actor is the last item to draw
            actorsCollection = self.graphicsFrameWidget.ren.GetActors()
            if actorsCollection.GetLastItem() != self.FPPLinksActor:
                self.graphicsFrameWidget.ren.RemoveActor(self.FPPLinksActor)
                self.graphicsFrameWidget.ren.AddActor(self.FPPLinksActor) 
        # print "self.currentActors.keys()=",self.currentActors.keys()  
