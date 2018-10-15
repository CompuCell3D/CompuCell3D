import sys
import os
import string
import Configuration

# def setVTKPaths():
   # import sys
   # from os import environ
   # import string
   # import sys
   # platform=sys.platform
   # if platform=='win32':
      # sys.path.append(environ["VTKPATH"])
      # sys.path.append(environ["VTKPATH1"])
      # sys.path.append(environ["PYQT_PATH"])
      # sys.path.append(environ["SIP_PATH"])
      # sys.path.append(environ["SIP_UTILS_PATH"])
# #   else:
# #      swig_path_list=string.split(environ["VTKPATH"])
# #      for swig_path in swig_path_list:
# #         sys.path.append(swig_path)

# # print "PATH=",sys.path
# setVTKPaths()
# # print "PATH=",sys.path  


# from PyQt4 import QtCore, QtGui,QtOpenGL
from PyQt5 import QtCore, QtGui,QtOpenGL, QtWidgets
import vtk
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from enums import *

from GraphicsOffScreen import GenericDrawer
from Utilities import ScreenshotData
from Utilities import qcolor_to_rgba, cs_string_to_typed_list

import sys
platform=sys.platform
if platform=='darwin':    
    from Utilities.QVTKRenderWindowInteractor_mac import QVTKRenderWindowInteractor
#     from Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
else:    
    from Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# from Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor    

from MVCDrawView2D import MVCDrawView2D
from MVCDrawModel2D import MVCDrawModel2D

from MVCDrawView3D import MVCDrawView3D
from MVCDrawModel3D import MVCDrawModel3D

MODULENAME = '---- GraphicsFrameWidget.py: '

from weakref import ref

# class GraphicsFrameWidget(QtGui.QFrame):
class GraphicsFrameWidget(QtWidgets.QFrame):
    # def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
    def __init__(self, parent=None, originatingWidget=None):

        QtWidgets.QFrame.__init__(self, parent)
        
        
        # print '\n\n\n\n\n CREATING NEW GRAPHICS FRAME WIDGET ',self
        
        
        # self.allowSaveLayout = True
        self.is_screenshot_widget = False
        self.qvtkWidget = QVTKRenderWindowInteractor(self)   # a QWidget
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        
        # MDIFIX
        self.parentWidget = originatingWidget
        # self.parentWidget = parent
        
        
        self.plane = None
        self.planePos = None

        self.lineEdit = QtWidgets.QLineEdit()
        
        self.__initCrossSectionActions()
        self.cstb = self.initCrossSectionToolbar()        
        
        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        layout.addWidget(self.cstb)
        layout.addWidget(self.qvtkWidget)
        self.setLayout(layout)
        self.setMinimumSize(100, 100) #needs to be defined to resize smaller than 400x400
        self.resize(600, 600)
        
        self.qvtkWidget.Initialize()
        self.qvtkWidget.Start()

        # todo 5 - adding generic drawer

        self.gd = GenericDrawer()
        self.gd.set_interactive_camera_flag(True)
        self.current_screenshot_data = None


        self.renWin = self.qvtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.gd.get_renderer())

        # todo 5 - ok previous code
        # self.ren = vtk.vtkRenderer()
        # self.renWin = self.qvtkWidget.GetRenderWindow()
        # self.renWin.AddRenderer(self.ren)




#        print MODULENAME,"GraphicsFrameWidget():__init__:   parent=",parent

        #    some objects below create cyclic dependencies - the widget will not free its memory unless  in the close event e.g.  self.drawModel2D gets set to None
        # # # from weakref import ref
        
        # # # self_weakref=ref(self)
        # # # self.drawModel2D = MVCDrawModel2D(self_weakref,parent)
        # # # self.draw2D = MVCDrawView2D(self.drawModel2D,self_weakref,parent)
        
        # # # self.drawModel3D = MVCDrawModel3D(self_weakref,parent)
        # # # self.draw3D = MVCDrawView3D(self.drawModel3D,self_weakref,parent)
        
        # todo 5
        # # MDIFIX
        # self.drawModel2D = MVCDrawModel2D(self, self.parentWidget)
        # self.draw2D = MVCDrawView2D(self.drawModel2D, self, self.parentWidget)
        #
        # self.drawModel3D = MVCDrawModel3D(self, self.parentWidget)
        # self.draw3D = MVCDrawView3D(self.drawModel3D, self, self.parentWidget)




        # self.drawModel2D = MVCDrawModel2D(self,parent)
        # self.draw2D = MVCDrawView2D(self.drawModel2D,self,parent)
        #
        # self.drawModel3D = MVCDrawModel3D(self,parent)
        # self.draw3D = MVCDrawView3D(self.drawModel3D,self,parent)





#         # self.draw2D=Draw2D(self,parent)
#         # self.draw3D=Draw3D(self,parent)
#         self.camera3D = self.ren.MakeCamera()
#         self.camera2D = self.ren.GetActiveCamera()
#         self.ren.SetActiveCamera(self.camera2D)
#
#
#         self.currentDrawingObject = self.draw2D
#
#
#         self.draw3DFlag = False
#         self.usedDraw3DFlag = False
#         # self.getattrFcn=self.getattrDraw2D
#
#         # rwh test
# #        print MODULENAME,' self.parentWidget.playerSettingsFileName= ',self.parentWidget.playerSettingsFileName
#         if self.parentWidget.playerSettingsFileName:
#             # does not work on Windows with Python 2.5
#             # with open(self.parentWidget.playerSettingsFileName, 'r') as f:
#             f=None
#             try:
#                 f=open(self.parentWidget.playerSettingsFileName, 'r')
#                 while True:
#                     l = f.readline()
#                     if l == "": break
#                     v = l.split()
#                     print v
#                     if string.find(v[0], 'CameraPosition') > -1:
#                         self.camera3D.SetPosition(float(v[1]), float(v[2]), float(v[3]))
#                     elif string.find(v[0], 'CameraFocalPoint') > -1:
#                         self.camera3D.SetFocalPoint(float(v[1]), float(v[2]), float(v[3]))
#                     elif string.find(v[0], 'CameraViewUp') > -1:
#                         self.camera3D.SetViewUp(float(v[1]), float(v[2]), float(v[3]))
# #                    elif string.find(v[0], 'ViewPlaneNormal') > -1:   # deprecated
# #                        self.camera3D.SetViewPlaneNormal(float(v[1]), float(v[2]), float(v[3]))
#                     elif string.find(v[0], 'CameraClippingRange') > -1:
#                         self.camera3D.SetClippingRange(float(v[1]), float(v[2]))
#                     elif string.find(v[0], 'CameraDistance') > -1:
#                         print 'SetDistance = ',float(v[1])
#                         self.camera3D.SetDistance(float(v[1]))
#                     elif string.find(v[0], 'ViewAngle') > -1:
#                         self.camera3D.SetViewAngle(float(v[1]))
#
#             except IOError:
#                 pass
#
#         self.screenshotWindowFlag=False
#             # # # self.setDrawingStyle("3D")
#     #        self.currentDrawingObject = self.draw3D
#     #        self.draw3DFlag = True
#     #        self.usedDraw3DFlag = False
#     #        self.ren.SetActiveCamera(self.camera3D)
# #            self.camera3D.SetPosition(100,100,100)
# #            self.qvtkWidget.SetSize(200,200)
#

        self.metadata_fetcher_dict = {
            'CellField': self.get_cell_field_metadata,
            'ConField':self.get_con_field_metadata,
            'ScalarField':self.get_con_field_metadata,
            'ScalarFieldCellLevel': self.get_con_field_metadata,
            'VectorField': self.get_con_field_metadata,
            'VectorFieldCellLevel': self.get_vector_field_metadata,
        }

    def get_metadata(self, field_name, field_type):
        try:
            metadata_fetcher_fcn = self.metadata_fetcher_dict[field_type]
        except KeyError:
            return {}

        metadata = metadata_fetcher_fcn(field_name=field_name, field_type=field_type)

        return metadata

    def get_cell_field_metadata(self, field_name, field_type):
        metadata_dict = self.get_color_metadata(field_name=field_name, field_type=field_type)
        return metadata_dict

    def get_color_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render borders
        :param field_name:{str} field_name
        :return: {dict}
        """
        metadata_dict = {}
        metadata_dict['BorderColor'] = qcolor_to_rgba(Configuration.getSetting('BorderColor'))
        metadata_dict['ClusterBorderColor'] = qcolor_to_rgba(Configuration.getSetting('ClusterBorderColor'))
        metadata_dict['BoundingBoxColor'] = qcolor_to_rgba(Configuration.getSetting('BoundingBoxColor'))
        metadata_dict['AxesColor'] = qcolor_to_rgba(Configuration.getSetting('AxesColor'))
        metadata_dict['ContourColor'] = qcolor_to_rgba(Configuration.getSetting('ContourColor'))
        metadata_dict['WindowColor'] = qcolor_to_rgba(Configuration.getSetting('WindowColor'))
        # todo - fix color of fpp links
        metadata_dict['FppLinksColor'] = qcolor_to_rgba(Configuration.getSetting('ContourColor'))

        return metadata_dict

    def get_con_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene
        :param field_name:{str} field_name
        :return: {dict}
        """

        # metadata_dict = {}
        metadata_dict = self.get_color_metadata(field_name=field_name, field_type=field_type)
        con_field_name = field_name
        metadata_dict['MinRangeFixed'] = Configuration.getSetting("MinRangeFixed", con_field_name)
        metadata_dict['MaxRangeFixed'] = Configuration.getSetting("MaxRangeFixed", con_field_name)
        metadata_dict['MinRange'] = Configuration.getSetting("MinRange", con_field_name)
        metadata_dict['MaxRange'] = Configuration.getSetting("MaxRange", con_field_name)
        metadata_dict['ContoursOn'] = Configuration.getSetting("ContoursOn", con_field_name)
        metadata_dict['NumberOfContourLines'] = Configuration.getSetting("NumberOfContourLines", field_name)
        metadata_dict['ScalarIsoValues'] = cs_string_to_typed_list(Configuration.getSetting("ScalarIsoValues", field_name))
        metadata_dict['LegendEnable'] = Configuration.getSetting("LegendEnable", field_name)



        return metadata_dict

    def get_vector_field_metadata(self,field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene
        :param field_name:{str} field_name
        :return: {dict}
        """

        metadata_dict = self.get_con_field_metadata(field_name=field_name, field_type=field_type)
        metadata_dict['ArrowLength'] = Configuration.getSetting('ArrowLength',field_name)
        metadata_dict['FixedArrowColorOn'] = Configuration.getSetting('FixedArrowColorOn', field_name)
        metadata_dict['ArrowColor'] = qcolor_to_rgba(Configuration.getSetting('ArrowColor', field_name))
        metadata_dict['ScaleArrowsOn'] = Configuration.getSetting('ScaleArrowsOn', field_name)


        return metadata_dict

    def initialize_scene(self):
        self.current_screenshot_data = self.compute_current_screenshot_data()

        self.gd.set_field_extractor(field_extractor=self.parentWidget.fieldExtractor)


    def compute_current_screenshot_data(self):
        """
        Computes/populates Screenshot Description data based ont he current GUI configuration
        for the current window
        :return: {screenshotData}
        """

        scr_data = ScreenshotData()
        self.store_gui_vis_config(scr_data=scr_data)
        metadata = self.get_metadata(field_name=scr_data.plotData[0], field_type=scr_data.plotData[1])

        scr_data.metadata = metadata

        return scr_data


    def store_gui_vis_config(self, scr_data):
        """
        Stores visualization settings such as cell borders, on/or cell on/off etc...

        :param scr_data: {instance of ScreenshotDescriptionData}
        :return: None
        """
        # todo 5 make it a weakref
        tvw = self.parentWidget
        # tvw = self.tabViewWidget()
        # if tvw:
        #     tvw.updateActiveWindowVisFlags(self.screenshotGraphicsWidget)

        scr_data.cell_borders_on = tvw.borderAct.isChecked()
        scr_data.cells_on = tvw.cellsAct.isChecked()
        scr_data.cluster_borders_on = tvw.clusterBorderAct.isChecked()
        scr_data.cell_glyphs_on = tvw.cellGlyphsAct.isChecked()
        scr_data.fpp_links_on = tvw.FPPLinksAct.isChecked()
        scr_data.lattice_axes_on = Configuration.getSetting('ShowHorizontalAxesLabels') or Configuration.getSetting(
            'ShowVerticalAxesLabels')
        scr_data.lattice_axes_labels_on = Configuration.getSetting("ShowAxes")
        scr_data.bounding_box_on = Configuration.getSetting("BoundingBoxOn")

        invisible_types = Configuration.getSetting("Types3DInvisible")
        invisible_types = invisible_types.strip()

        if invisible_types:
            scr_data.invisible_types = list(map(lambda x : int(x), invisible_types.split(',')))
        else:
            scr_data.invisible_types = []

    def draw(self,basic_simulation_data):

        if self.current_screenshot_data is None:
            self.initialize_scene()
            # self.current_screenshot_data = self.compute_current_screenshot_data()



        self.gd.draw(screenshot_data=self.current_screenshot_data, bsd=basic_simulation_data, screenshot_name='')
        print 'drawing scene'

    def conMinMax(self):
        print 'CHANGE THIS'
        return ['0','0']

    def setStatusBar(self, statusBar):
        self._statusBar = statusBar

    # Break the settings read into groups?
    def readSettings(self):  # only performed at startup
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
        # colorsDefaults
        self._colorMap = Configuration.getSetting("TypeColorMap")
        self._borderColor = Configuration.getSetting("BorderColor")
        self._contourColor = Configuration.getSetting("ContourColor")
        self._brushColor = Configuration.getSetting("BrushColor")
        self._penColor = Configuration.getSetting("PenColor")
        self._windowColor = Configuration.getSetting("WindowColor")
        self._boundingBoxColor = Configuration.getSetting("BoundingBoxColor")

    def readViewSets(self):
        # For 3D only?
        # viewDefaults
        self._types3D = Configuration.getSetting("Types3DInvisible")

    def readOutputSets(self):
        # Should I read the settings here?
        # outputDefaults
        self._updateScreen     = Configuration.getSetting("ScreenUpdateFrequency")
        self._imageOutput      = Configuration.getSetting("ImageOutputOn")
        self._shotFrequency    = Configuration.getSetting("ScreenUpdateFrequency")

    def readVisualSets(self):
        # visualDefaults
        self._cellBordersOn = Configuration.getSetting("CellBordersOn")
        self._clusterBordersOn = Configuration.getSetting("ClusterBordersOn")
        #        print MODULENAME,'   readVisualSets():  cellBordersOn, clusterBordersOn = ',self._cellBordersOn, self._clusterBordersOn
        self._conLimitsOn = Configuration.getSetting("ConcentrationLimitsOn")
        self._zoomFactor = Configuration.getSetting("ZoomFactor")

    def configsChanged(self):
        """

        :return:
        """
        print 'configsChanged'


    # def clearDisplayOnDemand(self):
        # self.draw2D.clearDisplay()
        
    # def changeStateMonitor(self,oldState,newState):
        # print 'oldState=',oldState
        # print 'newState=',newState
        
    # def sizeHint(self):
        # return QSize(400,400)
        
    # def changeEvent(self, ev):
        # print 'INSIDE CHANGE EVENT', self, ' type=',ev.type()
        # if ev.type() == QEvent.WindowStateChange:
            # ev.ignore()
            # print 'CHANGING STATE OF THE QWINDOW'
            # print 'max=',self.isMaximized(),' min=',self.isMinimized()
            
            # if self.isMaximized()  and self.screenshotWindowFlag:
                # self.qvtkWidget.GetRenderWindow().SetSize(400,400) # default size
                # self.qvtkWidget.resize(400,400)
        
                # self.showNormal()
                # self.hide()
                
                # print 'self.parent=',
                # # self.parentWidget.activateMainGraphicsWindow()
                # # self.setShown(False)                  
                # # self.showMinimized()
                # # self.parent.activatePreviousSubWindow()
                
            
            
#     def  resizeEvent(self, ev) :
#         print 'THIS IS RESIZE EVENT'
        # print 'resizing graphics window ',self, 'ev.type()=',ev.type(), ' isMaximized=',self.isMaximized()
        # if self.isMaximized():
            # ev.ignore()


    # todo 5
    # def resetAllCameras(self):
    #     print 'resetAllCameras in GraphicsFrame =',self
    #
    #     self.draw2D.resetAllCameras()
    #     self.draw3D.resetAllCameras()
        
    # def __getattr__(self, attr):
    #     """Makes the object behave like a DrawBase"""
    #     if not self.draw3DFlag:
    #         if hasattr(self.draw2D, attr):
    #             return getattr(self.draw2D, attr)
    #         else:
    #             raise AttributeError, self.__class__.__name__ + \
    #                   " has no attribute named " + attr
    #     else:
    #         if hasattr(self.draw3D, attr):
    #             return getattr(self.draw3D, attr)
    #         else:
    #             raise AttributeError, self.__class__.__name__ + \
    #                   " has no attribute named " + attr
            


    
        # print "FINDING ATTRIBUTE ", attr
        # self.getattrFcn(attr)
        
    # def getattrDraw2D(self,attr):
        # print "INSIDE getattrDraw2D ",attr
        # print "hasattr(self.draw2D, attr)=",hasattr(self.draw2D, attr)
        # if hasattr(self.draw2D, attr):
            # print getattr(self.draw2D, attr)
            # return getattr(self.draw2D, attr)
        # else:
            # raise AttributeError, self.__class__.__name__ + \
                  # " has no attribute named " + attr
                  
    # def getattrDraw3D(self,attr):
        # if hasattr(self.draw3D, attr):            
            # return getattr(self.draw3D, attr)
        # else:
            # raise AttributeError, self.__class__.__name__ + \
                  # " has no attribute named " + attr
    def populateLookupTable(self):
        self.drawModel2D.populateLookupTable()
        self.drawModel3D.populateLookupTable()
        
    def Render(self):        
        
#        print MODULENAME, ' ---------Render():'
#        Configuration.getSetting("CurrentFieldName",name)
        color = Configuration.getSetting("WindowColor")
#        r = color.red()
#        g = color.green()
#        b = color.blue()
#        print MODULENAME,'  setBorderColor():   r,g,b=',r,g,b
#        self.borderActor.GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
        self.ren.SetBackground(float(color.red())/255, float(color.green())/255, float(color.blue())/255)
        self.qvtkWidget.Render()
        
    def getCamera(self):
        return self.getActiveCamera()
    
    def getActiveCamera(self):
        return self.ren.GetActiveCamera()
    
    def setActiveCamera(self,_camera):
        return self.ren.SetActiveCamera(_camera)
        
    def getCamera2D(self):    
        return self.camera2D        
        
    def setZoomItems(self,_zitems):
        # todo 5
        # self.draw2D.setZoomItems(_zitems)
        # self.draw3D.setZoomItems(_zitems)
        print 'set zoom items'

    def showBorder(self):
        pass

    def hideBorder(self):
        pass

    def showClusterBorder(self):
        pass

    def hideClusterBorder(self):
        pass


    def showCells(self):
        pass

    def hideCells(self):
        pass

    def setPlane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)
        # print (self.plane, self.planePos)
    def getPlane(self):
        return (self.plane, self.planePos)



    def initCrossSectionToolbar(self):
        cstb = QtWidgets.QToolBar("CrossSection", self)
        #viewtb.setIconSize(QSize(20, 18))
        cstb.setObjectName("CrossSection")
        cstb.setToolTip("Projection")
        
#        cstb.addWidget( QtGui.QLabel("  ")) # Spacer, just make it look pretty
#        cstb.addWidget( QtGui.QLabel("  ")) # Spacer, just make it look pretty 
#        cstb.addWidget(self.threeDRB)
#        cstb.addWidget( QtGui.QLabel("  "))
#        cstb.addWidget(self.xyRB)
#        cstb.addWidget(self.xySB)
#        cstb.addWidget( QtGui.QLabel("  "))
#        cstb.addWidget(self.xzRB)
#        cstb.addWidget(self.xzSB)
#        cstb.addWidget( QtGui.QLabel("  "))
#        cstb.addWidget(self.yzRB)
#        cstb.addWidget(self.yzSB)
#        cstb.addWidget( QtGui.QLabel("    ")) 

        # new technique (rwh, May 2011), instead of individual radio buttons as originally done (above)
        cstb.addWidget(self.projComboBox)
        cstb.addWidget(self.projSpinBox)
        
        cstb.addWidget(self.fieldComboBox)
        cstb.addAction(self.screenshotAct)
        
        return cstb
        
    def __initCrossSectionActions(self):
        # Do I need actions? Probably not, but will leave for a while
        
        # old way
#        self.threeDAct = QtGui.QAction(self)
#        self.threeDRB  = QtGui.QRadioButton("3D")
#        self.threeDRB.addAction(self.threeDAct)
#
#        self.xyAct = QtGui.QAction(self)
#        self.xyRB  = QtGui.QRadioButton("xy")
#        self.xyRB.addAction(self.xyAct)
#
#        self.xySBAct = QtGui.QAction(self)
#        self.xySB  = QtGui.QSpinBox()
#        self.xySB.addAction(self.xySBAct)
#
#        self.xzAct = QtGui.QAction(self)
#        self.xzRB  = QtGui.QRadioButton("xz")
#        self.xzRB.addAction(self.xzAct)
#
#        self.xzSBAct = QtGui.QAction(self)
#        self.xzSB  = QtGui.QSpinBox()
#        self.xzSB.addAction(self.xzSBAct)
#
#        self.yzAct = QtGui.QAction(self)
#        self.yzRB  = QtGui.QRadioButton("yz")
#        self.yzRB.addAction(self.yzAct)
#
#        self.yzSBAct = QtGui.QAction(self)
#        self.yzSB  = QtGui.QSpinBox()
#        self.yzSB.addAction(self.yzSBAct)
        
        # new (rwh, May 2011)
        self.projComboBoxAct = QtWidgets.QAction(self)
        self.projComboBox  = QtWidgets.QComboBox()
        self.projComboBox.addAction(self.projComboBoxAct)
        
        # NB: the order of these is important; rf. setInitialCrossSection where we set 'xy' to be default projection
        self.projComboBox.addItem("3D")
        self.projComboBox.addItem("xy")
        self.projComboBox.addItem("xz")
        self.projComboBox.addItem("yz")
        
        self.projSBAct = QtWidgets.QAction(self)
        self.projSpinBox  = QtWidgets.QSpinBox()
        self.projSpinBox.addAction(self.projSBAct)

        self.fieldComboBoxAct = QtWidgets.QAction(self)
        self.fieldComboBox  = QtWidgets.QComboBox()   # Note that this is different than the fieldComboBox in the Prefs panel (rf. SimpleTabView.py)
        self.fieldComboBox.addAction(self.fieldComboBoxAct)
        self.fieldComboBox.addItem("-- Field Type --")
        # self.fieldComboBox.addItem("cAMP")  # huh?
        import DefaultData
        gip = DefaultData.getIconPath
        self.screenshotAct = QtWidgets.QAction(QtGui.QIcon(gip("screenshot.png")), "&Take Screenshot", self)
        
        

    def _xyChecked(self, checked):
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
        if checked:
            self.projComboBox.setCurrentIndex(1)
#            self.setPlane("xy", self.xySB.value())
#            self.xyRB.setChecked(checked)
#            self.currentDrawingObject.setPlane("xy", self.xySB.value())
#            self.parentWidget._drawField()
#            
    def _xzChecked(self, checked):
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
        if checked:
            self.projComboBox.setCurrentIndex(2)
#            self.setPlane("xz", self.xzSB.value())
#            self.xzRB.setChecked(checked)
#            self.currentDrawingObject.setPlane("xz", self.xzSB.value())
#            self.parentWidget._drawField()
#            
    def _yzChecked(self, checked):
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
        if checked:
            self.projComboBox.setCurrentIndex(3)
#            self.setPlane("yz", self.yzSB.value())
#            self.yzRB.setChecked(checked)
#            self.currentDrawingObject.setPlane("yz", self.yzSB.value())
#            self.parentWidget._drawField()
            
            
    def _projComboBoxChanged(self):   # new (rwh, May 2011)
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
        name = str(self.projComboBox.currentText())
#        print MODULENAME, ' _projectionChanged:  name =',name
        self.currentProjection = name
        
        if self.currentProjection == '3D':  # disable spinbox
#            print '   _projSpinBoxChanged, doing 3D'
            self.projSpinBox.setEnabled(False)
            self.setDrawingStyle("3D")
            if self.parentWidget.completedFirstMCS:
                self.parentWidget.newDrawingUserRequest = True
            self.parentWidget._drawField()   # SimpleTabView.py
        
        elif self.currentProjection == 'xy':
            self.projSpinBox.setEnabled(True)
#            print ' self.xyPlane, xyMaxPlane=',self.xyPlane, self.xyMaxPlane
#            self._xyChanged(2)
#            val = 2
#            self.setPlane("xy", val)
##            self.xySB.setValue(val)
#            self.currentDrawingObject.setPlane("xy", val)
#            self.parentWidget._drawField()
#            self.projSpinBox.setMaximum(self.xyMaxPlane)
#            self.projSpinBox.setValue(self.xyPlane)  # automatically invokes the callback (--Changed)

            self.projSpinBox.setValue(self.xyPlane)
            self._projSpinBoxChanged(self.xyPlane)
#            print '  xy: new max = ', self.projSpinBox.maximum()
#            print '  xy: new val = ', self.projSpinBox.value()
            
        elif self.currentProjection == 'xz':
            self.projSpinBox.setEnabled(True)
#            print ' self.xzPlane, xzMaxPlane=',self.xzPlane, self.xzMaxPlane
#            print ' got xz proj'
#            self.projSpinBox.setMaximum(self.xzMaxPlane)
#            self.projSpinBox.setValue(self.xzPlane)
#            self.projSpinBox.setMaximum(self.xzMaxPlane)
            self.projSpinBox.setValue(self.xzPlane)
            self._projSpinBoxChanged(self.xzPlane)
            
        elif self.currentProjection == 'yz':
            self.projSpinBox.setEnabled(True)
#            print ' self.yzPlane, yzMaxPlane=',self.yzPlane, self.yzMaxPlane
#            self.projSpinBox.setMaximum(self.yzMaxPlane)
#            self.projSpinBox.setValue(self.yzPlane)
            self.projSpinBox.setValue(self.yzPlane)
            self._projSpinBoxChanged(self.yzPlane)
            
            
            
    def _projSpinBoxChanged(self, val):
#        print ' GFW: _projSpinBoxChanged: val =',val
#        print ' spinBoxChanged: self.xyPlane, xyMaxPlane=',self.xyPlane, self.xyMaxPlane

        return

        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
            
        self.setDrawingStyle("2D")
        
        if self.currentProjection == 'xy':
            if val > self.xyMaxPlane: val = self.xyMaxPlane
            self.projSpinBox.setValue(val)
            self.setPlane(self.currentProjection, val)  # for some bizarre(!) reason, val=0 for xyPlane
            self.xyPlane = val
#            self.projSpinBox.setValue(val)
#            print ' _projSpinBoxChanged: set xy val=',val
            # print 'self.currentDrawingObject=',self.currentDrawingObject
            # print 'self.draw2D=',self.draw2D
            
            self.currentDrawingObject.setPlane(self.currentProjection, self.xyPlane)
#            self.parentWidget._drawField()

        elif self.currentProjection == 'xz':
            if val > self.xzMaxPlane: val = self.xzMaxPlane
            self.projSpinBox.setValue(val)
            self.setPlane(self.currentProjection, val)
            self.currentDrawingObject.setPlane(self.currentProjection, val)
#            self.parentWidget._drawField()
            self.xzPlane = val
            
        elif self.currentProjection == 'yz':
            if val > self.yzMaxPlane: val = self.yzMaxPlane
            self.projSpinBox.setValue(val)
            self.setPlane(self.currentProjection, val)
            self.currentDrawingObject.setPlane(self.currentProjection, val)
#            self.parentWidget._drawField()
            self.yzPlane = val
    
        self.parentWidget._drawField()   # SimpleTabView.py
        
            
#        if self.parentWidget.fieldTypes.has_key(name):
#            # Tuple that holds (FieldName, FieldType), e.g. ("FGF", ConField)
#            self.parentWidget.setFieldType((name, self.parentWidget.fieldTypes[name])) 
#            self.parentWidget._drawField()
            
#    def _xyChanged(self, val):
#        if self.parentWidget.completedFirstMCS:
#            self.parentWidget.newDrawingUserRequest = True
#        if self.xyRB.isChecked():
#            self.setPlane("xy", val)
#            self.xySB.setValue(val)
#            self.currentDrawingObject.setPlane("xy", val)
#            self.parentWidget._drawField()
#
#    def _xzChanged(self, val):
#        if self.parentWidget.completedFirstMCS:
#            self.parentWidget.newDrawingUserRequest = True
#        if self.xzRB.isChecked():
#            self.setPlane("xz", val)
#            self.xzSB.setValue(val)
#            self.currentDrawingObject.setPlane("xz", val)
#            self.parentWidget._drawField()
#
#    def _yzChanged(self, val):
#        if self.parentWidget.completedFirstMCS:
#            self.parentWidget.newDrawingUserRequest = True
#        if self.yzRB.isChecked():
#            self.setPlane("yz", val)
#            self.yzSB.setValue(val)
#            self.currentDrawingObject.setPlane("yz", val)
#            self.parentWidget._drawField()
            
    def _fieldTypeChanged(self):
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
#        print MODULENAME,' _fieldTypeChanged(): self.fieldComboBox.count() = ',self.fieldComboBox.count()
#        if self.fieldComboBox.count() > 0:
        name = str(self.fieldComboBox.currentText())
#        print MODULENAME,' _fieldTypeChanged(): name = ',name
#        Configuration.setSetting("CurrentFieldName",name)
#        print MODULENAME,' _fieldTypeChanged(): self.parentWidget=',self.parentWidget
#        print MODULENAME,' _fieldTypeChanged(): type(self.parentWidget)=',type(self.parentWidget)
    #        print MODULENAME,' _fieldTypeChanged(): dir(self.parentWidget)=',dir(self.parentWidget)
        if self.parentWidget.fieldTypes.has_key(name):
            # Tuple that holds (FieldName, FieldType), e.g. ("FGF", ConField)
#            print MODULENAME,' _fieldTypeChanged():      self.parentWidget.fieldTypes[name]=',self.parentWidget.fieldTypes[name]
            self.parentWidget.setFieldType((name, self.parentWidget.fieldTypes[name])) 
            self.parentWidget._drawField()
            
    def setDrawingStyle(self,_style):
        style=string.upper(_style)
        if style=="2D":
            self.draw3DFlag = False
            self.currentDrawingObject = self.draw2D
            self.ren.SetActiveCamera(self.camera2D)
            self.qvtkWidget.setMouseInteractionSchemeTo2D()
            self.draw3D.clearDisplay()
        elif style=="3D":
            self.draw3DFlag = True
            self.currentDrawingObject = self.draw3D
            self.ren.SetActiveCamera(self.camera3D)
            self.qvtkWidget.setMouseInteractionSchemeTo3D()           
            self.draw2D.clearDisplay()
            
    def getCamera3D(self):
        return self.camera3D
        
    def _switchDim(self,checked):
        print MODULENAME, '  _switchDim, checked=',checked
        if self.parentWidget.completedFirstMCS:
            self.parentWidget.newDrawingUserRequest = True
        
        if checked:
            # self.getattrFcn=self.__getattrDraw3D
            self.draw3DFlag = True
            self.ren.SetActiveCamera(self.camera3D)
            self.qvtkWidget.setMouseInteractionSchemeTo3D()           
            self.draw2D.clearDisplay()
#            self.threeDRB.setChecked(True)
            self.projComboBox.setCurrentIndex(0)
            self.parentWidget._drawField()            
            # if not self.usedDraw3DFlag and len(self.draw3D.currentActors.keys()):                
                # self.usedDraw3DFlag=True
                # self.qvtkWidget.resetCamera()
                
        else:
            self.draw3DFlag = False
            self.ren.SetActiveCamera(self.camera2D)
            self.qvtkWidget.setMouseInteractionSchemeTo2D()
            self.draw3D.clearDisplay()
            self.parentWidget._drawField()
            # self.getattrFcn=self.__getattrDraw2D
            
    def getActiveCamera(self):
        return self.ren.GetActiveCamera()
        
    def getCurrentSceneNameAndType(self):
        # this is usually field name but we can also allow other types of visualizations hence I am calling it getCurrerntSceneName
        sceneName = str(self.fieldComboBox.currentText())
        return sceneName , self.parentWidget.fieldTypes[sceneName]        
        
    def apply3DGraphicsWindowData(self,gwd):
    
    
        for p in xrange(self.projComboBox.count()):
            
            if str(self.projComboBox.itemText(p)) == '3D':
            
                # camera = self.getActiveCamera()
                # print 'activeCamera=',activeCamera
            
                self.projComboBox.setCurrentIndex(p)            
                
                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D                
                self.camera3D.SetClippingRange(gwd.cameraClippingRange)
                self.camera3D.SetFocalPoint(gwd.cameraFocalPoint)
                self.camera3D.SetPosition(gwd.cameraPosition)
                self.camera3D.SetViewUp(gwd.cameraViewUp)                
                
                
                break
        
        
    def apply2DGraphicsWindowData(self,gwd):        
    
        for p in xrange(self.projComboBox.count()):
            
            if str(self.projComboBox.itemText(p)).lower() == str(gwd.planeName).lower():
                self.projComboBox.setCurrentIndex(p)            
                # print 'self.projSpinBox.maximum()= ', self.projSpinBox.maximum()    
                # if gwd.planePosition <= self.projSpinBox.maximum():
                self.projSpinBox.setValue(gwd.planePosition)  # automatically invokes the callback (--Changed)
                
                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D                
                self.camera2D.SetClippingRange(gwd.cameraClippingRange)
                self.camera2D.SetFocalPoint(gwd.cameraFocalPoint)
                self.camera2D.SetPosition(gwd.cameraPosition)
                self.camera2D.SetViewUp(gwd.cameraViewUp)                
                
        
    def applyGraphicsWindowData(self,gwd):
        # print 'COMBO BOX CHECK '
        # for i in xrange(self.fieldComboBox.count()):
            # print 'self.fieldComboBox.itemText(i)=',self.fieldComboBox.itemText(i)
        
        
        for i in xrange(self.fieldComboBox.count()):
        
            if str(self.fieldComboBox.itemText(i)) == gwd.sceneName:
            
                
                self.fieldComboBox.setCurrentIndex(i)
                # setting 2D projection or 3D
                if gwd.is3D:
                    self.apply3DGraphicsWindowData(gwd)
                else:                
                    self.apply2DGraphicsWindowData(gwd)
                    

      
                break
                
        # import time
        # time.sleep(2)
        
        
    def getGraphicsWindowData(self):
        from GraphicsWindowData import GraphicsWindowData
        gwd = GraphicsWindowData()
        activeCamera = self.getActiveCamera()
        # gwd.camera = self.getActiveCamera()
        gwd.sceneName = str(self.fieldComboBox.currentText())
        gwd.sceneType = self.parentWidget.fieldTypes[gwd.sceneName]      
        # gwd.winType = 'graphics'
        gwd.winType = GRAPHICS_WINDOW_LABEL
        # winPosition and winPosition will be filled externally by the SimpleTabView , since it has access to mdi windows
        
        # gwd.winPosition = self.pos()        
        # gwd.winSize = self.size()
        
        if self.draw3DFlag:            
            gwd.is3D = True
        else:
            planePositionTupple = self.draw2D.getPlane()
            gwd.planeName = planePositionTupple[0]
            gwd.planePosition = planePositionTupple[1]
            
        # print 'GetClippingRange()=',activeCamera.GetClippingRange()
        gwd.cameraClippingRange = activeCamera.GetClippingRange()
        gwd.cameraFocalPoint = activeCamera.GetFocalPoint()
        gwd.cameraPosition = activeCamera.GetPosition()
        gwd.cameraViewUp = activeCamera.GetViewUp()
        
        
        # import time
        # time.sleep(2)
        return gwd

    def _takeShot(self):
        """
        Adds screenshot data for a current scene
        :return: None
        """

        print MODULENAME, '  _takeShot():  self.renWin.GetSize()=', self.renWin.GetSize()
        camera = self.getActiveCamera()

        if self.parentWidget.screenshotManager is not None:
            field_name = str(self.fieldComboBox.currentText())

            field_type = self.parentWidget.fieldTypes[field_name]
            field_name_type_tuple = (field_name, field_type)
            print MODULENAME, '  _takeShot():  fieldType=', field_name_type_tuple

            if self.draw3DFlag:
                metadata = self.drawModel3D.get_metadata(field_name=field_name, field_type=field_type)
                self.parentWidget.screenshotManager.add3DScreenshot(field_name, field_type, camera, metadata)
            else:
                planePositionTupple = self.draw2D.getPlane()
                metadata = self.drawModel3D.get_metadata(field_name=field_name, field_type=field_type)
                self.parentWidget.screenshotManager.add2DScreenshot(field_name, field_type, planePositionTupple[0],
                                                                    planePositionTupple[1], camera, metadata)


    def setConnects(self,_workspace):   # rf. Plugins/ViewManagerPlugins/SimpleTabView.py

        # TODO
        # self.connect(self.projComboBox,  SIGNAL('currentIndexChanged (int)'), self._projComboBoxChanged)
        # self.connect(self.projSpinBox,   SIGNAL('valueChanged(int)'), self._projSpinBoxChanged)
        #
        # self.connect(self.fieldComboBox, SIGNAL('currentIndexChanged (int)'), self._fieldTypeChanged)
        #
        # self.connect(self.screenshotAct, SIGNAL('triggered()'), self._takeShot)

        self.projComboBox.currentIndexChanged.connect(self._projComboBoxChanged)
        self.projSpinBox.valueChanged.connect(self._projSpinBoxChanged)

        self.fieldComboBox.currentIndexChanged.connect(self._fieldTypeChanged)

        self.screenshotAct.triggered.connect(self._takeShot)

    def setInitialCrossSection(self,_basicSimulationData):
#        print MODULENAME, '  setInitialCrossSection'
        fieldDim = _basicSimulationData.fieldDim
        
#        self.xyRB.setChecked(True)
#        self.xySB.setMinimum(0)
#        self.xySB.setMaximum(fieldDim.z - 1)
#        if fieldDim.z/2 >= 1: # do this trick to avoid empty vtk widget after stop, play sequence for some 3D simulations
#            self.xySB.setValue(fieldDim.z/2+1)  
#        self.xySB.setValue(fieldDim.z/2) # If you want to set the value from configuration
#        self.xySB.setWrapping(True)
#
#        self.xzSB.setMinimum(0)
#        self.xzSB.setMaximum(fieldDim.y - 1)
#        self.xzSB.setValue(fieldDim.y/2)
#        self.xzSB.setWrapping(True)
#
#        self.yzSB.setMinimum(0)
#        self.yzSB.setMaximum(fieldDim.x - 1)
#        self.yzSB.setValue(fieldDim.x/2)
#        self.yzSB.setWrapping(True)
        
        self.updateCrossSection(_basicSimulationData)
        
        # new (rwh, May 2011)
        self.currentProjection = 'xy'   # rwh
        
        
        # # # self.xyMaxPlane = fieldDim.z - 1
# # # #        self.xyPlane = fieldDim.z/2 + 1
        # # # self.xyPlane = fieldDim.z/2
        
        # # # self.xzMaxPlane = fieldDim.y - 1
        # # # self.xzPlane = fieldDim.y/2
        
        # # # self.yzMaxPlane = fieldDim.x - 1
        # # # self.yzPlane = fieldDim.x/2
        
        self.projComboBox.setCurrentIndex(1)   # set to be 'xy' projection by default, regardless of 2D or 3D sim?
        
        self.projSpinBox.setMinimum(0)
#        self.projSpinBox.setMaximum(fieldDim.z - 1)
        self.projSpinBox.setMaximum(10000)
#        if fieldDim.z/2 >= 1: # do this trick to avoid empty vtk widget after stop, play sequence for some 3D simulations
#            self.projSpinBox.setValue(fieldDim.z/2 + 1)  
        self.projSpinBox.setValue(fieldDim.z/2) # If you want to set the value from configuration
#        self.projSpinBox.setWrapping(True)

    def updateCrossSection(self,_basicSimulationData):
        fieldDim = _basicSimulationData.fieldDim
        self.xyMaxPlane = fieldDim.z - 1
#        self.xyPlane = fieldDim.z/2 + 1
        self.xyPlane = fieldDim.z/2
        
        self.xzMaxPlane = fieldDim.y - 1
        self.xzPlane = fieldDim.y/2
        
        self.yzMaxPlane = fieldDim.x - 1
        self.yzPlane = fieldDim.x/2
        
    def takeSimShot(self, *args, **kwds):
        print 'CHANGE - TAKE SIM SHOT'
    def setFieldTypesComboBox(self,_fieldTypes):
        return
        self.fieldTypes=_fieldTypes # assign field types to be the same as field types in the workspace
        self.draw2D.setFieldTypes(self.fieldTypes) # make sure that field types are the same in graphics widget and in the drawing object
        self.draw3D.setFieldTypes(self.fieldTypes) # make sure that field types are the same in graphics widget and in the drawing object
        # self.draw3D.setFieldTypes(self.fieldTypes)# make sure that field types are the same in graphics widget and in the drawing object
        
        self.fieldComboBox.clear()
        self.fieldComboBox.addItem("-- Field Type --")
        self.fieldComboBox.addItem("Cell_Field")
        for key in self.fieldTypes.keys():
            if key !="Cell_Field":
                self.fieldComboBox.addItem(key)
        self.fieldComboBox.setCurrentIndex(1) # setting value of the Combo box to be cellField - default action 
        
        
        # self.qvtkWidget.resetCamera() # last call triggers fisrt call to draw function so we here reset camera so that all the actors are initially visible
        self.resetCamera() # last call triggers fisrt call to draw function so we here reset camera so that all the actors are initially visible

    def zoomIn(self):
        '''
        Zooms in view
        :return:None
        '''
        self.qvtkWidget.zoomIn()

    def zoomOut(self):
        '''
        Zooms in view
        :return:None
        '''
        self.qvtkWidget.zoomOut()

    def resetCamera(self):
        '''
        Resets camera to default settings
        :return:None
        '''
        self.qvtkWidget.resetCamera()


    # note that if you close widget using X button this slot is not called
    # we need to reimplement closeEvent
    # def close(self):           
    def closeEvent(self,ev):
        # print '\n\n\n closeEvent GRAPHICS FRAME'
        
        # cleaning up to release memory - notice that if we do not do this cleanup this widget will not be destroyed and will take sizeable portion of the memory 
        # not a big deal for a single simulation but repeated runs can easily exhaust all system memory
        
        self.clearEntireDisplay()        
        
        
        self.qvtkWidget.close()
        self.qvtkWidget=None
        self.ren.SetActiveCamera(None)
        
        self.ren=None
        self.renWin=None
        
        
        # return
        # cleaning up objects with cyclic references 
        self.drawModel2D=None                
        self.draw2D=None        
        
        
        self.drawModel3D = None
        self.draw3D = None
        # print 'self.currentDrawingObject=',self.currentDrawingObject
        
        self.currentDrawingObject=None
        
        
        self.camera3D = None
        self.camera2D = None
        
        
        

        self.fieldTypes=None

        #MDIFIX
        # self.parentWidget.removeWindowWidgetFromRegistry(self)

        # print 'AFTER CLOSE GFW self.graphicsWindowDict=',self.parentWidget.graphicsWindowDict 
        # print 'self.windowDict=',self.parentWidget.windowDict        
        
        self.parentWidget = None

        print 'GRAPHICS CLOSED'

        
