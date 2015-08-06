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


from PyQt4 import QtCore, QtGui,QtOpenGL
import vtk
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from enums import *

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

class GraphicsFrameWidget(QtGui.QFrame):
    # def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
    def __init__(self, parent=None, originatingWidget=None):

        QtGui.QFrame.__init__(self, parent)
        
        
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

        self.lineEdit = QtGui.QLineEdit()
        
        self.__initCrossSectionActions()
        self.cstb = self.initCrossSectionToolbar()        
        
        layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        layout.addWidget(self.cstb)
        layout.addWidget(self.qvtkWidget)
        self.setLayout(layout)
        self.setMinimumSize(100, 100) #needs to be defined to resize smaller than 400x400
        self.resize(600, 600)
        
        self.qvtkWidget.Initialize()
        self.qvtkWidget.Start()
        
        self.ren = vtk.vtkRenderer()
        self.renWin = self.qvtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)

#        print MODULENAME,"GraphicsFrameWidget():__init__:   parent=",parent

        #    some objects below create cyclic dependencies - the widget will not free its memory unless  in the close event e.g.  self.drawModel2D gets set to None
        # # # from weakref import ref
        
        # # # self_weakref=ref(self)
        # # # self.drawModel2D = MVCDrawModel2D(self_weakref,parent)
        # # # self.draw2D = MVCDrawView2D(self.drawModel2D,self_weakref,parent)
        
        # # # self.drawModel3D = MVCDrawModel3D(self_weakref,parent)
        # # # self.draw3D = MVCDrawView3D(self.drawModel3D,self_weakref,parent)
        

        # MDIFIX
        self.drawModel2D = MVCDrawModel2D(self, self.parentWidget)
        self.draw2D = MVCDrawView2D(self.drawModel2D, self, self.parentWidget)

        self.drawModel3D = MVCDrawModel3D(self, self.parentWidget)
        self.draw3D = MVCDrawView3D(self.drawModel3D, self, self.parentWidget)

        # self.drawModel2D = MVCDrawModel2D(self,parent)
        # self.draw2D = MVCDrawView2D(self.drawModel2D,self,parent)
        #
        # self.drawModel3D = MVCDrawModel3D(self,parent)
        # self.draw3D = MVCDrawView3D(self.drawModel3D,self,parent)





        # self.draw2D=Draw2D(self,parent)
        # self.draw3D=Draw3D(self,parent)
        self.camera3D = self.ren.MakeCamera()        
        self.camera2D = self.ren.GetActiveCamera()
        self.ren.SetActiveCamera(self.camera2D)
        
        
        self.currentDrawingObject = self.draw2D
        
        
        self.draw3DFlag = False
        self.usedDraw3DFlag = False
        # self.getattrFcn=self.getattrDraw2D
        
        # rwh test
#        print MODULENAME,' self.parentWidget.playerSettingsFileName= ',self.parentWidget.playerSettingsFileName
        if self.parentWidget.playerSettingsFileName:
            # does not work on Windows with Python 2.5
            # with open(self.parentWidget.playerSettingsFileName, 'r') as f:    
            f=None
            try:
                f=open(self.parentWidget.playerSettingsFileName, 'r')
                while True:
                    l = f.readline()
                    if l == "": break
                    v = l.split()
                    print v
                    if string.find(v[0], 'CameraPosition') > -1:
                        self.camera3D.SetPosition(float(v[1]), float(v[2]), float(v[3]))
                    elif string.find(v[0], 'CameraFocalPoint') > -1:
                        self.camera3D.SetFocalPoint(float(v[1]), float(v[2]), float(v[3]))
                    elif string.find(v[0], 'CameraViewUp') > -1:
                        self.camera3D.SetViewUp(float(v[1]), float(v[2]), float(v[3]))
#                    elif string.find(v[0], 'ViewPlaneNormal') > -1:   # deprecated
#                        self.camera3D.SetViewPlaneNormal(float(v[1]), float(v[2]), float(v[3]))
                    elif string.find(v[0], 'CameraClippingRange') > -1:
                        self.camera3D.SetClippingRange(float(v[1]), float(v[2]))
                    elif string.find(v[0], 'CameraDistance') > -1:
                        print 'SetDistance = ',float(v[1])
                        self.camera3D.SetDistance(float(v[1]))
                    elif string.find(v[0], 'ViewAngle') > -1:
                        self.camera3D.SetViewAngle(float(v[1]))
                
            except IOError:
                pass
            
        self.screenshotWindowFlag=False
            # # # self.setDrawingStyle("3D")
    #        self.currentDrawingObject = self.draw3D
    #        self.draw3DFlag = True
    #        self.usedDraw3DFlag = False
    #        self.ren.SetActiveCamera(self.camera3D)
#            self.camera3D.SetPosition(100,100,100)
#            self.qvtkWidget.SetSize(200,200)

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
        
        
    def resetAllCameras(self):
        print 'resetAllCameras in GraphicsFrame =',self
        
        self.draw2D.resetAllCameras()
        self.draw3D.resetAllCameras()
        
    def __getattr__(self, attr):
        """Makes the object behave like a DrawBase"""
        if not self.draw3DFlag:
            if hasattr(self.draw2D, attr):            
                return getattr(self.draw2D, attr)
            else:
                raise AttributeError, self.__class__.__name__ + \
                      " has no attribute named " + attr
        else:
            if hasattr(self.draw3D, attr):            
                return getattr(self.draw3D, attr)
            else:
                raise AttributeError, self.__class__.__name__ + \
                      " has no attribute named " + attr
            


    
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
        self.draw2D.setZoomItems(_zitems)
        self.draw3D.setZoomItems(_zitems)
        
    def setPlane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)
        # print (self.plane, self.planePos)
    def getPlane(self):
        return (self.plane, self.planePos)
        
    def initCrossSectionToolbar(self):
        cstb = QtGui.QToolBar("CrossSection", self)
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
        self.projComboBoxAct = QtGui.QAction(self)
        self.projComboBox  = QtGui.QComboBox()
        self.projComboBox.addAction(self.projComboBoxAct)
        
        # NB: the order of these is important; rf. setInitialCrossSection where we set 'xy' to be default projection
        self.projComboBox.addItem("3D")
        self.projComboBox.addItem("xy")
        self.projComboBox.addItem("xz")
        self.projComboBox.addItem("yz")
        
        self.projSBAct = QtGui.QAction(self)
        self.projSpinBox  = QtGui.QSpinBox()
        self.projSpinBox.addAction(self.projSBAct)

        self.fieldComboBoxAct = QtGui.QAction(self)
        self.fieldComboBox  = QtGui.QComboBox()   # Note that this is different than the fieldComboBox in the Prefs panel (rf. SimpleTabView.py)
        self.fieldComboBox.addAction(self.fieldComboBoxAct)
        self.fieldComboBox.addItem("-- Field Type --")
        # self.fieldComboBox.addItem("cAMP")  # huh?
        import DefaultData
        gip = DefaultData.getIconPath
        self.screenshotAct = QtGui.QAction(QtGui.QIcon(gip("screenshot.png")), "&Take Screenshot", self)
        
        

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
#        print MODULENAME, '  _takeShot():  self.parentWidget.screenshotManager=',self.parentWidget.screenshotManager
        print MODULENAME, '  _takeShot():  self.renWin.GetSize()=',self.renWin.GetSize()
        camera = self.getActiveCamera()
        # # # camera = self.ren.GetActiveCamera()
#        print MODULENAME, '  _takeShot():  camera=',camera
#        clippingRange= camera.GetClippingRange()
#        focalPoint= camera.GetFocalPoint()
#        position= camera.GetPosition()
#        viewUp= camera.GetViewUp()
#        viewAngle= camera.GetViewAngle()
#        print MODULENAME,"_takeShot():  Range,FP,Pos,Up,Angle=",clippingRange,focalPoint,position,viewUp,viewAngle

        if self.parentWidget.screenshotManager is not None:
            name = str(self.fieldComboBox.currentText())
            self.parentWidget.fieldTypes[name]
            fieldType = (name,self.parentWidget.fieldTypes[name])
            print MODULENAME, '  _takeShot():  fieldType=',fieldType
        
#            if self.threeDRB.isChecked():
            if self.draw3DFlag:
                self.parentWidget.screenshotManager.add3DScreenshot(fieldType[0],fieldType[1],camera)
            else:
                planePositionTupple = self.draw2D.getPlane()
                # print "planePositionTupple=",planePositionTupple
                self.parentWidget.screenshotManager.add2DScreenshot(fieldType[0],fieldType[1],planePositionTupple[0],planePositionTupple[1],camera)

    
    def setConnects(self,_workspace):   # rf. Plugins/ViewManagerPlugins/SimpleTabView.py
#        self.connect(self.threeDRB, SIGNAL('toggled(bool)'), self._switchDim)
#        
#        self.connect(self.xyRB,     SIGNAL('clicked(bool)'), self._xyChecked)
#        self.connect(self.xzRB,     SIGNAL('clicked(bool)'), self._xzChecked)
#        self.connect(self.yzRB,     SIGNAL('clicked(bool)'), self._yzChecked)
#        
#        self.connect(self.xySB,     SIGNAL('valueChanged(int)'), self._xyChanged)
#        self.connect(self.xzSB,     SIGNAL('valueChanged(int)'), self._xzChanged)
#        self.connect(self.yzSB,     SIGNAL('valueChanged(int)'), self._yzChanged)
        
        self.connect(self.projComboBox,  SIGNAL('currentIndexChanged (int)'), self._projComboBoxChanged)
        self.connect(self.projSpinBox,   SIGNAL('valueChanged(int)'), self._projSpinBoxChanged)
        
        self.connect(self.fieldComboBox, SIGNAL('currentIndexChanged (int)'), self._fieldTypeChanged)
        
        self.connect(self.screenshotAct, SIGNAL('triggered()'), self._takeShot)
    

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
        
    
    def setFieldTypesComboBox(self,_fieldTypes):    
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

        
