from __future__ import with_statement
# enabling with statement in python 2.5

# -*- coding: utf-8 -*-
import os, sys
import re
import inspect
import string
import time

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtXml import *

from enums import *

from Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging

setDebugging(1)

FIELD_TYPES = (
    "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel", "CustomVis")
PLANES = ("xy", "xz", "yz")

MODULENAME = '---- SimpleTabView.py: '

# from ViewManager.ViewManager import ViewManager
from ViewManager.SimpleViewManager import SimpleViewManager
from  Graphics.GraphicsFrameWidget import GraphicsFrameWidget

# from Utilities.QVTKRenderWidget import QVTKRenderWidget
from Utilities.SimModel import SimModel
from Configuration.ConfigurationDialog import ConfigurationDialog
import Configuration
import DefaultData

from Simulation.CMLResultReader import CMLResultReader
from Simulation.SimulationThread import SimulationThread
# from Simulation.SimulationThread1 import SimulationThread1

import ScreenshotManager
import vtk
from RollbackImporter import RollbackImporter

try:
    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)
    import CompuCellSetup
except:
    print 'STView: sys.path=', sys.path


# *********** TO DO
# 1. UNCOMMENT # if Configuration.getSetting('FloatingWindows'):
# 2. ADD WEAKREF TO PLOT FRAME WIDGET< PLOT INTERFACE CARTESIAN ETC...
# 3. CELLS OFF, ON removes borders and outline
# 4. CHECK IF IT IS NECESSARY TO FIX CLOSE EVENTS AND REMOVE GRAPHICS WIDGET PLOT WIDGET FROM ANY TYPE OF REGISTRIES
# 5. GET RID OF self.saveWindowsGeometryAct
# 6. FIX updateWindow menu

from MainArea import MainArea
# if Configuration.getSetting('FloatingWindows'):
#     from MainArea import MainArea
# else:
#     from MainAreaMdi import MainArea




# class SimpleTabView(QMdiArea, SimpleViewManager):
class SimpleTabView(MainArea, SimpleViewManager):
    def __init__(self, parent):
        # sys.path.append(os.environ["PYTHON_MODULE_PATH"])
        import CompuCellSetup

        # MainArea.__init__(self, parent=None)
        # MainArea.__init__(self, parent=self)
        # QMdiArea.__init__(self, parent=self)

        # self.MDI_ON = False

        self.__parent = parent  # QMainWindow -> UI.UserInterface
        self.UI = parent


        # QTabWidget.__init__(self, parent)
        # MainArea.__init__(self, parent=parent)
        SimpleViewManager.__init__(self, parent)
        MainArea.__init__(self, stv = self, ui = parent)



        self.__createStatusBar()
        self.__setConnects()

        # MDIFIX
        # if self.MDI_ON:
        #
        #     self.scrollView = QScrollArea(self)
        #     self.scrollView.setBackgroundRole(QPalette.Dark)
        #     self.scrollView.setVisible(False)
        #
        #     #had to introduce separate scrollArea for 2D and 3D widgets. for some reason switching graphics widgets in Scroll area  did not work correctly.
        #     self.scrollView3D = QScrollArea(self)
        #     self.scrollView3D.setBackgroundRole(QPalette.Dark)
        #     self.scrollView3D.setVisible(False)
        #
        #     # qworkspace
        #     # self.setScrollBarsEnabled(True)
        #
        #     self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #     self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)


        #holds ptr (stored as long int) to original cerr stream buffer
        self.cerrStreamBufOrig = None

        # turning off vtk debug output. This requires small modification to the vtk code itself.
        # Files affected vtkOutputWindow.h vtkOutputWindow.cxx vtkWin32OutputWindow.h vtkWin32OutputWindow.cxx
        if hasattr(vtk.vtkOutputWindow, "setOutputToWindowFlag"):
            vtkOutput = vtk.vtkOutputWindow.GetInstance()
            vtkOutput.setOutputToWindowFlag(False)

        self.rollbackImporter = None

        from PlotManagerSetup import createPlotManager

        self.useVTKPlots = False
        self.plotManager = createPlotManager(self,
                                             self.useVTKPlots)  # object responsible for creating/managing plot windows so they're accessible from steppable level
        #        print MODULENAME," __init__:  self.plotManager=",self.plotManager

        self.fieldTypes = {}

        self.pluginTab = None
        self.mysim = None

        self.simulation = None  # gets assigned to SimulationThread down in prepareForNewSimulation()
        self.screenshotManager = None
        self.zitems = []
        self.__fileName = ""  # simulation model filename
        self.__windowsXMLFileName = ""
        #        self.setParams()

        self.__fieldType = ("Cell_Field", FIELD_TYPES[0])
        self.simulationIsStepping = False
        self.simulationIsRunning = False
        self.screenshotDirectoryName = ""
        self.playerSettingsFileName = ""
        self.resultStorageDirectory = ""
        self.customScreenshotDirectoryName = ""
        self.prevOutputDir = ""
        self.baseScreenshotName = ""
        self.latticeType = Configuration.LATTICE_TYPES["Square"]
        self.newDrawingUserRequest = False
        self.completedFirstMCS = False

        self.customSettingPath = ''
        self.cmlHandlerCreated = False

        self.basicSimulationData = None
        self.saveSettings = True

        self.closePlayerAfterSimulationDone = False

        self.__screenshotDescriptionFileName = ""
        self.__outputDirectory = ""
        self.__prefsFile = ""

        self.__viewManagerType = "Regular"

        self.screenshotNumberOfDigits = 10  #this determines how many digits screenshot number of screenshot file name should have

        # self.windowCounter = 0
        # self.windowDict = {}
        # self.plotWindowDict = {}
        # self.graphicsWindowDict = {}
        self.graphicsWindowVisDict = {} # stores visualization seeetings for each open window

        # This extra dictionary is needed to map widget number to mdiWidget. When inserting window to  
        # mdiArea it is wrapped as QMdiSubWindow and QMdiSubWindow are used as identifiers 
        # in managing QMdiArea. If we sore only underlying widgets (i.e. those that QMdisubwindow wrap 
        # then we will not have access to QMdiSubWindow when e.g. we want to make it active )
        # We might also use subWindowList to deal with it, but for now we will use the solution with extra dictionary.
        # self.mdiWindowDict = {}

        # self.graphicsWindowActionsDict = {}

        self.lastActiveRealWindow = None

        self.lastActiveWindow = None
        self.lastPositionMainGraphicsWindow = None
        self.newWindowDefaultPlane = None

        self.cc3dSimulationDataHandler = None

        # for more information on QSignalMapper see Mark Summerfield book "Rapid GUI Development with PyQt"
        self.windowMapper = QSignalMapper(self)
        self.connect(self.windowMapper, SIGNAL("mapped(QWidget*)"), self.setActiveSubWindowCustomSlot)

        self.prepareForNewSimulation(_forceGenericInitialization=True)
        #        print MODULENAME,'__init__:   after prepareForNewSimulation(),  self.mysim = ',self.mysim

        self.setParams()
        # self.keepOldTabs = False  #this flag sets if tabs should be removed before creating new one or not
        self.mainGraphicsWindow = None  # vs.  lastActiveWindow

        # determine if some relevant plugins are defined in the model
        self.pluginFPPDefined = False  # FocalPointPlasticity
        self.pluginCOMDefined = False  # CenterOfMass
        # is there a better way to check for plugins being defined?
        # mainGraphicsWindow.drawModel2D.currentDrawingParameters.bsd.sim.getCC3DModuleData("Plugin","FocalPointPlasticity"):

        # Note: we cannot check the plugins here as CompuCellSetup.cc3dXML2ObjConverter.root is not defined

        # nextSimulation holds the name of the file that will be inserted as a new simulation to run after current simulation gets stopped
        self.nextSimulation = ""
        self.dlg = None

        #parameter scan variables
        self.singleSimulation = False
        self.parameterScanFile = ''
        self.parameterScanOutputDir = ''
        self.consecutiveRunCounter = 0

        self.maxNumberOfConsecutiveRuns = 50
        # extracting from the runScript maximum number of consecutive runs
        try:
            self.maxNumberOfConsecutiveRuns = int(os.environ["MAX_NUMBER_OF_CONSECUTIVE_RUNS"])
        except:  # if for whatever reason we cannot do it we stay with the default value
            pass

            # note that this variable will be the same as self.simulation when doing CMLReplay mode. I keep it under diffferent name to keep track of the places in the code where I am using SimulationThread API and where I use CMLResultReade replay part of the API
        # this means that further refactoring is needed but I leave it for now    
        self.cmlReplayManager = None


    def getSimFileName(self):
        return self.__fileName

    def getWindowsXMLFileName(self):
        return self.__windowsXMLFileName

    def updateRecentFileMenu(self):
        menusDict = self.__parent.getMenusDictionary()
        rencentSimulationsMenu = menusDict["recentSimulations"]
        rencentSimulationsMenu.clear()
        recentSimulations = Configuration.getSetting("RecentSimulations")

        simCounter = 1
        for simulationFileName in recentSimulations:
            actionText = self.tr("&%1 %2").arg(simCounter).arg(simulationFileName)
            # action=rencentSimulationsMenu.addAction(actionText)
            action = QAction("&%d %s " % (simCounter, simulationFileName), self)
            rencentSimulationsMenu.addAction(action)
            action.setData(QVariant(simulationFileName))
            self.connect(action, SIGNAL("triggered()"), self.__openRecentSim)

            simCounter += 1
        return

    def setActiveSubWindowCustomSlot(self, window):
        #MDIFIX
        return

        # print 'INSIDE setActiveSubWindowCustomSlot'
        #        print MODULENAME,"setActiveSubWindow: window=",window
        #        print MODULENAME,'\n ------------------  setActiveSubWindowCustomSlot():  self.mdiWindowDict =', self.mdiWindowDict
        windowNames = self.plotWindowDict.keys()
        #        for windowName in windowNames:
        #          print MODULENAME,'     setActiveSubWindowCustomSlot():  windowName=', windowName

        #        print 'dir(window)=',dir(window)
        # print 'WINDOW=',window

        mdiWindow = self.findMDISubWindowForWidget(window)

        if mdiWindow:
            #            self.setActiveSubWindow(window)

            # MDIFIX
            self.setActiveSubWindow(mdiWindow)

            self.lastActiveRealWindow = mdiWindow

            self.lastActiveWindow = window

            # if window:
        # #            self.setActiveSubWindow(window)
        # self.setActiveSubWindow(self.mdiWindowDict.values()[0])

        # self.lastActiveWindow = window
        #            print "MODULENAME,'         setActiveSubWindowCustomSlot(): self.lastActiveWindow.winId().__int__()=",self.lastActiveWindow.windowId().__int__()
        #            print MODULENAME,"         setActiveSubWindowCustomSlot(): self.lastActiveWindow is ",self.lastActiveWindow.windowTitle()

        #            self.updateActiveWindowVisFlags()

        if (self.lastActiveWindow is not None) and (
                    self.lastActiveWindow.winId().__int__() in self.graphicsWindowVisDict.keys()):
            dictKey = self.lastActiveWindow.winId().__int__()
            if dictKey in self.graphicsWindowVisDict.keys():
                #               self.simulation.drawMutex.lock()  # lock/unlock necessary or not?
                #                print MODULENAME,'------- setActiveSubWindowCustomSlot():  updating *Act.setChecked:  cellsAct=',self.graphicsWindowVisDict[self.lastActiveWindow.winId().__int__()][0]
                #                print MODULENAME,'------- setActiveSubWindowCustomSlot():  updating *Act.setChecked: borderAct=',self.graphicsWindowVisDict[self.lastActiveWindow.winId().__int__()][1]
                self.cellsAct.setChecked(self.graphicsWindowVisDict[dictKey][0])
                self.borderAct.setChecked(self.graphicsWindowVisDict[dictKey][1])
                self.clusterBorderAct.setChecked(self.graphicsWindowVisDict[dictKey][2])
                self.cellGlyphsAct.setChecked(self.graphicsWindowVisDict[dictKey][3])
                self.FPPLinksAct.setChecked(self.graphicsWindowVisDict[dictKey][4])


    def updateActiveWindowVisFlags(self, window=None):

        try:
            if window:
                dictKey = window.winId().__int__()
            else:
                dictKey = self.lastActiveWindow.winId().__int__()
        except StandardError:
            print MODULENAME, 'updateActiveWindowVisFlags():  Could not find any open windows. Ignoring request'
            return
        #        print MODULENAME, 'updateActiveWindowVisFlags():  dictKey (of lastActiveWindow)=',dictKey
        #        if self.lastActiveWindow:
        #            print MODULENAME, 'updateActiveWindowVisFlags():  self.lastActiveWindow.windowTitle()=',self.lastActiveWindow.windowTitle()
        self.graphicsWindowVisDict[dictKey] = (self.cellsAct.isChecked(), self.borderAct.isChecked(), \
                                               self.clusterBorderAct.isChecked(), self.cellGlyphsAct.isChecked(),
                                               self.FPPLinksAct.isChecked() )

    #        print MODULENAME, 'updateActiveWindowVisFlags():  self.graphicsWindowVisDict[self.lastActiveWindow.winId().__int__()]=',self.graphicsWindowVisDict[self.lastActiveWindow.winId().__init__()]
    #        print MODULENAME, 'updateActiveWindowVisFlags():  self.graphicsWindowVisDict=',self.graphicsWindowVisDict


    # Invoked whenever 'Window' menu is clicked. It does NOT modify lastActiveWindow directly (setActiveSubWindowCustomSlot does)
    def updateWindowMenu(self):

        #        if self.lastActiveWindow is not None:
        #            print MODULENAME,'------- updateWindowMenu(): (starting)  self.lastActiveWindow.winId()=',self.lastActiveWindow.winId()
        menusDict = self.__parent.getMenusDictionary()
        windowMenu = menusDict["window"]
        windowMenu.clear()
        windowMenu.addAction(self.newGraphicsWindowAct)
        # windowMenu.addAction(self.newPlotWindowAct)
        windowMenu.addAction(self.tileAct)
        windowMenu.addAction(self.cascadeAct)
        windowMenu.addAction(self.saveWindowsGeometryAct)
        windowMenu.addAction(self.minimizeAllGraphicsWindowsAct)
        windowMenu.addAction(self.restoreAllGraphicsWindowsAct)
        windowMenu.addSeparator()
        windowMenu.addAction(self.closeActiveWindowAct)
        windowMenu.addAction(self.closeAdditionalGraphicsWindowsAct)
        windowMenu.addSeparator()

        # adding graphics windows
        counter = 0

        # for windowName in self.graphicsWindowDict.keys():
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsWidget = win.widget()
            if graphicsWidget.is_screenshot_widget:
                continue


            if counter < 9:
                actionText = self.tr("&%1 %2").arg(counter + 1).arg(win.windowTitle())
            else:
                actionText = self.tr("%1 %2").arg(counter + 1).arg(win.windowTitle())

            action = windowMenu.addAction(actionText)
            action.setCheckable(True)
            myFlag = self.lastActiveWindow == graphicsWidget
            action.setChecked(myFlag)

            self.connect(action, SIGNAL("triggered()"), self.windowMapper, SLOT("map()"))
            self.windowMapper.setMapping(action, win)
            counter += 1

        # for windowName in self.graphicsWindowDict.keys():
        #     graphicsWindow = self.graphicsWindowDict[windowName]
        #     if counter < 9:
        #         actionText = self.tr("&%1 %2").arg(counter + 1).arg(graphicsWindow.windowTitle())
        #     else:
        #         actionText = self.tr("%1 %2").arg(counter + 1).arg(graphicsWindow.windowTitle())
        #
        #     action = windowMenu.addAction(actionText)
        #     action.setCheckable(True)
        #     myFlag = self.lastActiveWindow == graphicsWindow
        #     action.setChecked(myFlag)
        #
        #     self.connect(action, SIGNAL("triggered()"), self.windowMapper, SLOT("map()"))
        #     self.windowMapper.setMapping(action, graphicsWindow)
        #     counter += 1


    # def findMDISubWindowForWidget(self, _widget):
    #     # we look here for an mdiWindows that has widget which is same as _widget
    #     for windowName, mdiWindow in self.mdiWindowDict.iteritems():
    #         try:
    #             if self.windowDict[windowName] == _widget:
    #                 return mdiWindow  #
    #         except LookupError, e:
    #             pass
    #     return None

    # def addGraphicsWindowToWindowRegistry(self, _window):
    #
    #     self.graphicsWindowDict[self.windowCounter] = _window
    #     self.windowDict[self.windowCounter] = _window

    # def addMDIWindowToRegistry(self, _mdiWindow):
    #
    #     self.mdiWindowDict[self.windowCounter] = _mdiWindow

    # def removeWindowFromRegistry(self, _window):
    #
    #     _windowWidget = _window.widget()
    #     self.removeWindowWidgetFromRegistry(_windowWidget=_windowWidget)

    # def removeWindowWidgetFromRegistry(self, _windowWidget):
    #
    #     #remove window from general-purpose self.windowDict
    #
    #
    #     for windowName, windowWidget in self.windowDict.iteritems():
    #
    #         if windowWidget == _windowWidget:
    #             del self.windowDict[windowName]
    #
    #             del self.mdiWindowDict[windowName]
    #
    #             if self.mainGraphicsWindow == windowWidget:
    #                 self.mainGraphicsWindow = None
    #
    #             if self.lastActiveWindow == windowWidget:
    #                 self.lastActiveWindow = None
    #
    #             break
    #     # MDIFIX
    #     # #try removing window from graphics Window dict
    #     # for windowName, windowWidget in self.graphicsWindowDict.iteritems():
    #     #
    #     #     if windowWidget == _windowWidget:
    #     #         del self.graphicsWindowDict[windowName]
    #     #         break
    #
    #
    #             # def addNewPlotWindow(self):
    #             # # from PlotManager import CustomPlot
    #             # # customPlot=CustomPlot(self.plotManager)
    #             # # customPlot.initPlot()
    #
    #             # # print "ADDING NEW WINDOW"
    #             # # sys.exit()
    #             # # import time
    #             # # time.sleep(2)
    #
    #             # return self.plotManager.addNewPlotWindow()

    # def createDockWindow(self, name):
    #     """
    #     Private method to create a dock window with common properties.
    #
    #     @param name object name of the new dock window (string or QString)
    #     @return the generated dock window (QDockWindow)
    #     """
    #     # dock = QDockWidget(self)
    #     dock = QDockWidget(self)
    #     dock.setObjectName(name)
    #     #dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))
    #     return dock
    #
    # def setupDockWindow(self, dock, where, widget, caption):
    #     """
    #     Private method to configure the dock window created with __createDockWindow().
    #
    #     @param dock the dock window (QDockWindow)
    #     @param where dock area to be docked to (Qt.DockWidgetArea)
    #     @param widget widget to be shown in the dock window (QWidget)
    #     @param caption caption of the dock window (string or QString)
    #     """
    #     if caption is None:
    #         caption = QString()
    #
    #     dock.setFloating(True)
    #
    #     self.UI.addDockWidget(where, dock)
    #     dock.setWidget(widget)
    #     dock.setWindowTitle(caption)
    #     dock.show()

    def addNewGraphicsWindow(self):  # callback method to create additional ("Aux") graphics windows
        print MODULENAME, '--------- addNewGraphicsWindow() '
        # if self.pauseAct.isEnabled():
        # self.__pauseSim()

        if not self.simulationIsRunning:
            return
        self.simulation.drawMutex.lock()

        # MDIFIX
        # self.windowCounter += 1

        #MDIFIX

        newWindow = GraphicsFrameWidget(parent=None, originatingWidget=self)
        # newWindow = GraphicsFrameWidget(self)  # "newWindow" is actually a QFrame

        #MDIFIX
        # self.addGraphicsWindowToWindowRegistry(newWindow)

        # self.windowDict[self.windowCounter] = newWindow
        # self.graphicsWindowDict[self.windowCounter] = newWindow

        # MDIFIX
        # newWindow.setWindowTitle("Aux Graphics Window " + str(self.windowCounter))

        newWindow.setZoomItems(self.zitems)  # Set zoomFixed parameters

        self.lastActiveWindow = newWindow
        #        print MODULENAME,'  addNewGraphicsWindow():  self.lastActiveWindow=',self.lastActiveWindow
        #        print MODULENAME,'  addNewGraphicsWindow():  self.lastActiveWindow.winId().__int__()=',self.lastActiveWindow.winId().__int__()
        # self.updateWindowMenu()

        newWindow.setShown(False)
        self.connect(self, SIGNAL('configsChanged'), newWindow.draw2D.configsChanged)
        self.connect(self, SIGNAL('configsChanged'), newWindow.draw3D.configsChanged)

        newWindow.readSettings()  # Graphics/MVCDrawViewBase.py
        # setting up plane tuple based on window number 1
        # plane=self.windowDict[1].getPlane()
        # newWindow.setPlane(plane[0],plane[1])

        #each new window is painted in 2D mode xy projection with z coordinate set to fieldDim.z/2
        self.newWindowDefaultPlane = ("XY", self.basicSimulationData.fieldDim.z / 2)
        newWindow.setPlane(self.newWindowDefaultPlane[0], self.newWindowDefaultPlane[1])

        # newWindow.currentDrawingObject.setPlane(plane[0],plane[1])
        newWindow.currentDrawingObject.setPlane(self.newWindowDefaultPlane[0], self.newWindowDefaultPlane[1])
        # self.simulation.drawMutex.unlock()

        # newWindow.setConnects(self)
        # newWindow.setInitialCrossSection(self.basicSimulationData)
        # newWindow.setFieldTypesComboBox(self.fieldTypes)


        # self.simulation.setGraphicsWidget(self.mainGraphicsWindow)
        # self.mdiWindowDict[self.windowCounter] = self.addSubWindow(newWindow)
        mdiWindow = self.addSubWindow(newWindow)

        # MDIFIX
        self.lastActiveRealWindow = mdiWindow

        # # # mdiWindow.setFixedSize(300,300)
        # # # mdiWindow.move(0,0)

        #MDIFIX
        # self.addMDIWindowToRegistry(mdiWindow)


        # self.mdiWindowDict[self.windowCounter]
        self.updateActiveWindowVisFlags()

        newWindow.show()
        #        print MODULENAME, '--------- addNewGraphicsWindow: mdiWindowDict= ',self.mdiWindowDict
        # camera=self.windowDict[1].getCamera2D()
        # newWindow.setActiveCamera(camera)
        # newWindow.resetCamera()


        self.simulation.drawMutex.unlock()

        newWindow.setConnects(self)  # in GraphicsFrameWidget
        newWindow.setInitialCrossSection(self.basicSimulationData)
        newWindow.setFieldTypesComboBox(self.fieldTypes)

        return mdiWindow


    def activateMainGraphicsWindow(self):
        self.setActiveSubWindow(self.mainMdiSubWindow)

    def addVTKWindowToWorkspace(self):  # just called one time, for initial graphics window  (vs. addNewGraphicsWindow())
        # print MODULENAME,' =================================addVTKWindowToWorkspace ========='
        #        dbgMsg(' addVTKWindowToWorkspace =========')
        # self.graphics2D = Graphics2DNew(self)     
        # print 'BEFORE self.mainGraphicsWindow = GraphicsFrameWidget(self)'
        # time.sleep(5)

        # print 'BEFORE ADD VTK WINDOW TO WORKSPACE'
        # time.sleep(5)
        # print 'addVTKWindowToWorkspace'

        # MDIFIX
        self.mainGraphicsWindow = GraphicsFrameWidget(parent=None, originatingWidget=self)
        # if self.MDI_ON:
        #     gfw = GraphicsFrameWidget(parent=None, originatingWidget=self)
        #     self.mainGraphicsWindow = gfw
        #
        #     # subWindow = self.createDockWindow(name="Graphincs Window") # graphics dock window
        #     # self.setupDockWindow(subWindow, Qt.NoDockWidgetArea, gfw, self.trUtf8("Graphincs Window"))
        #
        #     # Qt.LeftDockWidgetArea
        # else:
        #     self.mainGraphicsWindow = GraphicsFrameWidget(parent=None, originatingWidget=self)
        #     # self.mainGraphicsWindow = GraphicsFrameWidget(self)

        # self.mainGraphicsWindow.deleteLater()

        # QTimer.singleShot(0, self.showNormal)

        # print 'AFTER DELETING GRAPHINCS WINDOW'
        # time.sleep(5)


        #        print MODULENAME,'-------- type(self.mainGraphicsWindow)= ',type(self.mainGraphicsWindow)
        #        print MODULENAME,' ====================addVTKWindowToWorkspace(): type(self.mainGraphicsWindow)=',type(self.mainGraphicsWindow)
        #        print MODULENAME,' ====================addVTKWindowToWorkspace(): dir(self.mainGraphicsWindow)=',dir(self.mainGraphicsWindow)
        # we make sure that first graphics window is positioned in the left upper corner
        # NOTE: we have to perform move prior to calling addSubWindow. or else we will get distorted window
        if self.lastPositionMainGraphicsWindow is not None:
            self.mainGraphicsWindow.move(self.lastPositionMainGraphicsWindow)
        else:
            self.lastPositionMainGraphicsWindow = self.mainGraphicsWindow.pos()

        self.mainGraphicsWindow.setShown(False)

        # self.connect(self, SIGNAL('configsChanged'), self.graphics2D.configsChanged)        
        self.connect(self, SIGNAL('configsChanged'), self.mainGraphicsWindow.draw2D.configsChanged)
        self.connect(self, SIGNAL('configsChanged'), self.mainGraphicsWindow.draw3D.configsChanged)
        self.mainGraphicsWindow.readSettings()
        self.simulation.setGraphicsWidget(self.mainGraphicsWindow)


        # self.addSubWindow(self.mainGraphicsWindow)

        # MDIFIX
        mdiSubWindow = self.addSubWindow(self.mainGraphicsWindow)
        # mdiSubWindow = subWindow

        # mdiSubWindow = self.mainGraphicsWindow

        # mdiSubWindow = self.addSubWindow(self.mainGraphicsWindow)



        self.mainMdiSubWindow = mdiSubWindow
        #        print MODULENAME,'-------- type(mdiSubWindow)= ',type(mdiSubWindow)  # =  <class 'PyQt4.QtGui.QMdiSubWindow'>
        #        print MODULENAME,'-------- dir(mdiSubWindow)= ',dir(mdiSubWindow)
        self.mainGraphicsWindow.show()
        self.mainGraphicsWindow.setConnects(self)
        #        mdiSubWindow.setGeometry(100,100,400,300)  # rwh: this is how we would specify the position/size of the window


        # MDIFIX
        # self.windowCounter += 1

        # self.addGraphicsWindowToWindowRegistry(self.mainGraphicsWindow)
        # self.addMDIWindowToRegistry(mdiSubWindow)

        # self.graphicsWindowVisDict = {}  # re-init this dict
        # print MODULENAME, '--------- addVTKWindowToWorkspace: graphicsWindowDict= ',self.graphicsWindowDict

        # # # self.windowDict[self.windowCounter] = self.mainGraphicsWindow
        # # # self.graphicsWindowDict[self.windowCounter] = self.mainGraphicsWindow        
        # # # self.mdiWindowDict[self.windowCounter] = mdiSubWindow
        #        print MODULENAME, '--------- addVTKWindowToWorkspace: mdiWindowDict= ',self.mdiWindowDict

        # self.windowDict[self.windowCounter]=mdiSubWindow
        # self.graphicsWindowDict[self.windowCounter]=mdiSubWindow

        # MDIFIX
        # self.mainGraphicsWindow.setWindowTitle("Main Graphics Window " + str(self.windowCounter))

        self.lastActiveWindow = self.mainGraphicsWindow

        #MDIFIX
        self.lastActiveRealWindow = mdiSubWindow

        self.setActiveSubWindowCustomSlot(self.mainGraphicsWindow)  # rwh: do this to "check" this in the "Window" menu


        self.updateWindowMenu()
        self.updateActiveWindowVisFlags()
        print self.graphicsWindowVisDict


    # def removeAllVTKWindows(self, _leaveFirstWindowFlag=False):
    #
    #     windowNames = self.graphicsWindowDict.keys()
    #     print 'windowNames=', windowNames
    #     print 'self.graphicsWindowDict=', self.graphicsWindowDict
    #
    #     if self.MDI_ON:
    #
    #         for windowName in windowNames:
    #             if _leaveFirstWindowFlag and windowName == 1:
    #                 # print "leaving first window"
    #                 continue
    #
    #             self.setActiveSubWindow(self.mdiWindowDict[windowName])
    #             self.closeActiveSubWindowSlot()
    #
    #             continue
    #     else:
    #         for windowName, window in self.graphicsWindowDict.iteritems():
    #             if _leaveFirstWindowFlag and windowName == 1:
    #                 # print "leaving first window"
    #                 continue
    #
    #                 window.close()
    #                 self.removeWindowFromRegistry(window)
    #
    #     print 'GOT HERE REMOVE ALL VTK'
    #     print 'self.graphicsWindowDict=', self.graphicsWindowDict
    #     #MDIFIX
    #     print 'len(self.subWindowList())=', len(self.subWindowList())
    #
    #
    #
    #     self.updateWindowMenu()


    # def removeAllPlotWindows(self, _leaveFirstWindowFlag=False):
    #
    #     windowNames = self.plotWindowDict.keys()
    #     if self.MDI_ON:
    #         for windowName in windowNames:
    #             # print "windowName=",windowName
    #             # print "self.graphicsWindowDict=",self.graphicsWindowDict
    #             if _leaveFirstWindowFlag and windowName == 1:
    #                 # print "leaving first window"
    #                 continue
    #             # self.setActiveSubWindow(self.plotWindowDict[windowName])
    #             self.setActiveSubWindow(self.mdiWindowDict[windowName])
    #             self.closeActiveSubWindowSlot()
    #     else:
    #         for windowName, window in self.plotWindowDict.iteritems():
    #
    #             if _leaveFirstWindowFlag and windowName == 1:
    #                 # print "leaving first window"
    #                 continue
    #             window.close()
    #             # MDIFIX - have to see how to handle plots and plots registry...
    #
    #             # self.removeWindowFromRegistry(window)
    #             # self.closeActiveSubWindowSlot()
    #
    #
    #     self.updateWindowMenu()
    #     self.plotWindowDict = {}
    #     self.plotManager.reset()
    #     # from PlotManagerSetup import createPlotManager
    #     # self.plotManager=createPlotManager(self) # object that is responsible for creating and managing plot windows sdo that they are accessible from steppable level

    # def removeAuxiliaryGraphicsWindows(self):
    #     self.removeAllVTKWindows(True)

    # def saveWindowsGeometry(
    #         self):  # want mdiWindowDict (PyQt4.QtGui.QMdiSubWindow), NOT windowDict (Graphics.GraphicsFrameWidget.GraphicsFrameWidget)
    #     print MODULENAME, '--------> windows.xml'
    #     #        print MODULENAME,'self.mdiWindowDict=',self.mdiWindowDict
    #     fpout = open("windows.xml", "w")
    #     fpout.write('<Windows>\n')
    #
    #     #        print 'dir()=', dir(self.mdiWindowDict[self.mdiWindowDict.keys()[0]])
    #     #        for windowName in self.graphicsWindowDict.keys():
    #     for windowName in self.mdiWindowDict.keys():
    #         #            print 'windowName=', windowName
    #         #            print 'windowTitle=', self.mdiWindowDict[windowName].windowTitle()
    #         line = '    <Window Name="%s">\n' % self.mdiWindowDict[windowName].windowTitle()
    #         fpout.write(line)
    #         #            print 'mdi x,y=', self.mdiWindowDict[windowName].pos().x(),self.mdiWindowDict[windowName].pos().y()
    #         #            print 'parentWidget pos=', self.windowDict[windowName].parentWidget().pos()
    #         #            print 'type(self.windowDict[windowName])=', type(self.windowDict[windowName])
    #         #            print 'type(self.mdiWindowDict[windowName])=', type(self.mdiWindowDict[windowName])
    #         #            print 'mdi width,height=', self.mdiWindowDict[windowName].geometry().width(), self.mdiWindowDict[windowName].geometry().height()
    #         line = '    <Location x="%d" y="%d"/>\n' % (
    #             self.mdiWindowDict[windowName].pos().x(), self.mdiWindowDict[windowName].pos().y())
    #         fpout.write(line)
    #         line = '    <Size width="%d" height="%d"/>\n' % (
    #             self.mdiWindowDict[windowName].geometry().width(), self.mdiWindowDict[windowName].geometry().height())
    #         fpout.write(line)
    #         fpout.write('  </Window>\n')
    #
    #     fpout.write('</Windows>\n')
    #     fpout.close()

    def minimizeAllGraphicsWindows(self):
        if not self.MDI_ON: return

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showMinimized()


        # for windowName in self.graphicsWindowDict.keys():
        #     self.windowDict[windowName].showMinimized()

    def restoreAllGraphicsWindows(self):

        if not self.MDI_ON: return

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showNormal()

        # for windowName in self.graphicsWindowDict.keys():
        #     self.windowDict[windowName].showNormal()

    def closeActiveSubWindowSlot(self):  # this method is called whenever a user closes a graphics window

        print '\n\n\n BEFORE  closeActiveSubWindowSlot self.subWindowList().size()=', len(self.subWindowList())

        # print MODULENAME,"   ----- closeActiveSubWindowSlot()"
        activeWindow = self.activeSubWindow()
        if not activeWindow: return

        activeWindow.close()
        self.removeWindowFromRegistry(activeWindow)

        # print "activeWindow=",activeWindow.widget()
        # print "self.windowDict[1]=",self.windowDict[1]
        # print "self.graphicsWindowDict[1]=",self.graphicsWindowDict[1]
        # print MODULENAME,"closeActiveSubWindowSlot():   self.windowDict.keys()=",self.windowDict.keys()



        # # # for windowName in self.windowDict.keys():
        # # # # print MODULENAME,"closeActiveSubWindowSlot():   windowName=",windowName    #  = 1,2, etc
        # # # if self.windowDict[windowName] == activeWindow.widget():
        # # # # print 'self.removeSubWindow(activeWindow.widget())'
        # # # # activeWindow.widget().deleteLater()
        # # # # activeWindow.deleteLater()

        # # # del self.windowDict[windowName]
        # # # if windowName in self.graphicsWindowDict.keys():
        # # # del self.graphicsWindowDict[windowName]
        # # # del self.mdiWindowDict[windowName]
        # # # activeWindow.close()

        # # # # self.windowCounter -= 1

        self.updateWindowMenu()

        # print 'self.windowDict=', self.windowDict
        print 'AFTER closeActiveSubWindowSlot self.subWindowList().size()=', len(self.subWindowList())


    def processCommandLineOptions(self, opts):  # parse the command line (rf. player/compucell3d.pyw now)
        #        import getopt
        self.__screenshotDescriptionFileName = ""
        self.customScreenshotDirectoryName = ""
        startSimulation = False

        #        print MODULENAME,"-----------  processCommandLineOptions():  opts=",opts

        #        opts=None
        #        args=None
        #        try:
        #            #  NOTE: need ending ":" on single letter options string!
        #            opts, args = getopt.getopt(self.__parent.argv, "h:i:s:o:p:", ["help","noOutput","exitWhenDone","port=","tweditPID=","currentDir=","prefs="])
        #            print "opts=",opts
        #            print "args=",args
        #        except getopt.GetoptError, err:
        #            print str(err) # will print something like "option -a not recognized"
        #            # self.usage()
        #            sys.exit(2)
        output = None
        verbose = False
        currentDir = ""
        port = -1
        tweditPID = -1
        # connectTwedit=False
        self.__prefsFile = "cc3d_default"  # default name of QSettings .ini file (in ~/.config/Biocomplexity on *nix)
        for o, a in opts:
            print "o=", o
            print "a=", a
            if o in ("-i"):  # input file (e.g.  .dml for pre-dumped vtk files)
                self.__fileName = a
                startSimulation = True
            elif o in ("-h", "--help"):
                self.usage()
                sys.exit()
            elif o in ("-s"):
                self.__screenshotDescriptionFileName = a
            elif o in ("-o"):
                self.customScreenshotDirectoryName = a
                self.__imageOutput = True
            elif o in ("-p"):
                print ' handling - (playerSettings, e.g. camera)... a = ', a
                self.playerSettingsFileName = a
                print MODULENAME, 'self.playerSettingsFileName= ', self.playerSettingsFileName
            elif o in ("--noOutput"):
                self.__imageOutput = False
            elif o in ("--currentDir"):
                currentDir = a
                print "currentDirectory=", currentDir
            elif o in ("-w"):  # assume parameter is widthxheight smashed together, e.g. -w 500x300
                winSizes = a.split('x')
                print MODULENAME, "  winSizes=", winSizes
                width = int(winSizes[0])
                height = int(winSizes[1])
                Configuration.setSetting("GraphicsWinWidth", width)
                Configuration.setSetting("GraphicsWinHeight", height)

            elif o in ("--port"):
                port = int(a)
                print "port=", port
            elif o in ("--prefs"):
                self.__prefsFile = a
                print MODULENAME, '---------  doing QSettings ---------  prefsFile=', self.__prefsFile
                Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Biocomplexity",
                                                     self.__prefsFile)
                Configuration.setSetting("PreferencesFile", self.__prefsFile)

                # elif o in ("--tweditPID"):
                # tweditPID=int(a)
                # print "tweditPID=",tweditPID

            elif o in ("--exitWhenDone"):
                self.closePlayerAfterSimulationDone = True
            elif o in (
                    "--guiScan"):  # when user uses gui to do parameter scan all we have to do is to set self.closePlayerAfterSimulationDone to True
                self.closePlayerAfterSimulationDone = True
                # we reset max number of consecutive runs to 1 because we want each simulation in parameter scan
                # initiated by the psrun.py script to be an independent run after which player gets closed and reopened again for the next run
                self.maxNumberOfConsecutiveRuns = 1

                pass
            elif o in ("--maxNumberOfRuns"):
                self.maxNumberOfConsecutiveRuns = int(a)


                # elif o in ("--connectTwedit"):
                # connectTwedit=True 
            else:
                assert False, "unhandled option"

        # import UI.ErrorConsole
        # self.UI.console.getSyntaxErrorConsole().closeCC3D.connect(qApp.closeAllWindows)

        self.UI.console.getSyntaxErrorConsole().setPlayerMainWidget(self)
        self.connect(self.UI.console.getSyntaxErrorConsole(), SIGNAL("closeCC3D()"), qApp.closeAllWindows)
        # establishConnection starts twedit and hooks it up via sockets to player
        self.connect(self.tweditAct, SIGNAL("triggered()"),
                     self.UI.console.getSyntaxErrorConsole().cc3dSender.establishConnection)
        #        print MODULENAME,"    self.UI.console=",self.UI.console
        if port != -1:
            self.UI.console.getSyntaxErrorConsole().cc3dSender.setServerPort(port)

            # if tweditPID != -1:
            # self.UI.console.getSyntaxErrorConsole().cc3dSender.setTweditPID(tweditPID)

            # if connectTwedit:
            # self.UI.console.getSyntaxErrorConsole().cc3dSender.connectTwedit
        # ...
        # checking if file path needs to be remapped to point to files in the directories from which run script was called
        simFileFullName = os.path.join(currentDir, self.__fileName)
        if startSimulation:
            if os.access(simFileFullName, os.F_OK):  # checking if such a file exists
                self.__fileName = simFileFullName
                print "self.__fileName=", self.__fileName
                import CompuCellSetup

                CompuCellSetup.simulationFileName = self.__fileName

            elif not os.access(self.__fileName, os.F_OK):
                assert False, "Could not find simulation file: " + self.__fileName
            from os.path import basename

            self.__parent.setWindowTitle(self.trUtf8(basename(self.__fileName) + " - CompuCell3D Player"))

        if self.__screenshotDescriptionFileName != "":
            screenshotDescriptionFullFileName = os.path.abspath(self.__screenshotDescriptionFileName)
            if os.access(screenshotDescriptionFullFileName, os.F_OK):  # checking if such a file exists
                self.__screenshotDescriptionFileName = screenshotDescriptionFullFileName
            elif os.access(self.__screenshotDescriptionFileName):  # checking if such a file exists
                assert False, "Could not find screenshot Description file: " + self.__screenshotDescriptionFileName

        if self.playerSettingsFileName != "":
            playerSettingsFullFileName = os.path.abspath(self.playerSettingsFileName)
            if os.access(playerSettingsFullFileName, os.F_OK):  # checking if such a file exists
                self.playerSettingsFileName = playerSettingsFullFileName
                print MODULENAME, '(full) playerSettings filename=', self.playerSettingsFileName
            else:
                assert False, "Could not find playerSettings file: " + self.playerSettingsFileName

        # print "This is startSimulation=",startSimulation
        if startSimulation:
            self.__runSim()

    def usage(self):
        print "\n--------------------------------------------------------"
        print "USAGE: ./compucell3d.sh -i <sim file (.cc3d or .xml or .py)> -s <ScreenshotDescriptionFile> -o <custom outputDirectory>"
        print "-w <widthxheight of graphics window>"
        print "--exitWhenDone   close the player after simulation is done"
        print "--noOutput   ensure that no screenshots are stored regardless of Player settings"
        print "--prefs    name of preferences file to use/save"
        print "-p    playerSettingsFileName (e.g. 3D camera settings)"
        print "-h or --help   print (this) help message"
        print "\ne.g.  compucell3d.sh -i Demos/cellsort_2D/cellsort_2D/cellsort_2D.cc3d -w 500x500 --prefs myCellSortPrefs"

    def setRecentSimulationFile(self, _fileName):
        self.__fileName = _fileName
        from os.path import basename

        self.__parent.setWindowTitle(self.trUtf8(basename(self.__fileName) + " - CompuCell3D Player"))
        import CompuCellSetup

        CompuCellSetup.simulationFileName = self.__fileName

    def resetControlButtonsAndActions(self):
        self.runAct.setEnabled(True)
        self.stepAct.setEnabled(True)
        self.pauseAct.setEnabled(False)
        self.stopAct.setEnabled(False)
        self.openAct.setEnabled(True)
        self.openLDSAct.setEnabled(True)
        self.pifFromSimulationAct.setEnabled(False)
        self.pifFromVTKAct.setEnabled(False)

    def resetControlVariables(self):

        self.steppingThroughSimulation = False
        self.cmlHandlerCreated = False

        CompuCellSetup.simulationFileName = ""

        self.drawingAreaPrepared = False
        self.simulationIsRunning = False

        self.newDrawingUserRequest = False
        self.completedFirstMCS = False

    def prepareForNewSimulation(self, _forceGenericInitialization=False, _inStopFcn=False):
        """
        This function creates new instance of computational thread and sets various flags to initial values i.e. to a state before the beginnig of the simulations
        """
        self.resetControlButtonsAndActions()

        self.steppingThroughSimulation = False

        #        import CompuCellSetup
        CompuCellSetup.viewManager = self
        CompuCellSetup.simulationFileName = ""

        # print 'INSIDE PREPARE FOR NEWSIMULATION'
        # time.sleep(2)


        # if not _inStopFcn:


        from BasicSimulationData import BasicSimulationData

        self.basicSimulationData = BasicSimulationData()

        # this import has to be here not inside is statement to ensure that during switching from playing one type of files to another there is no "missing module" issue due to imoprer imports
        # import CMLResultReader 
        from Simulation.CMLResultReader import CMLResultReader

        self.cmlHandlerCreated = False

        # this is used to perform generic preparation for new simulation , normally called after "stop". If users decide to use *.dml  prepare simulation will be called again with False argument
        if _forceGenericInitialization:
            CompuCellSetup.playerType = "new"

        # print '_forceGenericInitialization=',_forceGenericInitialization
        if CompuCellSetup.playerType == "CMLResultReplay":
            self.__viewManagerType = "CMLResultReplay"

            # note that this variable will be the same as self.simulation when doing CMLReplay mode. I keep it under diffferent name to keep track of the places in the code where I am using SimulationThread API and where I use CMLResultReade replay part of the API                
            # this means that further refactoring is needed but I leave it for now
            self.cmlReplayManager = self.simulation = CMLResultReader(self)

            # print "GOT THIS self.__fileName=",self.__fileName
            self.simulation.extractLatticeDescriptionInfo(self.__fileName)
            #filling out basic simulation data
            self.basicSimulationData.fieldDim = self.simulation.fieldDim
            self.basicSimulationData.numberOfSteps = self.simulation.numberOfSteps

            #old connections
            # self.connect(self.simulation,SIGNAL("simulationInitialized(bool)"),self.initializeSimulationViewWidget)
            # self.connect(self.simulation,SIGNAL("steppablesStarted(bool)"),self.runSteppablePostStartPlayerPrep)            
            # self.connect(self.simulation,SIGNAL("simulationFinished(bool)"),self.handleSimulationFinished)
            # self.connect(self.simulation,SIGNAL("completedStep(int)"),self.handleCompletedStep)


            self.cmlReplayManager.initial_data_read.connect(self.initializeSimulationViewWidget)
            self.cmlReplayManager.subsequent_data_read.connect(self.handleCompletedStep)
            self.cmlReplayManager.final_data_read.connect(self.handleSimulationFinished)

            import PlayerPython

            self.fieldExtractor = PlayerPython.FieldExtractorCML()
            self.fieldExtractor.setFieldDim(self.basicSimulationData.fieldDim)

        else:
            self.__viewManagerType = "Regular"
            #            import CompuCellSetup
            # print MODULENAME,'prepareForNewSimulation(): setting cmlFieldHandler = None'
            CompuCellSetup.cmlFieldHandler = None  # have to reinitialize cmlFieldHandler to None

            # print 'BEFORE CONSTRUCTING NEW SIMULATINO THREAD'
            self.simulation = SimulationThread(self)
            # print 'AFTER CONSTRUCTING NEW SIMULATINO THREAD'



            self.connect(self.simulation, SIGNAL("simulationInitialized(bool)"), self.initializeSimulationViewWidget)
            self.connect(self.simulation, SIGNAL("steppablesStarted(bool)"), self.runSteppablePostStartPlayerPrep)
            self.connect(self.simulation, SIGNAL("simulationFinished(bool)"), self.handleSimulationFinished)
            self.connect(self.simulation, SIGNAL("completedStep(int)"), self.handleCompletedStep)
            self.connect(self.simulation, SIGNAL("finishRequest(bool)"), self.handleFinishRequest)

            # self.connect(self.plotManager,SIGNAL("newPlotWindow(bool)"),self.addNewPlotWindow)
            self.plotManager.initSignalAndSlots()

            import PlayerPython
            # print  'BEFORE FIELD STORAGE'
            # time.sleep(5)
            self.fieldStorage = PlayerPython.FieldStorage()
            self.fieldExtractor = PlayerPython.FieldExtractor()
            self.fieldExtractor.setFieldStorage(self.fieldStorage)
            # print  'AFTER FIELD STORAGE'
            # time.sleep(5)

        self.simulation.setCallingWidget(self)

        """Commented out"""
        # if not _forceGenericInitialization:    
        # self.simulation.setCallingWidget(self)
        # self.__setupArea()
        self.resetControlVariables()

        # self.drawingAreaPrepared=False
        # self.simulationIsRunning=False

        # self.newDrawingUserRequest=False
        # self.completedFirstMCS=False



        # sys.exit()

    # MDIFIX
    def __setupArea(self):
        # print '------------------- __setupArea'
        # time.sleep(5)
        # print 'before removeAllVTKWindows'
        self.close_all_windows()
        # self.windowCounter = 0

        self.addVTKWindowToWorkspace()

        # print 'after addVTKWindowToWorkspace'
        # print 'AFTER ------------------- __setupArea'
        # time.sleep(5)

    # def __setupArea(self):
    #     # print '------------------- __setupArea'
    #     # time.sleep(5)
    #     # print 'before removeAllVTKWindows'
    #     self.removeAllVTKWindows()
    #     # print 'after removeAllVTKWindows'
    #     self.removeAllPlotWindows()
    #     self.windowCounter = 0
    #     # print 'before addVTKWindowToWorkspace'
    #     self.addVTKWindowToWorkspace()
    #
    #     # print 'after addVTKWindowToWorkspace'
    #     # print 'AFTER ------------------- __setupArea'
    #     # time.sleep(5)



        # print MODULENAME,'    __setupArea():   self.mdiWindowDict=',self.mdiWindowDict

    # def __layoutGraphicsWindows(self):
    #     # rwh: if user specified a windows layout .xml in the .cc3d, then use it
    #     #        print MODULENAME,'    __layoutGraphicsWindows():   self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict=',self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict
    #     #--> e.g. = {'Aux Graphics Window 2': [450, 100, 300, 250], 'Main Graphics Window 1': [10, 10, 400, 300]}
    #     #        self.mdiWindowDict[self.windowCounter] = self.addSubWindow(newWindow)
    #     #        print MODULENAME,'    __layoutGraphicsWindows():   self.mdiWindowDict=',self.mdiWindowDict  # {1: <PyQt4.QtGui.QMdiSubWindow object at 0x11ff415f0>}
    #     if self.cc3dSimulationDataHandler is None:
    #         return
    #     windowIndex = 1
    #     for key in self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict.keys():
    #         winGeom = self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict[key]
    #         if 'Main' in key:
    #             #                print MODULENAME,'    __layoutGraphicsWindows():   resizing Main window'
    #             self.mdiWindowDict[1].setGeometry(winGeom[0], winGeom[1], winGeom[2], winGeom[3])
    #         elif 'Aux' in key:
    #             windowIndex += 1
    #             #                print MODULENAME,'    __layoutGraphicsWindows():   need to create/resize Aux window'
    #             self.addNewGraphicsWindow()  #rwh
    #             #                print MODULENAME,'    __layoutGraphicsWindows():   after addNewGraphicsWindow, mdiWindowDict=',self.mdiWindowDict
    #             self.mdiWindowDict[windowIndex].setGeometry(winGeom[0], winGeom[1], winGeom[2], winGeom[3])


    def handleErrorMessage(self, _errorType, _traceback_message):
        msg = QMessageBox.warning(self, _errorType, \
                                  _traceback_message, \
                                  QMessageBox.Ok,
                                  QMessageBox.Ok)

        # # # self.__stopSim()
        #         print 'INSIDE HANDLE ERROR MESSAGE'
        #         print '_errorType=',_errorType
        #         print '_traceback_message=',_traceback_message
        import ParameterScanEnums

        if _errorType == 'Assertion Error' and _traceback_message.startswith(
                        'Parameter Scan ERRORCODE=' + str(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)):
            self.__cleanAfterSimulation(_exitCode=ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)
        else:
            self.__cleanAfterSimulation()
            print 'errorType=', _errorType
            syntaxErrorConsole = self.UI.console.getSyntaxErrorConsole()
            text = "Search \"file.xml\"\n"
            text += "    file.xml\n"
            text += _traceback_message
            syntaxErrorConsole.setText(text)

            # def handleErrorMessageDetailed(self,_errorType,_file,_line,_col,_traceback_message):
            # print "INSIDE handleErrorMessageDetailed"
            # self.__stopSim()
            # syntaxErrorConsole=self.UI.console.getSyntaxErrorConsole()

            # text="Error:"+_errorType+"\n"
            # text+="  File: "+_file+"\n"
            # text+="    Line: "+str(_line)+" col: "+str(_col)+" "+_traceback_message+"\n\n\n\n"
            # syntaxErrorConsole.setText(text)
            # return

    def handleErrorFormatted(self, _errorMessage):
        # print "INSIDE handleErrorFormatted"
        # # # self.__stopSim()
        self.__cleanAfterSimulation()
        syntaxErrorConsole = self.UI.console.getSyntaxErrorConsole()

        syntaxErrorConsole.setText(_errorMessage)
        self.UI.console.bringUpSyntaxErrorConsole()
        return

    def processIncommingSimulation(self, _fileName, _stopCurrentSim=False):
        print "processIncommingSimulation = ", _fileName, ' _stopCurrentSim=', _stopCurrentSim
        if _stopCurrentSim:
            startNewSimulation = False
            if not self.simulationIsRunning and not self.simulationIsStepping:
                startNewSimulation = True

            self.__stopSim()

            import os

            self.__fileName = os.path.abspath(str(_fileName))  # normalizing path
            import CompuCellSetup

            CompuCellSetup.simulationFileName = self.__fileName

            # import time
            # time.sleep(1.0)
            # print 'startNewSimulation=',startNewSimulation
            if startNewSimulation:
                self.__runSim()
        else:
            self.__fileName = _fileName
            self.nextSimulation = _fileName

        from os.path import basename

        self.__parent.setWindowTitle(self.trUtf8(basename(str(_fileName)) + " - CompuCell3D Player"))

    #    def prepareXMLTreeView(self,_xmlFileName):
    def prepareXMLTreeView(self):
        import XMLUtils
        #        import CompuCellSetup

        _simulationFileName = "D:\Program Files\COMPUCELL3D_3.4.0\Demos\cellsort_2D\cellsort_2D.xml"

        self.root_element = CompuCellSetup.cc3dXML2ObjConverter.root
        self.model = SimModel(self.root_element, self.__modelEditor)
        self.simulation.setSimModel(
            self.model)  # hook in simulation thread class to XML model TreeView panel in the GUI - needed for steering

        # self.model.checkSanity()

        self.__modelEditor.setModel(self.model)
        #        print MODULENAME,' --------- prepareXMLTreeView(self):'
        #        import pdb; pdb.set_trace()
        self.model.setPrintFlag(True)

    def prepareLatticeDataView(self):
        self.__parent.latticeDataModel.setLatticeDataFileList(self.simulation.ldsFileList)
        self.latticeDataModel = self.__parent.latticeDataModel
        self.__parent.latticeDataModelTable.setModel(
            self.__parent.latticeDataModel)  # this sets up the model and actually displays model data- so use this function when model is ready to be used
        self.__parent.latticeDataModelTable.setParams()
        self.latticeDataModelTable = self.__parent.latticeDataModelTable


    def __loadSim(self, file):
        # resetting reference to SimulationDataHandler

        self.prepareForNewSimulation(_forceGenericInitialization=True)

        self.cc3dSimulationDataHandler = None

        fileName = str(self.__fileName)
        # print 'INSIDE LOADSIM file=',fileName
        #        print MODULENAME,"Load file ",fileName
        self.UI.console.bringUpOutputConsole()

        # have to connect error handler to the signal emited from self.simulation object
        self.connect(self.simulation, SIGNAL("errorOccured(QString,QString)"), self.handleErrorMessage)
        # self.connect(self.simulation,SIGNAL("errorOccuredDetailed(QString,QString,int,int,QString)"),self.handleErrorMessageDetailed)
        self.connect(self.simulation, SIGNAL("errorFormatted(QString)"), self.handleErrorFormatted)

        # We need to create new SimulationPaths object for each new simulation.    
        #        import CompuCellSetup
        CompuCellSetup.simulationPaths = CompuCellSetup.SimulationPaths()

        if re.match(".*\.xml$", fileName):  # If filename ends with .xml
            # print "GOT FILE ",fileName
            # self.prepareForNewSimulation()
            self.simulation.setRunUserPythonScriptFlag(True)
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(fileName)
            pythonScriptName = CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)

            # import xml
            # try:
            # pythonScriptName=CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)
            # except xml.parsers.expat.ExpatError,e:
            # import CompuCellSetup
            # xmlFileName=CompuCellSetup.simulationPaths.simulationXMLFileName
            # print "Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message
            # import sys
            # return
            # sys.exit()

            if pythonScriptName != "":
                CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(pythonScriptName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)


            # if self.__parent.latticeDataDock.isVisible():
            #     self.__parent.latticeDataAct.trigger(False)
            #
            # if self.__parent.modelEditorDock.isHidden():
            #     self.__parent.modelAct.trigger(True)

        elif re.match(".*\.py$", fileName):
            globals = {'simTabView': 20}
            locals = {}
            self.simulation.setRunUserPythonScriptFlag(True)

            # NOTE: extracting of xml file name from python script is done during script run time so we cannot use CompuCellSetup.simulationPaths.setXmlFileNameFromPython function here
            CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(self.__fileName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

            # if self.__parent.latticeDataDock.isVisible():
            #     self.__parent.latticeDataAct.trigger(True)
            #
            # if self.__parent.modelEditorDock.isHidden():
            #     self.__parent.modelAct.trigger(False)

        elif re.match(".*\.cc3d$", fileName):
            self.__loadCC3DFile(fileName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

            # if self.__parent.latticeDataDock.isVisible():
            #     self.__parent.latticeDataAct.trigger(False)
            #
            # if self.__parent.modelEditorDock.isHidden():
            #     self.__parent.modelAct.trigger(True)

                #self.prepareForNewSimulation()   # rwh: do this?

        elif re.match(".*\.dml$", fileName):
            # Let's toggle these off (and not tell the user for now)
            #            Configuration.setSetting("ImageOutputOn",False)  # need to make it possible to save images from .dml/vtk files
            if Configuration.getSetting("LatticeOutputOn"):
                QMessageBox.warning(self, "Message",
                                    "Warning: Turning OFF 'Save lattice...' in Preferences",
                                    QMessageBox.Ok)
                print '-----------------------'
                print '  WARNING:  Turning OFF "Save lattice" in Preferences|Output'
                print '-----------------------'
                Configuration.setSetting("LatticeOutputOn", False)
            #            self.dlg.enableLatticeOutput(False)   # disable the Lattice Output toggle (doesn't seem to work)

            #        if (self.lastActiveWindow is not None) and (self.lastActiveWindow.winId() in self.graphicsWindowVisDict.keys()):
            #            self.cellsAct.setChecked(self.graphicsWindowVisDict[self.lastActiveWindow.winId()][0])
            #            self.borderAct.setChecked(self.graphicsWindowVisDict[self.lastActiveWindow.winId()][1])
            #            self.clusterBorderAct.setChecked(self.graphicsWindowVisDict[self.lastActiveWindow.winId()][2])
            #            self.cellGlyphsAct.setChecked(self.graphicsWindowVisDict[self.lastActiveWindow.winId()][3])
            #            self.FPPLinksAct.setChecked(self.graphicsWindowVisDict[self.lastActiveWindow.winId()][4])

            if Configuration.getSetting("CellGlyphsOn"):
                QMessageBox.warning(self, "Message",
                                    "Warning: Turning OFF 'Vis->Cell Glyphs' ",
                                    QMessageBox.Ok)
                print '-----------------------'
                print '  WARNING:  Turning OFF "Vis->Cell Glyphs"'
                print '-----------------------'
                Configuration.setSetting("CellGlyphsOn", False)
                #                self.graphicsWindowVisDict[self.lastActiveWindow.winId()][3] = False
                self.cellGlyphsAct.setChecked(False)

            if Configuration.getSetting("FPPLinksOn"):
                QMessageBox.warning(self, "Message",
                                    "Warning: Turning OFF 'Vis->FPP Links' ",
                                    QMessageBox.Ok)
                print '-----------------------'
                print '  WARNING:  Turning OFF "Vis->FPP Links"'
                print '-----------------------'
                Configuration.setSetting("FPPLinksOn", False)
                #                self.graphicsWindowVisDict[self.lastActiveWindow.winId()][4] = False
                self.FPPLinksAct.setChecked(False)

            CompuCellSetup.playerType = "CMLResultReplay"

            self.prepareForNewSimulation()

            CompuCellSetup.simulationPaths.setSimulationResultDescriptionFile(fileName)

            self.__parent.toggleLatticeData(True)
            self.__parent.toggleModelEditor(False)

            self.prepareLatticeDataView()

        Configuration.setSetting("RecentFile", os.path.abspath(self.__fileName))
        Configuration.setSetting("RecentSimulations", os.path.abspath(self.__fileName))  #  each loaded simulation has to be passed to a function which updates list of recent files
        # Configuration.addItemToStrlist(item = os.path.abspath(self.__fileName),strListName = 'RecentSimulations',maxLength = Configuration.getSetting('NumberOfRecentSimulations'))


        # if self.saveSettings:
        # Configuration.syncPreferences()

        # recentSim=Configuration.getSetting("RecentSimulations")     
        # print 'recent sim=',recentSim
        # sys.exit()


    #        print MODULENAME,'__loadSim():  on exit,  self.graphicsWindowVisDict=',self.graphicsWindowVisDict



    def __loadCC3DFile(self, fileName):
        #        import CompuCellSetup

        import CC3DSimulationDataHandler

        self.cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(self)
        try:
            f = open(fileName, 'r')
            f.close()
        except IOError, e:
            msg = QMessageBox.warning(self, "Not A Valid Simulation File", \
                                      "Please make sure <b>%s</b> exists" % fileName, \
                                      QMessageBox.Ok)
            raise IOError("%s does not exist" % fileName)

        self.cc3dSimulationDataHandler.readCC3DFileFormat(fileName)
        #check if current CC3D version is greater or equal to the version (minimal required version) specified in the project
        import Version

        currentVersion = Version.getVersionAsString()
        currentVersionInt = currentVersion.replace('.', '')
        projectVersion = self.cc3dSimulationDataHandler.cc3dSimulationData.version
        projectVersionInt = projectVersion.replace('.', '')
        print 'projectVersion=', projectVersion
        print 'currentVersion=', currentVersion

        if int(projectVersionInt) > int(currentVersionInt):
            msg = QMessageBox.warning(self, "CompuCell3D Version Mismatch", \
                                      "Your CompuCell3D version <b>%s</b> might be too old for the project you are trying to run. The least version project requires is <b>%s</b>. You may run project at your own risk" % (
                                          currentVersion, projectVersion), \
                                      QMessageBox.Ok)

        if self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource:
            self.customSettingPath = self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource.path
            print 'GOT CUSTOM SETTINGS RESOURCE = ', self.customSettingPath

            # # # Configuration.readCustomFile(self.customSettingPath)
            Configuration.initializeCustomSettings(self.customSettingPath)
            self.__paramsChanged()

            # Configuration.getSetting('PlayerSizes')
            # sys.exit()
        else:
            self.customSettingPath = os.path.abspath(
                os.path.join(self.cc3dSimulationDataHandler.cc3dSimulationData.basePath, 'Simulation/_settings.xml'))
            # Configuration.writeCustomFile(self.customSettingPath)
            Configuration.writeSettingsForSingleSimulation(self.customSettingPath)

        # sys.exit()

        if self.cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource:

            cc3dProjectDir = os.path.dirname(fileName)
            paramScanXMLFileName = self.cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path

            # checking if simulation file directory is writeable if not parameterscan cannot run properly - writeable simulation fiel directory is requirement for parameter scan
            if not os.access(cc3dProjectDir, os.W_OK):
                raise AssertionError(
                    'parameter Scan Error: CC3D project directory:' + cc3dProjectDir + ' has to be writeable. Please change permission on the directory of the .cc3d project')
            # check if parameter scan file is writeable
            if not os.access(paramScanXMLFileName, os.W_OK):
                raise AssertionError(
                    'parameter Scan Error: Parameter Scan xml file :' + paramScanXMLFileName + ' has to be writeable. Please change permission on this file')

            try:

                from FileLock import FileLock

                with FileLock(file_name=fileName, timeout=10, delay=0.05)  as flock:

                    self.singleSimulation = False
                    self.parameterScanFile = self.cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path  # parameter scan file path
                    pScanFilePath = self.parameterScanFile
                    # We use separate ParameterScanUtils object to handle parameter scan
                    from ParameterScanUtils import ParameterScanUtils

                    psu = ParameterScanUtils()

                    psu.readParameterScanSpecs(pScanFilePath)

                    paramScanSpecsDirName = os.path.dirname(pScanFilePath)

                    outputDir = str(Configuration.getSetting('OutputLocation'))

                    customOutputPath = psu.prepareParameterScanOutputDirs(_outputDirRoot=outputDir)
                    # # # print 'customOutputPath=',customOutputPath
                    #FIX ERROR MESSAGE TO INDICATE THE FILE WHICH COULD NOT BE CREATED
                    # # # if not customOutputPath:                    
                    # # # raise AssertionError('Parameter Scan Error: Could not create simulation output directory: '+outputDir)
                    # # # return False,False

                    self.cc3dSimulationDataHandler.copySimulationDataFiles(customOutputPath)


                    #construct path to the just-copied .cc3d file
                    cc3dFileBaseName = os.path.basename(self.cc3dSimulationDataHandler.cc3dSimulationData.path)
                    cc3dFileFullName = os.path.join(customOutputPath, cc3dFileBaseName)

                    # # # print 'cc3dFileFullName=',cc3dFileFullName


                    psu.replaceValuesInSimulationFiles(_pScanFileName=pScanFilePath, _simulationDir=customOutputPath)
                    # save parameter Scan spec file with incremented ityeration
                    psu.saveParameterScanState(_pScanFileName=pScanFilePath)


                    # # # if not customOutputPath:
                    # # # return False,False




                    from os.path import basename

                    self.__parent.setWindowTitle(self.trUtf8('ParameterScan: ') + self.trUtf8(
                        basename(self.__fileName) + self.trUtf8(' Iteration: ') + basename(
                            customOutputPath) + " - CompuCell3D Player"))

                    # read newly created .cc3d file     
                    self.cc3dSimulationDataHandler.readCC3DFileFormat(cc3dFileFullName)

                    # # setting simultaion output dir names
                    self.customScreenshotDirectoryName = customOutputPath
                    CompuCellSetup.screenshotDirectoryName = customOutputPath
                    self.screenshotDirectoryName = customOutputPath
                    self.parameterScanOutputDir = customOutputPath
                    # print 'self.screenshotDirectoryName=',self.screenshotDirectoryName

            except AssertionError, e:  # propagating exception
                raise e

        else:
            self.singleSimulation = True

        #        print MODULENAME,'   __loadCC3DFile:  sim data:'
        print self.cc3dSimulationDataHandler.cc3dSimulationData
        #        print MODULENAME,'   end sim data'

        CompuCellSetup.simulationPaths.setSimulationBasePath(self.cc3dSimulationDataHandler.cc3dSimulationData.basePath)

        if self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":
            self.simulation.setRunUserPythonScriptFlag(True)
            CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(
                self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
            if self.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":
                CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(
                    self.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)

        elif self.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript != "":
            self.simulation.setRunUserPythonScriptFlag(True)
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(
                self.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript)
            # import xml
            # try:
            # pythonScriptName=CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)
            # except xml.parsers.expat.ExpatError,e:
            # import CompuCellSetup
            # xmlFileName=CompuCellSetup.simulationPaths.simulationXMLFileName
            # print "Error in XML File","File:\n "+xmlFileName+"\nhas the following problem\n"+e.message
            # import sys
            # return
            # sys.exit()

            if self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":
                CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(
                    self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)

        if self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript != "":
            #            print MODULENAME,'   __loadCC3DFile:  windowScript=',self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript
            CompuCellSetup.simulationPaths.setPlayerSimulationWindowsFileName(
                self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript)
            self.__windowsXMLFileName = self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript
        #            print MODULENAME,'   __loadCC3DFile:  self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict=',self.cc3dSimulationDataHandler.cc3dSimulationData.windowDict

        # Configuration.setSetting("RecentSimulations",fileName)   # updating recent files

        # self.simulation.setRunUserPythonScriptFlag(True)
        # # NOTE: extracting of xml file name from python script is done during script run time so we cannot use CompuCellSetup.simulationPaths.setXmlFileNameFromPython function here
        # CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(self.__fileName)


    def __dumpPlayerParams(self):  # QShortcut
        fname = 'player.txt'
        print '     self.lastActiveWindow=', self.lastActiveWindow
        f = open(fname, 'w')
        cam = self.lastActiveWindow.camera3D

        print '\n-----------------'
        v = cam.GetPosition()
        s = str(v)
        s = 'CameraPosition= ' + s.replace('(', ' ').replace(')', ' ').replace(',', ' ') + '\n'
        print s,
        f.write(s)

        v = cam.GetFocalPoint()
        s = str(v)
        s = 'CameraFocalPoint= ' + s.replace('(', ' ').replace(')', ' ').replace(',', ' ') + '\n'
        print s,
        f.write(s)

        v = cam.GetViewUp()
        s = str(v)
        s = 'CameraViewUp= ' + s.replace('(', ' ').replace(')', ' ').replace(',', ' ') + '\n'
        print s,
        f.write(s)

        v = cam.GetClippingRange()
        s = str(v)
        s = 'CameraClippingRange= ' + s.replace('(', ' ').replace(')', ' ').replace(',', ' ') + '\n'
        print s,
        f.write(s)

        #        v = cam.GetViewPlaneNormal()  # deprecated; computed automatically
        #        s = str(v)
        #        s = 'ViewPlaneNormal= '+ s.replace('(',' ').replace(')',' ').replace(',',' ') + '\n'
        #        print s,
        #        f.write(s)

        v = cam.GetDistance()
        s = str(v)
        s = 'CameraDistance= ' + s + '\n'
        print s,
        f.write(s)

        v = cam.GetViewAngle()
        s = str(v)
        s = 'ViewAngle= ' + s + '\n'
        print s,
        f.write(s)

        f.close()
        print MODULENAME, '  dumpPlayerParams  --> ', fname
        print '-----------------'


    def __setConnects(self):
        QShortcut(QKeySequence("Ctrl+p"), self, self.__dumpPlayerParams)  # Cmd-3 on Mac
        self.connect(self.runAct, SIGNAL('triggered()'), self.__runSim)
        self.connect(self.stepAct, SIGNAL('triggered()'), self.__stepSim)
        self.connect(self.pauseAct, SIGNAL('triggered()'), self.__pauseSim)
        self.connect(self.stopAct, SIGNAL('triggered()'), self.__simulationStop)

        self.connect(self.addVTKWindowAct, SIGNAL('triggered()'), self.__addVTKWindow)

        self.connect(self.serializeAct, SIGNAL('triggered()'), self.__simulationSerialize)

        self.connect(self.openAct, SIGNAL('triggered()'), self.__openSim)
        self.connect(self.openLDSAct, SIGNAL('triggered()'), self.__openLDSFile)

        self.connect(self.saveAct, SIGNAL('triggered()'), self.__saveSim)
        self.connect(self.saveScreenshotDescriptionAct, SIGNAL('triggered()'), self.__saveScrDesc)
        self.connect(self.openScreenshotDescriptionAct, SIGNAL('triggered()'), self.__openScrDesc)

        # self.connect(self.savePlayerParamsAct, SIGNAL('triggered()'), self.__savePlayerParams)
        self.connect(self.exitAct, SIGNAL('triggered()'),
                     qApp.closeAllWindows)  #qApp is a member of QtGui. closeAllWindows will cause closeEvent and closeEventSimpleTabView will be called
        #        self.connect(self.openPlayerParamsAct,  SIGNAL('triggered()'), self.__openPlayerParams)

        self.connect(self.cellsAct, SIGNAL('triggered(bool)'), self.__checkCells)
        self.connect(self.borderAct, SIGNAL('triggered(bool)'), self.__checkBorder)
        self.connect(self.clusterBorderAct, SIGNAL('triggered(bool)'), self.__checkClusterBorder)
        self.connect(self.cellGlyphsAct, SIGNAL('triggered(bool)'), self.__checkCellGlyphs)
        self.connect(self.FPPLinksAct, SIGNAL('triggered(bool)'), self.__checkFPPLinks)
        #        self.connect(self.FPPLinksColorAct,  SIGNAL('triggered(bool)'),  self.__checkFPPLinksColor)

        # self.connect(self.contourAct,   SIGNAL('triggered(bool)'),  self.__checkContour)
        self.connect(self.limitsAct, SIGNAL('triggered(bool)'), self.__checkLimits)
        self.connect(self.configAct, SIGNAL('triggered()'), self.__showConfigDialog)
        self.connect(self.cc3dOutputOnAct, SIGNAL('triggered(bool)'), self.__checkCC3DOutput)

        self.connect(self.pifFromSimulationAct, SIGNAL('triggered()'), self.__generatePIFFromCurrentSnapshot)
        self.connect(self.pifFromVTKAct,    SIGNAL('triggered()'),      self.__generatePIFFromVTK)

        #window menu actions
        self.connect(self.newGraphicsWindowAct, SIGNAL('triggered()'), self.addNewGraphicsWindow)
        # self.connect(self.newPlotWindowAct,    SIGNAL('triggered()'),      self.addNewPlotWindow)

        self.connect(self.tileAct, SIGNAL('triggered()'), self.tileSubWindows)
        self.connect(self.cascadeAct, SIGNAL('triggered()'), self.cascadeSubWindows)

        # self.connect(self.saveWindowsGeometryAct, SIGNAL('triggered()'), self.saveWindowsGeometry)
        self.connect(self.minimizeAllGraphicsWindowsAct, SIGNAL('triggered()'), self.minimizeAllGraphicsWindows)
        self.connect(self.restoreAllGraphicsWindowsAct, SIGNAL('triggered()'), self.restoreAllGraphicsWindows)

        self.connect(self.closeActiveWindowAct, SIGNAL('triggered()'), self.closeActiveSubWindowSlot)
        # self.connect(self.closeAdditionalGraphicsWindowsAct, SIGNAL('triggered()'), self.removeAuxiliaryGraphicsWindows)

        self.connect(self, SIGNAL('configsChanged'), self.__paramsChanged)


    # Connections that are related to the simulation view
    # Change to Graphics2D or Graphics3D
    def __setSimConnects(self):
        # Set connections is the self.mainGraphicsWindow is instance of Graphics2D 
        if self.mainGraphicsWindow is not None and isinstance(self.mainGraphicsWindow, (Graphics2D)):
            self.connect(self, SIGNAL('configsChanged'), self.mainGraphicsWindow.configsChanged)

    def __addVTKWindow(self):

        self.closeActiveSubWindowSlot()


    def setFieldType(self, _fieldTypeTuple):
        self.__fieldType = _fieldTypeTuple


    def closeEventSimpleTabView(self, event=None):


        if self.saveSettings:
            Configuration.syncPreferences()
            Configuration.writeAllSettings()

            """
            For some reason have to introduce delay to avoid problems with application becoming unresponsive
            """
            # # # import time
            # # # time.sleep(0.5)
            # self.simulation.stop()
            # self.simulation.wait()

            # self.__simulationStop()



            return

            """
            For some reason have to introduce delay to avoid problems with application becoming unresponsive
            """
            import time

            time.sleep(0.5)
            self.simulation.stop()
            self.simulation.wait()

            # # # self.removeAllVTKWindows()
            # # # self.removeAllPlotWindows()


    # # Core method for running simulation
    def printInfo(self):
        # print "INFO"
        self.simulation.sem.acquire()
        self.simulation.sem.release()


    def initializeSimulationViewWidgetCMLResultReplay(self):
        # self.pifFromVTKAct.setEnabled(True)
        self.fieldDim = self.simulation.fieldDim
        self.mysim = self.simulation.sim

        #        print MODULENAME,"--------- initializeSimulationViewWidgetCMLResultReplay()"

        latticeTypeStr = self.simulation.latticeType
        if latticeTypeStr in Configuration.LATTICE_TYPES.keys():
            self.latticeType = Configuration.LATTICE_TYPES[latticeTypeStr]
        else:
            self.latticeType = Configuration.LATTICE_TYPES["Square"]  # default choice

        simulationDataIntAddr = self.extractAddressIntFromVtkObject(self.simulation.simulationData)
        self.fieldExtractor.setSimulationData(simulationDataIntAddr)
        self.simulation.newFileBeingLoaded = False  # this flag is used to prevent calling  draw function when new data is read from hard drive
        # at this moment new data has been read and is ready to be used


        # this fcn will draw initial lattice configuration so data has to be available by then and appropriate pointers set - see line above
        self.prepareSimulationView()

        self.screenshotManager = ScreenshotManager.ScreenshotManager(self)
        self.screenshotNumberOfDigits = len(str(self.basicSimulationData.numberOfSteps))

        # print "self.screenshotManager",self.screenshotManager
        # print "self.__fileName=",self.__fileName

        if self.__screenshotDescriptionFileName != "":
            self.screenshotManager.readScreenshotDescriptionFile(self.__screenshotDescriptionFileName)

        if self.simulationIsStepping:
            # print "BEFORE STEPPING PAUSE"
            self.__pauseSim()


            # after this call I can access self.root_element of the XML File
            # self.loadCustomPlayerSettings(self.root_element)
            # creating simulation directory depending on whether user requests simulation output or not
        #        import CompuCellSetup

        if self.__imageOutput:
            if self.customScreenshotDirectoryName == "":
                #                import CompuCellSetup
                outputDir = str(Configuration.getSetting("OutputLocation"))
                #                print MODULENAME, 'initializeSimulationViewWidgetCMLResultReplay(): outputDir, self.__outputDirectory= ',outputDir, self.__outputDirectory
                (self.screenshotDirectoryName, self.baseScreenshotName) = CompuCellSetup.makeSimDir(self.__fileName,
                                                                                                    outputDir)
                #                print MODULENAME, 'initializeSimulationViewWidgetCMLResultReplay(): self.screenshotDirectoryName= ',self.screenshotDirectoryName
                CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName

            else:
                (self.screenshotDirectoryName, self.baseScreenshotName) = self.makeCustomSimDir(
                    self.customScreenshotDirectoryName, self.__fileName)
                CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName
                if self.screenshotDirectoryName == "":
                    self.__imageOutput = False  # do not output screenshots when custom directory was not created or already exists
                #                print MODULENAME, 'initializeSimulationViewWidgetCMLResultReplay(): self.screenshotDirectoryName= ',self.screenshotDirectoryName

        self.cmlReplayManager.keepGoing()

    def createOutputDirs(self):

        import CompuCellSetup
        #        import pdb; pdb.set_trace()
        if self.customScreenshotDirectoryName == "":
            #            import CompuCellSetup
            #            outputDir = self.__outputDirectory
            #            print MODULENAME,'createOutputDirs():  self.__outputDirectory, self.__outputDirectory= ',self.__outputDirectory, self.__outputDirectory
            (self.screenshotDirectoryName, self.baseScreenshotName) = CompuCellSetup.makeSimDir(self.__fileName,
                                                                                                self.__outputDirectory)
            #                (self.screenshotSubdirName, self.baseScreenshotName) = CompuCellSetup.makeSimDir(self.__fileName,outputDir)
            #            print MODULENAME, 'createOutputDirs(): (not custom dir) screenshotDirectoryName,baseScreenshotName=',self.screenshotDirectoryName,', ',self.baseScreenshotName
            CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName
            self.prevOutputDir = self.__outputDirectory
        #            print MODULENAME,'createOutputDirs:  self.prevOutputDir=',self.__outputDirectory
        else:
            # for parameter scan the directories are created in __loadCC3DFile
            if self.singleSimulation:

                (self.screenshotDirectoryName, self.baseScreenshotName) = self.makeCustomSimDir(
                    self.customScreenshotDirectoryName, self.__fileName)
                #            print MODULENAME, 'createOutputDirs(): (custom dir: screenshotDirectoryName,baseScreenshotName=',self.screenshotDirectoryName,', ',self.baseScreenshotName
                CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName

            else:
                self.screenshotDirectoryName = self.parameterScanOutputDir

                # fullFileName = os.path.abspath(_simulationFileName)
                # (filePath,baseFileName) = os.path.split(fullFileName)
                # baseFileNameForDirectory = baseFileName.replace('.','_')    

                pScanBaseFileName = os.path.basename(self.__fileName)
                pScanBaseFileName, extension = os.path.splitext(pScanBaseFileName)
                # .replace('.','_')                
                screenshotSuffix = os.path.basename(self.screenshotDirectoryName)

                self.baseScreenshotName = pScanBaseFileName + '_' + screenshotSuffix

                # print 'self.baseScreenshotName=',self.baseScreenshotName 

            if self.screenshotDirectoryName == "":
                self.__imageOutput = False  # do not output screenshots when custom directory was not created or already exists

            #        if Configuration.getSetting("LatticeOutputOn"):
        if not self.cmlHandlerCreated:
            #            print MODULENAME,'createOutputDirs():  calling CompuCellSetup.createCMLFieldHandler()'
            CompuCellSetup.createCMLFieldHandler()
            self.cmlHandlerCreated = True  # rwh

        self.resultStorageDirectory = os.path.join(self.screenshotDirectoryName, "LatticeData")

        #        print MODULENAME, 'createOutputDirs(), lattice: resultStorageDirectory=',self.resultStorageDirectory

        #        import CompuCellSetup
        #        print MODULENAME,'createOutputDirs:  calling initCMLFieldHandler()'
        #        print MODULENAME,'createOutputDirs():  calling CompuCellSetup.initCMLFieldHandler()'
        if (self.mysim == None):
            print MODULENAME, '\n\n\n createOutputDirs():  self.mysim is None!!!'  # bad, very bad
            # return
            # sys.exit(0)
        #        else:
        #            print MODULENAME,'createOutputDirs():   type(self.mysim) = ',type(self.mysim)
        #            print MODULENAME,'createOutputDirs():   self.mysim = ',self.mysim

        # simObj=self.mysim() # extracting object from weakref object wrapper
        # if not simObj:return

        CompuCellSetup.initCMLFieldHandler(self.mysim(), self.resultStorageDirectory,
            self.fieldStorage)  # also creates the /LatticeData dir
        # # # CompuCellSetup.initCMLFieldHandler(self.mysim,self.resultStorageDirectory,self.fieldStorage)  # also creates the /LatticeData dir


    #        else:
    #            print MODULENAME,'createOutputDirs:  LatticeOutputOn is False'


    def initializeSimulationViewWidgetRegular(self):
        # print MODULENAME,'  --------- initializeSimulationViewWidgetRegular:'

        # self.pifFromVTKAct.setEnabled(False)
        # import time
        # time.sleep(2)
        # # # sim=self.simulation.sim()
        sim = self.simulation.sim()
        if sim:
            self.fieldDim = sim.getPotts().getCellFieldG().getDim()
            # any references to simulator shuold be weak to avoid possible memory leaks - when not using weak references one has to be super careful to set to Non all references to sim to break any reference cycles
            # weakref is much easier to handle and code is cleaner
            from weakref import ref

            self.mysim = ref(sim)

        simObj = self.mysim()  # extracting object from weakref object wrapper
        if not simObj:
            sys.exit()
            return

        if not self.cerrStreamBufOrig:  #get original cerr stream buffer - do it only once per session
            self.cerrStreamBufOrig = simObj.getCerrStreamBufOrig()

        # if Configuration.getVisualization("CC3DOutputOn"):
        if self.UI.viewmanager.cc3dOutputOnAct.isChecked():
            if Configuration.getSetting("UseInternalConsole"):
                #redirecting output from C++ to internal console
                import sip

                stdErrConsole = self.UI.console.getStdErrConsole()  # we use __stdout console (see UI/Consile.py) as main output console for both stdout and std err from C++ and Python - sort of internal system console
                stdErrConsole.clear()
                addr = sip.unwrapinstance(stdErrConsole)

                simObj.setOutputRedirectionTarget(addr)
                #redirecting Python output to internal console
                self.UI.useInternalConsoleForPythonOutput(True)
            else:
                #C++ output goes to system console
                # simObj.setOutputRedirectionTarget(-1)
                simObj.restoreCerrStreamBufOrig(self.cerrStreamBufOrig)
                #Python output goes to system console
                self.UI.enablePythonOutput(True)
        else:
            #silencing output from C++ 
            simObj.setOutputRedirectionTarget(0)
            #silencing output from Python
            self.UI.enablePythonOutput(False)


            # check if we will be outputting fields in vtk format
        #        import CompuCellSetup
        # print "THIS IS Configuration.getSetting(LatticeOutputOn)",Configuration.getSetting("LatticeOutputOn")
        # print simObj

        self.basicSimulationData.fieldDim = self.fieldDim
        self.basicSimulationData.sim = simObj
        self.basicSimulationData.numberOfSteps = simObj.getNumSteps()

        # # # print 'BEFORE self.fieldStorage.allocateCellField'
        # # # time.sleep(5)
        self.fieldStorage.allocateCellField(self.fieldDim)

        self.fieldExtractor.init(simObj)
        # # # print 'AFTER self.fieldStorage.allocateCellField'
        # # # time.sleep(5)





        self.screenshotNumberOfDigits = len(str(self.basicSimulationData.numberOfSteps))

        #        import CompuCellSetup
        latticeTypeStr = CompuCellSetup.ExtractLatticeType()
        #        print MODULENAME,' initializeSimulationViewWidgetRegular():  latticeTypeStr=',latticeTypeStr
        if latticeTypeStr in Configuration.LATTICE_TYPES.keys():
            self.latticeType = Configuration.LATTICE_TYPES[latticeTypeStr]
        else:
            self.latticeType = Configuration.LATTICE_TYPES["Square"]  # default choice

        # print 'BEFORE prepareSimulationView'
        # time.sleep(5)

        self.prepareSimulationView()

        # print 'AFTER prepareSimulationView'
        # time.sleep(5)


        self.screenshotManager = ScreenshotManager.ScreenshotManager(self)

        # print "self.screenshotManager",self.screenshotManager

        if self.__screenshotDescriptionFileName != "":
            self.screenshotManager.readScreenshotDescriptionFile(self.__screenshotDescriptionFileName)

        if self.simulationIsStepping:
            # print "BEFORE STEPPING PAUSE REGULAR SIMULATION"
            self.__pauseSim()


        #        self.prepareXMLTreeView(self.__fileName)
        self.prepareXMLTreeView()
        # after this call I can access self.root_element of the XML File

    #        self.multiWindowPlayerSettings(self.root_element)
    # # # self.loadCustomPlayerSettings(self.root_element)




    def initializeSimulationViewWidget(self):
        import CompuCellSetup

        CompuCellSetup.simulationFileName = self.__fileName
        # closing all open windows
        self.close_all_windows()

        initializeSimulationViewWidgetFcn = getattr(self, "initializeSimulationViewWidget" + self.__viewManagerType)
        initializeSimulationViewWidgetFcn()

        #        import CompuCellSetup
        # print "self.cc3dSimulationDataHandler=",self.cc3dSimulationDataHandler
        # print 'CompuCellSetup.screenshotDirectoryName=',CompuCellSetup.screenshotDirectoryName

        #        print MODULENAME, 'initializeSimulationViewWidget(),  __imageOutput,__latticeOutputFlag,screenshotDirectoryName=', self.__imageOutput,self.__latticeOutputFlag,self.screenshotDirectoryName
        if (self.__imageOutput or self.__latticeOutputFlag) and self.screenshotDirectoryName == "":
            #            print MODULENAME, 'initializeSimulationViewWidget(),  calling createOutputDirs'
            self.createOutputDirs()


        #copy simulation files to output directory  for simgle simulation- copying of the simulations files for parameter scan is doen in the __loadCC3DFile       
        if self.singleSimulation:
            if self.cc3dSimulationDataHandler and CompuCellSetup.screenshotDirectoryName != "":
                self.cc3dSimulationDataHandler.copySimulationDataFiles(CompuCellSetup.screenshotDirectoryName)


                # print MODULENAME, " initializeSimulationViewWidget():  before set_trace"
            #        import pdb; pdb.set_trace()
        print MODULENAME, " initializeSimulationViewWidget():  before TRY ACQUIRE"
        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()
        print MODULENAME, " initializeSimulationViewWidget():  AFTER RELEASE"

    #        import pdb; pdb.set_trace()

    def runSteppablePostStartPlayerPrep(self):

        self.setFieldTypes()
        # # # self.windowDict[1].setFieldTypesComboBox(self.fieldTypes) #we have only one window at this stage of the simulation run

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()


    #    def multiWindowPlayerSettings(self, _root_element):
    #        print MODULENAME,'multiWindowPlayerSettings(): --------------------------------\n'
    #        import pdb; pdb.set_trace()

    # def setWindowView(self, w, attrKeys):
    #     camera3D = self.lastActiveWindow.getCamera3D()
    #
    #     #                print 'w.getAttributes()=',w.getAttributes()
    #     #                print 'dir(w.getAttributes())=',dir(w.getAttributes())
    #
    #     if "Projection" in attrKeys:
    #         winProj = w.getAttribute("Projection")
    #         print MODULENAME, 'winProj=', winProj
    #         if winProj == '3D':
    #             self.lastActiveWindow._switchDim(True)
    #         else:
    #             if w.findAttribute("XYProj"):
    #                 zPos = w.getAttributeAsUInt("XYProj")
    #                 print MODULENAME, '  loadCustomPlayerSettings(): XYProj, zPos =', zPos  #rwh
    #                 if zPos >= self.xySB.minimum() and zPos <= self.xySB.maximum():
    #                     self.lastActiveWindow._xyChecked(True)
    #                     self.lastActiveWindow._projSpinBoxChanged(zPos)
    #             elif w.findAttribute("XZProj"):
    #                 yPos = w.getAttributeAsUInt("XZProj")
    #                 if yPos >= self.xzSB.minimum() and yPos <= self.xzSB.maximum():
    #                     self.lastActiveWindow._xzChecked(True)
    #                     self.lastActiveWindow._projSpinBoxChanged(yPos)
    #
    #             elif w.findAttribute("YZProj"):
    #                 xPos = w.getAttributeAsUInt("YZProj")
    #                 if xPos >= self.yzSB.minimum() and xPos <= self.yzSB.maximum():
    #                     self.lastActiveWindow._yzChecked(True)
    #                     self.lastActiveWindow._projSpinBoxChanged(xPos)
    #
    #     if "WindowNumber" in attrKeys:
    #         winNum = w.getAttributeAsUInt("WindowNumber")
    #         print MODULENAME, 'winNum=', winNum
    #
    #     # set camera params for each window
    #     if "CameraPos" in attrKeys:
    #         p = w.getAttribute("CameraPos")
    #         pStr = p.split()
    #         cameraPos = [float(pStr[0]), float(pStr[1]), float(pStr[2])]
    #         print MODULENAME, 'cameraPos=', cameraPos
    #         camera3D.SetPosition(cameraPos)
    #     if "CameraFocalPoint" in attrKeys:
    #         p = w.getAttribute("CameraFocalPoint")
    #         pStr = p.split()
    #         cameraFocalPoint = [float(pStr[0]), float(pStr[1]), float(pStr[2])]
    #         print MODULENAME, 'cameraFocalPoint=', cameraFocalPoint
    #         camera3D.SetFocalPoint(cameraFocalPoint)
    #     if "CameraViewUp" in attrKeys:
    #         p = w.getAttribute("CameraViewUp")
    #         pStr = p.split()
    #         cameraViewUp = [float(pStr[0]), float(pStr[1]), float(pStr[2])]
    #         print MODULENAME, 'cameraViewUp=', cameraViewUp
    #         camera3D.SetViewUp(cameraViewUp)
    #     if "CameraClippingRange" in attrKeys:
    #         p = w.getAttribute("CameraClippingRange")
    #         pStr = p.split()
    #         cameraClippingRange = [float(pStr[0]), float(pStr[1])]
    #         print MODULENAME, 'cameraClippingRange=', cameraClippingRange
    #         camera3D.SetClippingRange(cameraClippingRange)
    #     if "CameraDistance" in attrKeys:
    #         p = w.getAttributeAsDouble("CameraDistance")
    #         print MODULENAME, 'camera distance=', p
    #         camera3D.SetDistance(p)
    #
    #     self.lastActiveWindow.ren.ResetCameraClippingRange()  # need to do this, else might have clipped actors


    # def __savePlayerParams(self):
    #     # print "THIS IS __saveScrDesc"
    #     filter = "Player parameters File (*.txt )"  # self._getOpenFileFilter()
    #     self.playerParamsFileName = QFileDialog.getSaveFileName( \
    #         self.ui,
    #         QApplication.translate('ViewManager', "Save Player Parameters File"),
    #         os.getcwd(),
    #         filter
    #         )
    #     #        if self.screenshotManager:
    #     #            self.screenshotManager.writePlayerParamsFile(self.playerParamsFileName)
    #
    #     print MODULENAME, "playerParamsFileName=", self.playerParamsFileName
    #     #        pFile = open(self.playerParamsFileName,'w')
    #     #        pFile.write('size 512\n')
    #     #        pFile.close()
    #     #        from csv import writer
    #     paramsDict = Configuration.getPlayerParams()
    #     #        paramWriter = writer(open(self.playerParamsFileName, 'w'), delimiter=' ')
    #     #        paramWriter.writerow(params)
    #     paramFile = open(self.playerParamsFileName, 'w')
    #     paramFile.write(repr(paramsDict))
    #
    #     #        for idx in paramsDict.keys():
    #     #            paramFile.write(idx, repr(paramsDict))
    #     paramFile.close()
    #     print MODULENAME, 'paramsDict =', paramsDict


    def extractAddressIntFromVtkObject(self, _vtkObj):
        return self.fieldExtractor.unmangleSWIGVktPtrAsLong(_vtkObj.__this__)

    def handleSimulationFinishedCMLResultReplay(self, _flag):
        #        import CompuCellSetup
        if CompuCellSetup.playerType == "CMLResultReplay":
            self.latticeDataModelTable.prepareToClose()

        # # # self.__stopSim()
        self.__cleanAfterSimulation()

    def launchNextParameterScanRun(self):
        fileName = self.__fileName
        # when runnign parameter scan after simulatino finish we run again the same simulation file. When cc3d project with parameter scan gets opened 'next iteration' simulation is generatet and this 
        # newly generated cc3d file is substituted instead of the "master" cc3d with parameter scan 
        # From user stand point whan matters is that the only thing that user needs to worry abuot is the "master" .cc3d project and this is what is opened in the player
        self.consecutiveRunCounter += 1
        if self.consecutiveRunCounter >= self.maxNumberOfConsecutiveRuns:

            from ParameterScanUtils import getParameterScanCommandLineArgList
            from SystemUtils import getCC3DPlayerRunScriptPath

            print 'getCC3DPlayerRunScriptPath=', getCC3DPlayerRunScriptPath()
            # had to use tw-line to do simple thing
            cc3dPlayerRunScriptPath = getCC3DPlayerRunScriptPath()
            popenArgs = [cc3dPlayerRunScriptPath] + getParameterScanCommandLineArgList(fileName)
            # this code, although valid, will not work on Apple....
            # popenArgs =[ getCC3DPlayerRunscriptPath() ] +getParameterScanCommandLineArgList(fileName)

            from subprocess import Popen

            cc3dProcess = Popen(popenArgs)
            sys.exit()
        else:
            self.__runSim()


    def handleSimulationFinishedRegular(self, _flag):
        print 'INSIDE handleSimulationFinishedRegular'
        self.__cleanAfterSimulation()

        if not self.singleSimulation:
            self.launchNextParameterScanRun()

            # self.__stopSim()

    def handleSimulationFinished(self, _flag):
        handleSimulationFinishedFcn = getattr(self, "handleSimulationFinished" + self.__viewManagerType)
        handleSimulationFinishedFcn(_flag)


    def handleCompletedStepCMLResultReplay(self, _mcs):
        self.simulation.drawMutex.lock()  # had to add synchronization here . without it I would get weird behavior in CML replay mode

        # print  '\n\n ---------------- handleCompletedStepCMLResultReplay'                                        
        # print '--------------BEFORE self.fieldDim=',self.fieldDim        

        print "THIS IS handleCompletedStepCMLResultReplay"
        # print "Before extracting the address self.simulation.simulationData=",self.simulation.simulationData
        simulationDataIntAddr = self.extractAddressIntFromVtkObject(self.simulation.simulationData)
        # print "simulationDataIntAddr=%X\n"% (simulationDataIntAddr)
        # print "self.simulation.simulationData=",self.simulation.simulationData
        self.fieldExtractor.setSimulationData(simulationDataIntAddr)
        self.__step = self.simulation.currentStep
        print 'SIMPLE TAB VIEW self.__step=', self.__step
        print 'SIMPLE TAB VIEW self.simulation.frequency=', self.simulation.frequency
        # # # self.latticeDataModelTable.selectRow(self.__step / self.simulation.frequency ) # here elf.step holds the value of the "user step" i.e. not multiplied by frequency. it gets multiplied by frequency next
        self.latticeDataModelTable.selectRow(
            self.simulation.stepCounter - 1)  #self.simulation.stepCounter is incremented by one before it reaches this function

        #there is additional locking inside draw to acccount for the fact that users may want to draw lattice on demand
        # self.simulation.newFileBeingLoaded=False
        self.simulation.drawMutex.unlock()  # had to add synchronization here . without it I would get weird behavior in CML replay mode      

        self.simulation.newFileBeingLoaded = False  # this flag is used to prevent calling  draw function when new data is read from hard drive
        # at this moment new data has been read and is ready to be used


        self.__drawField()
        # print '----------------AFTER self.fieldDim=',self.fieldDim                                                 

        self.simulation.drawMutex.lock()
        # will need to synchorinize screenshots with simulation thread . make sure that before simuklation thread writes new results all the screenshots are taken

        if self.__imageOutput and not (self.__step % self.__shotFrequency):  # dumping images? Check modulo MCS #
            mcsFormattedNumber = string.zfill(str(self.__step),
                                              self.screenshotNumberOfDigits)  # fills string wtih 0's up to self.screenshotNumberOfDigits width
            screenshotFileName = os.path.join(self.screenshotDirectoryName,
                                              self.baseScreenshotName + "_" + mcsFormattedNumber + ".png")
            #            print MODULENAME," handleCompletedStepCMLResultReplay():  calling takeSimShot w/ screenshotFileName=",screenshotFileName
            self.mainGraphicsWindow.takeSimShot(screenshotFileName)
            self.screenshotManager.outputScreenshots(self.screenshotDirectoryName, self.__step)

        self.simulation.drawMutex.unlock()

        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

        self.cmlReplayManager.keepGoing()

    def handleFinishRequest(self, _flag):
        #this ensures that all the tasks in the GUI thread that need simulator to be alive are completed before proceeding further with finalizing the simulation 
        #e.g. SimpleTabViewpy. function handleCompletedStepRegular may need a lot of time to output simulations fields and those fields need to have alive simulator otherwise accessing to destroyed field will lead to segmentation fault

        self.simulation.drawMutex.lock()
        self.simulation.drawMutex.unlock()

        # this releases finish mutex which is a signal to simulation thread that is is OK to finish
        self.simulation.finishMutex.unlock()


    def handleCompletedStepRegular(self, _mcs):

        # print 'handleCompletedStepRegular = ', handleCompletedStepRegular

        self.__drawField()

        self.simulation.drawMutex.lock()
        # will need to sync screenshots with simulation thread. Be sure before simulation thread writes new results all the screenshots are taken

        # print MODULENAME, 'handleCompletedStepRegular():  __shotFrequency, __imageOutput = ',self.__shotFrequency,self.__imageOutput

        if self.__imageOutput and not (self.__step % self.__shotFrequency):  # dumping images? Check modulo MCS #
            mcsFormattedNumber = string.zfill(str(self.__step),
                                              self.screenshotNumberOfDigits)  # fills string wtih 0's up to self.screenshotNumberOfDigits width
            screenshotFileName = os.path.join(self.screenshotDirectoryName,
                                              self.baseScreenshotName + "_" + mcsFormattedNumber + ".png")
            #            print '       handleCompletedStepRegular():  screenshotDirectoryName=',self.screenshotDirectoryName
            #            print '       handleCompletedStepRegular():  baseScreenshotName=',self.baseScreenshotName
            #            print MODULENAME,'  handleCompletedStepRegular():  calling takeSimShot w/ screenshotFileName=',screenshotFileName
            if _mcs != 0:
                if self.mainGraphicsWindow:  # self.mainGraphicsWindow can be closed by the user
                    self.mainGraphicsWindow.takeSimShot(screenshotFileName)

            if self.screenshotManager:
                self.screenshotManager.outputScreenshots(self.screenshotDirectoryName, self.__step)

                # print 'self.screenshotDirectoryName=',self.screenshotDirectoryName
                # sys.exit()

            #        if (CompuCellSetup.cmlFieldHandler is not None) and self.__latticeOutputFlag and (not self.__step % self.__latticeOutputFrequency):  #rwh
        if self.cmlHandlerCreated and self.__latticeOutputFlag and (
                not self.__step % self.__latticeOutputFrequency):  #rwh
            #            print MODULENAME,' handleCompletedStepRegular(): cmlFieldHandler.writeFields'
            #            import CompuCellSetup
            CompuCellSetup.cmlFieldHandler.writeFields(self.__step)

        self.simulation.drawMutex.unlock()

        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()


    def handleCompletedStep(self, _mcs):


        #        print MODULENAME, 'handleCompletedStep:  _mcs =',_mcs
        if not self.completedFirstMCS:
            self.completedFirstMCS = True
        #            self.updateActiveWindowVisFlags()

        self.__step = _mcs

        #        print MODULENAME, 'handleCompletedStep:  self.__viewManagerType =',self.__viewManagerType
        #        fcnName = "handleCompletedStep" + self.__viewManagerType
        #        print MODULENAME, 'handleCompletedStep:  fcnName=',fcnName
        handleCompletedStepFcn = getattr(self, "handleCompletedStep" + self.__viewManagerType)
        #        handleCompletedStepFcn = getattr(self, fcnName)
        handleCompletedStepFcn(_mcs)
        return


    def updateSimPrefs(self):

        self.simulation.screenUpdateFrequency = self.__updateScreen
        self.simulation.imageOutputFlag = self.__imageOutput
        self.simulation.screenshotFrequency = self.__shotFrequency
        self.simulation.latticeOutputFlag = self.__latticeOutputFlag
        self.simulation.latticeOutputFrequency = self.__latticeOutputFrequency


    def prepareSimulation(self):

        if not self.drawingAreaPrepared:
            # checking if the simulation file is not an empty string 
            if self.__fileName == "":
                msg = QMessageBox.warning(self, "Not A Valid Simulation File", \
                                          "Please pick simulation file <b>File->OpenSimulation File ...</b>", \
                                          QMessageBox.Ok,
                                          QMessageBox.Ok)
                return
            file = QFile(self.__fileName)

            import xml

            try:
                self.__loadSim(file)
            except AssertionError, e:
                print "Assertion Error: ", e.message

                self.handleErrorMessage("Assertion Error", e.message)
                import ParameterScanEnums

                if _errorType == 'Assertion Error' and _traceback_message.startswith(
                                'Parameter Scan ERRORCODE=' + str(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)):
                    #                     print 'Exiting inside prepare simulation '
                    sys.exit(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)

                return
            except xml.parsers.expat.ExpatError, e:
                # if CompuCellSetup.simulationObjectsCreated:
                # sim.finish()
                # import PlayerPython
                # simthread=PlayerPython.getSimthreadBasePtr()
                xmlFileName = CompuCellSetup.simulationPaths.simulationXMLFileName
                print "Error in XML File", "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message
                self.handleErrorMessage("Error in XML File",
                                        "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message)
            except IOError, e:
                return

            self.updateSimPrefs()
            #            self.simulation.screenUpdateFrequency = self.__updateScreen
            #            self.simulation.imageOutputFlag = self.__imageOutput
            #            self.simulation.screenshotFrequency = self.__shotFrequency
            #            self.simulation.latticeOutputFlag = self.__latticeOutputFlag
            #            self.simulation.latticeOutputFrequency = self.__latticeOutputFrequency
            print '__runSim self.screenshotDirectoryName=', self.screenshotDirectoryName

            self.screenshotDirectoryName = ""
            print '__runSim self.screenshotDirectoryName=', self.screenshotDirectoryName

            # sys.exit()

            if self.rollbackImporter:
                self.rollbackImporter.uninstall()

            self.rollbackImporter = RollbackImporter()

    def __runSim(self):

        self.simulation.screenUpdateFrequency = self.__updateScreen  # when we run simulation we ensure that self.simulation.screenUpdateFrequency is whatever is written in the settings

        if not self.drawingAreaPrepared:
            self.prepareSimulation()

        # print 'SIMULATION PREPARED self.__viewManagerType=',self.__viewManagerType    
        if self.__viewManagerType == "CMLResultReplay":
            # print 'starting CMLREPLAY'
            import CompuCellSetup

            self.simulation.semPause.release()  # just in case
            self.cmlReplayManager.setRunState()
            self.cmlReplayManager.keepGoing()
            self.simulationIsRunning = True
            self.simulationIsStepping = False

            self.runAct.setEnabled(False)
            self.stepAct.setEnabled(True)
            self.stopAct.setEnabled(True)
            self.pauseAct.setEnabled(True)

            return
        else:
            if not self.simulationIsRunning:
                self.simulation.start()
                self.simulationIsRunning = True
                self.simulationIsStepping = False

                self.runAct.setEnabled(False)
                self.stepAct.setEnabled(True)
                self.stopAct.setEnabled(True)
                self.pauseAct.setEnabled(True)

            if Configuration.getSetting("LatticeOutputOn") and not self.cmlHandlerCreated:
                import CompuCellSetup

                CompuCellSetup.createCMLFieldHandler()
                self.cmlHandlerCreated = True
                #            CompuCellSetup.initCMLFieldHandler(self.mysim,self.resultStorageDirectory,self.fieldStorage)

            self.steppingThroughSimulation = False

            if self.simulationIsStepping:
                self.simulationIsStepping = False
                self.updateSimPrefs()

            if not self.pauseAct.isEnabled() and self.simulationIsRunning:
                self.runAct.setEnabled(False)
                self.pauseAct.setEnabled(True)
                self.simulation.semPause.release()
                return


    def __stepSim(self):

        self.simulation.screenUpdateFrequency = 1  # when we step we need to ensure screenUpdateFrequency is 1

        if not self.drawingAreaPrepared:
            self.prepareSimulation()

        # print 'SIMULATION PREPARED self.__viewManagerType=',self.__viewManagerType    
        if self.__viewManagerType == "CMLResultReplay":
            # print 'starting CMLREPLAY'
            import CompuCellSetup

            self.simulation.semPause.release()
            self.cmlReplayManager.setStepState()
            self.cmlReplayManager.step()
            self.simulationIsRunning = True
            self.simulationIsStepping = True

            self.stopAct.setEnabled(True)
            self.pauseAct.setEnabled(False)
            self.runAct.setEnabled(True)
            self.pifFromVTKAct.setEnabled(True)
            return

        else:
            if not self.simulationIsRunning:
                self.simulationIsStepping = True
                self.simulationIsRunning = True

                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.screenshotDirectoryName = ""

                self.runAct.setEnabled(True)
                self.pauseAct.setEnabled(False)
                self.stopAct.setEnabled(True)
                self.pifFromSimulationAct.setEnabled(True)
                self.openAct.setEnabled(False)
                self.openLDSAct.setEnabled(False)

                self.simulation.start()

            if self.completedFirstMCS and Configuration.getSetting(
                    "LatticeOutputOn") and not self.cmlHandlerCreated:  #rwh
                CompuCellSetup.createCMLFieldHandler()
                self.cmlHandlerCreated = True  #rwh

                CompuCellSetup.initCMLFieldHandler(self.mysim, self.resultStorageDirectory, self.fieldStorage)
                CompuCellSetup.cmlFieldHandler.getInfoAboutFields()  #rwh

            if self.simulationIsRunning and self.simulationIsStepping:
                #            print MODULENAME,'  __stepSim() - 1:'
                self.pauseAct.setEnabled(False)
                self.simulation.semPause.release()
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)

                return

            # if Pause button is enabled
            elif self.simulationIsRunning and not self.simulationIsStepping and self.pauseAct.isEnabled():  #transition from running simulation
                #            print MODULENAME,'  __stepSim() - 2:'
                #            updateSimPrefs()   # should we call this and then reset screenUpdateFreq = 1 ?
                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.simulationIsStepping = True
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)
            # if Pause button is disabled, meaning the sim is paused:
            elif self.simulationIsRunning and not self.simulationIsStepping and not self.pauseAct.isEnabled():  #transition from paused simulation
                #            print MODULENAME,'  __stepSim() - 3:'
                #            updateSimPrefs()   # should we call this and then reset screenUpdateFreq = 1 ?
                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.simulationIsStepping = True

                return

            return


    def requestRedraw(self):
        if self.simulationIsRunning or self.simulationIsStepping:
            self.__drawField()

    def drawFieldCMLResultReplay(self):

        self.simulation.drawMutex.lock()
        self.simulation.readFileSem.acquire()

        if not self.simulationIsRunning:
            self.simulation.drawMutex.unlock()
            self.simulation.readFileSem.release()

            return

        if self.newDrawingUserRequest:
            # print "entering newDrawingUserRequest"
            self.newDrawingUserRequest = False
            if self.pauseAct.isEnabled():
                # print "PAUSING THE SIMULATION"
                self.__pauseSim()

        self.simulation.drawMutex.unlock()
        self.simulation.readFileSem.release()

        # print "self.simulation.drawMutex=",self.simulation.drawMutex
        self.simulation.drawMutex.lock()
        self.simulation.readFileSem.acquire()

        self.__step = self.simulation.getCurrentStep()

        if True:
            # MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsFrame = win.widget()

            # for windowName in self.windowDict.keys():
            #     graphicsFrame = self.windowDict[windowName]
                # print "graphicsFrame=",graphicsFrame

                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()

                # print "NEW FILE IS LOADED =",self.simulation.newFileBeingLoaded

                if not self.simulation.newFileBeingLoaded:  # this flag is used to prevent calling  draw function when new data is read from hard drive
                    graphicsFrame.drawFieldLocal(self.basicSimulationData)

                self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())

        self.simulation.drawMutex.unlock()
        self.simulation.readFileSem.release()

    def drawFieldRegular(self):
        #        print MODULENAME,'  drawFieldRegular(): simulationIsRunning=',self.simulationIsRunning
        #        import pdb; pdb.set_trace()

        if not self.simulationIsRunning:
            return

        if self.newDrawingUserRequest:
            self.newDrawingUserRequest = False
            if self.pauseAct.isEnabled():
                self.__pauseSim()
        self.simulation.drawMutex.lock()

        self.__step = self.simulation.getCurrentStep()
        #        print MODULENAME,'  drawFieldRegular(): __step=',self.__step


        # self.simulation.drawMutex.unlock()
        # return


        if self.mysim:
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsFrame = win.widget()

                if graphicsFrame.is_screenshot_widget:
                    continue

                #rwh: error if we try to invoke switchdim earlier
                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()

                graphicsFrame.drawFieldLocal(self.basicSimulationData)

                self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())  # show MCS in lower-left GUI

        self.simulation.drawMutex.unlock()





        # if self.mysim:
        #     #            print MODULENAME,'  drawFieldRegular(): in self.mysim block; windowDict.keys=',self.graphicsWindowDict.keys()
        #     for windowName in self.graphicsWindowDict.keys():
        #         graphicsFrame = self.windowDict[windowName]
        #
        #
        #         #                print MODULENAME,"drawFieldRegular():   windowName, graphicsFrame=",windowName, graphicsFrame
        #
        #
        #         #rwh: error if we try to invoke switchdim earlier
        #         (currentPlane, currentPlanePos) = graphicsFrame.getPlane()
        #
        #         # print 'BEFORE graphicsFrame.drawFieldLocal'
        #         # time.sleep(5)
        #
        #         graphicsFrame.drawFieldLocal(self.basicSimulationData)
        #
        #         # print 'AFTER graphicsFrame.drawFieldLocal'
        #         # time.sleep(5)
        #
        #
        #         self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())  # show MCS in lower-left GUI
        #
        # self.simulation.drawMutex.unlock()


    def updateSimulationProperties(self):
        # print 'INSIDE updateSimulationProperties ',self.fieldDim

        fieldDim = None
        if self.__viewManagerType == "Regular":

            simObj = self.mysim()
            if not simObj: return

            fieldDim = simObj.getPotts().getCellFieldG().getDim()
            # # # fieldDim = self.simulation.sim.getPotts().getCellFieldG().getDim()

            if fieldDim.x == self.fieldDim.x and fieldDim.y == self.fieldDim.y and fieldDim.z == self.fieldDim.z:
                return False

            self.fieldDim = fieldDim
            self.basicSimulationData.fieldDim = self.fieldDim
            self.basicSimulationData.sim = simObj
            self.basicSimulationData.numberOfSteps = simObj.getNumSteps()

            # # # self.fieldDim= fieldDim   
            # # # self.basicSimulationData.fieldDim = self.fieldDim
            # # # self.basicSimulationData.sim = self.mysim
            # # # self.basicSimulationData.numberOfSteps = self.mysim.getNumSteps()

            return True

        elif self.__viewManagerType == "CMLResultReplay":
            fieldDim = self.simulation.fieldDim
            if self.simulation.dimensionChange():
                self.simulation.resetDimensionChangeMonitoring()
                self.fieldDim = self.simulation.fieldDim
                self.basicSimulationData.fieldDim = self.fieldDim
                self.fieldExtractor.setFieldDim(self.basicSimulationData.fieldDim)
                # self.basicSimulationData.sim = self.mysim
                # self.basicSimulationData.numberOfSteps = self.mysim.getNumSteps()
                return True

            return False


    def updateVisualization(self):
        # print 'INSIDE f'
        self.fieldStorage.allocateCellField(self.fieldDim)
        # this updates cross sections when dimensions change

        #MDIFIX
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            win.widget().updateCrossSection(self.basicSimulationData)

        # for windowName in self.windowDict.keys():
        #     self.windowDict[windowName].updateCrossSection(self.basicSimulationData)


        # # # slf.setInitialCrossSection(self.basicSimulationData)

        #MDIFIX
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsWidget = win.widget()
            graphicsWidget.resetAllCameras()

        # for windowName, graphicsWindow in self.graphicsWindowDict.iteritems():
        #     graphicsWindow.resetAllCameras()


        # # # self.prepareSimulationView() # this pauses simulation        
        # self.__drawField()

        if self.simulationIsRunning and not self.simulationIsStepping:
            self.__runSim()  # we are immediately restarting it after e.g. lattice resizing took place

    def _drawField(self):  # called from GraphicsFrameWidget.py
        #        print MODULENAME,'   _drawField called'
        self.__drawField()

    def __drawField(self):

        self.displayWarning('') # here we are resetting previous warnings because draw functions may write their own warning

        __drawFieldFcn = getattr(self, "drawField" + self.__viewManagerType)
        # print MODULENAME, '__drawField():  calling ',"drawField"+self.__viewManagerType
        # # time.sleep(5)
        # import time
        # time.sleep(2)


        # print 'self.__viewManagerType=',self.__viewManagerType        
        propertiesUpdated = self.updateSimulationProperties()
        # print 'propertiesUpdated=',propertiesUpdated


        if propertiesUpdated:
            # __drawFieldFcn() # this call is actually unnecessary
            self.updateVisualization()  # for some reason cameras have to be initialized after drawing resized lattice and draw function has to be repeated

        __drawFieldFcn()

        # if propertiesUpdated:
        # __drawFieldFcn()

    def displayWarning(self, warning_text):
        self.warnings.setText(warning_text)

    def __updateStatusBar(self, step, conMinMax):

        self.mcSteps.setText("MC Step: %s" % step)
        self.conSteps.setText("Min: %s Max: %s" % conMinMax)

    def __pauseSim(self):
        # print "Pause Sim"
        if self.__viewManagerType == "CMLResultReplay":
            self.cmlReplayManager.setPauseState()

        self.simulation.semPause.acquire()
        self.runAct.setEnabled(True)
        self.pauseAct.setEnabled(False)

    def __saveWindowsLayout(self):

        # print 'SIMULATION STOP BEFORE self.fieldTypes = ',self.fieldTypes
        # print 'BEFORE self.fieldTypes = ',self.fieldTypes
        windowsLayout = {}

        for key, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            print 'key, win = ', (key, win)
            # mdiWidget = self.findMDISubWindowForWidget(win)
            # print 'mdiwidget = ', mdiWidget
            # print 'Current scene name = ', win.getCurrentSceneNameAndType()
            widget = win.widget()
            # if not widget.allowSaveLayout: continue
            if widget.is_screenshot_widget:
                continue

            gwd = widget.getGraphicsWindowData()
            # fill size and position of graphics windows data using mdiWidget, NOT the internal widget such as GraphicsFrameWidget - sizes and positions are base on MID widet settings
            gwd.winPosition = win.pos()
            gwd.winSize = win.size()

            print 'getGraphicsWindowData=', gwd
            print 'toDict=', gwd.toDict()

            windowsLayout[key] = gwd.toDict()

            # Configuration.setSetting('WindowsLayout',windowsLayout)



            # adding new widget
            # self.addNewGraphicsWindow()
            # # gfw = self.findMDISubWindowForWidget(self.lastActiveWindow)
            # gfw = self.lastActiveWindow
            # gfw.applyGraphicsWindowData(gwd)



        # for key, win in self.graphicsWindowDict.items():
        #     print 'key, win = ', (key, win)
        #     mdiWidget = self.findMDISubWindowForWidget(win)
        #     print 'mdiwidget = ', mdiWidget
        #     # print 'Current scene name = ', win.getCurrentSceneNameAndType()
        #     gwd = win.getGraphicsWindowData()
        #     # fill size and position of graphics windows data using mdiWidget, NOT the internal widget such as GraphicsFrameWidget - sizes and positions are base on MID widet settings
        #     gwd.winPosition = mdiWidget.pos()
        #     gwd.winSize = mdiWidget.size()
        #
        #     print 'getGraphicsWindowData=', gwd
        #     print 'toDict=', gwd.toDict()
        #
        #     windowsLayout[key] = gwd.toDict()
        #
        #     # Configuration.setSetting('WindowsLayout',windowsLayout)
        #
        #
        #
        #     # adding new widget
        #     # self.addNewGraphicsWindow()
        #     # # gfw = self.findMDISubWindowForWidget(self.lastActiveWindow)
        #     # gfw = self.lastActiveWindow
        #     # gfw.applyGraphicsWindowData(gwd)

            # break
        print 'AFTER self.fieldTypes = ', self.fieldTypes
        print self.plotManager.plotWindowList

        plotLayoutDict = self.plotManager.getPlotWindowsLayoutDict()
        for key, gwd in plotLayoutDict.iteritems():
            print 'key=', key
            print 'gwd=', gwd

        # combining two layout dicts
        windowsLayoutCombined = windowsLayout.copy()
        windowsLayoutCombined.update(plotLayoutDict)
        # print 'windowsLayoutCombined=',windowsLayoutCombined
        Configuration.setSetting('WindowsLayout', windowsLayoutCombined)

        # for key, win in self.windowDict.iteritems():
        # if key ==3:
        # print 'dir(win) =', dir(win)
        # import Graphics
        # print 'key, win = ' , (key , win)
        # print 'type(win)    ',  type(win)
        # print 'type(Graphics.PlotFrameWidget.PlotFrameWidget) = ', type(Graphics.PlotFrameWidget.PlotFrameWidget)
        # if type(win) == type(Graphics.PlotFrameWidget.PlotFrameWidget):
        # print 'key, win = ' , (key , win)
        # print 'win '

    def __simulationStop(self):
        # once user requests explicit stop of the simulation we stop regardless whether this is parameter scan or not. To stop parameter scan we reset vaiables used to seer parameter scanto their default (non-param scan) values

        self.runAgainFlag = False

        #we do not save windows layout for simulation replay
        if self.__viewManagerType != "CMLResultReplay":
            self.__saveWindowsLayout()

        if self.__viewManagerType == "CMLResultReplay":
            self.cmlReplayManager.setStopState()
            self.runAct.setEnabled(True)
            self.stepAct.setEnabled(True)
            self.pauseAct.setEnabled(False)
            self.stopAct.setEnabled(False)

            self.cmlReplayManager.initial_data_read.disconnect(self.initializeSimulationViewWidget)
            self.cmlReplayManager.subsequent_data_read.disconnect(self.handleCompletedStep)
            self.cmlReplayManager.final_data_read.disconnect(self.handleSimulationFinished)

        if not self.singleSimulation:
            self.singleSimulation = True
            self.parameterScanFile = ''

        if not self.pauseAct.isEnabled():
            self.__stopSim()
            self.__cleanAfterSimulation()
        else:
            self.simulation.setStopSimulation(True)


    def __simulationSerialize(self):
        print self.simulation.restartManager
        currentStep = self.simulation.sim.getStep()
        if self.pauseAct.isEnabled():
            self.__pauseSim()
        self.simulation.restartManager.outputRestartFiles(currentStep, True)


    def __cleanAfterSimulation(self, _exitCode=0):

        self.resetControlButtonsAndActions()
        self.resetControlVariables()

        self.fieldTypes = {}  # re-init (empty) the fieldTypes dict, otherwise get previous/bogus fields in graphics win field combobox

        #saving settings witht eh simulation
        if self.customSettingPath:
            # Configuration.writeCustomFile(self.customSettingPath)
            # Configuration.writeSettings()
            Configuration.writeSettingsForSingleSimulation(self.customSettingPath)
            self.customSettingPath = ''

        Configuration.writeAllSettings()
        Configuration.initConfiguration() # this flushes configuration

        if Configuration.getSetting("ClosePlayerAfterSimulationDone") or self.closePlayerAfterSimulationDone:
            Configuration.setSetting("RecentFile", os.path.abspath(self.__fileName))

            Configuration.setSetting("RecentSimulations", os.path.abspath(self.__fileName))
            # Configuration.addItemToStrlist(item = os.path.abspath(self.__fileName),strListName = 'RecentSimulations',maxLength = Configuration.getSetting('NumberOfRecentSimulations'))

            # if self.saveSettings:
            #     Configuration.syncPreferences()

            sys.exit(_exitCode)

        # in case there is pending simulation to be run we will put it a recent simulation so that it can be ready to run without going through open file dialog
        if self.nextSimulation != "":
            Configuration.setSetting("RecentSimulations", self.nextSimulation)
            # Configuration.addItemToStrlist(item = self.nextSimulation,strListName = 'RecentSimulations',maxLength = Configuration.getSetting('NumberOfRecentSimulations'))
            self.nextSimulation = ""

        self.simulation.sim = None
        self.basicSimulationData.sim = None
        self.mysim = None


        # print 'self.screenshotManager=',self.screenshotManager
        if self.screenshotManager:
            self.screenshotManager.cleanup()

        self.screenshotManager = None

        CompuCellSetup.resetGlobals()
        print 'AFTER __cleanupAfterSimulation'

        # self.close_all_windows()

        print self.win_inventory

    def close_all_windows(self):
        for win in self.win_inventory.values():
            self.win_inventory.remove_from_inventory(win)
            win.close()

        self.win_inventory.set_counter(0)

    def __stopSim(self):

        self.simulation.stop()
        self.simulation.wait()


    def makeCustomSimDir(self, _dirName, _simulationFileName):
        fullFileName = os.path.abspath(_simulationFileName)
        (filePath, baseFileName) = os.path.split(fullFileName)
        baseFileNameForDirectory = baseFileName.replace('.', '_')
        if not os.path.isdir(_dirName):
            os.mkdir(_dirName)
            return (_dirName, baseFileNameForDirectory)
        else:
            return ("", "")

    def mapCellTypeToColor(self, cellType):
        return self.colors[cellType]

    # Shows the plugin view tab
    def showPluginView(self, pluginInfo):
        textStr = QString('<div style="margin: 10px 10px 10px 20px; font-size: 14px"><br />\
        Plugin: &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b>%1</b><br />\
        Description: &nbsp; %2</div>').arg(pluginInfo[0]).arg(pluginInfo[1])

        gip = DefaultData.getIconPath
        if self.pluginTab is None:
            self.pluginTab = QTextEdit(textStr, self)
            self.addTab(self.pluginTab, QIcon(gip("plugin.png")), pluginInfo[0])
            # self.closeTab.show()
        else:
            # The plugin view always has index 1 if simview present 0 otherwhise
            if self.count() == 2:
                idx = 1
            else:
                idx = 0
            self.setTabText(idx, pluginInfo[0])  # self.currentIndex()
            self.pluginTab.setText(textStr)

        self.setCurrentIndex(1)

    def prepareDrawingArea(self):  # NEVER called?!  (instead, rf. showSimView())
        #        print MODULENAME, '----------  prepareDrawingArea'
        foo = 1 / 0
        self.__setupArea()
        # # # self.mainGraphicsWindow = self.mainGraphicsWindow # Graphics2D by default
        self.__step = 0
        self.__loadSim(file)
        self.setSimParams()
        self.mainGraphicsWindow.initSimArea(self.sim)
        # self.graphics3D.initSimArea(self.sim)
        self.drawingAreaPrepared = True

    #        self.mainGraphicsWindow.parentWidget.move(400,300)   # temporarily moves, but jumps back

    #        if True:
    #self.mainGraphicsWindow._switchDim(True)   #rwh

    def setInitialCrossSection(self, _basicSimulationData):

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsFrame = win.widget()
            graphicsFrame.setInitialCrossSection(_basicSimulationData)


        # for windowName in self.windowDict.keys():
        #     print 'windowName=', windowName
        #     self.windowDict[windowName].setInitialCrossSection(_basicSimulationData)
        #     print 'after windowName=', windowName

    def initGraphicsWidgetsFieldTypes(self):

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsFrame = win.widget()
            graphicsFrame.setFieldTypesComboBox(self.fieldTypes)

        # for windowName in self.windowDict.keys():
        #     # print 'windowName=',windowName
        #     # print 'self.windowDict[windowName]=',self.windowDict[windowName]
        #     # print 'dir of self.windowDict[windowName]=',dir(self.windowDict[windowName])
        #     self.windowDict[windowName].setFieldTypesComboBox(self.fieldTypes)


    # Shows simulation view tab
    def showSimView(self, file):


        self.__setupArea()

        isTest = False

        """      
        # For testing. Leave for a while
        if isTest:
            self.mainGraphicsWindow = QVTKRenderWidget(self)
            self.insertTab(0, self.mainGraphicsWindow, QIcon("player/icons/sim.png"), os.path.basename(str(self.__fileName)))
            self.setupArea()
        else:
        """

        # Create self.mainGraphicsWindow  
        # # # self.mainGraphicsWindow = self.mainGraphicsWindow # Graphics2D by default
        self.__step = 0

        self.showDisplayWidgets()

        simObj = None
        if self.mysim:
            simObj = self.mysim()
            # if not simObj:return

        self.__fieldType = ("Cell_Field", FIELD_TYPES[0])

        # self.__fieldType = ("FGF", FIELD_TYPES[1])

        # print MODULENAME,'  ------- showSimView \n\n'

        if self.basicSimulationData.sim:
            cellField = simObj.getPotts().getCellFieldG()
            # self.simulation.graphicsWidget.fillCellFieldData(cellField,"xy",0)

            # print "        BEFORE DRAW FIELD(1) FROM showSimView()"
            # time.sleep(5)

            self.__drawField()

            # print "        AFTER DRAW FIELD(1) FROM showSimView()"
            # time.sleep(5)



            # # Fields are available only after simulation is loaded
            self.setFieldTypes()
        else:
            # print "        BEFORE DRAW FIELD(2) FROM showSimView()"
            # if not self.simulation.dimensionChange():



            self.__drawField()

            self.setFieldTypesCML()
            # print "        AFTER DRAW FIELD(2) FROM showSimView()"

        #        import pdb; pdb.set_trace()

        #         Configuration.initFieldsParams(self.fieldTypes.keys())

        # # # self.__setCrossSection()

        print 'self.basicSimulationData=', dir(self.basicSimulationData)
        print 'self.basicSimulationData.fieldDim=', self.basicSimulationData.fieldDim
        print 'self.basicSimulationData.numberOfSteps=', self.basicSimulationData.numberOfSteps
        print 'self.basicSimulationData.sim=', self.basicSimulationData.sim

        self.setInitialCrossSection(self.basicSimulationData)
        print '   AFTER setInitialCrossSection'
        self.initGraphicsWidgetsFieldTypes()
        # self.closeTab.show()
        self.drawingAreaPrepared = True
        #        self.mainGraphicsWindow.parentWidget.move(400,300)   # temporarily moves, but jumps back

        # self.__layoutGraphicsWindows()

        # MDIFIX
        self.__restoreWindowsLayout()

    def __restoreWindowsLayout(self):

        windowsLayoutDict = Configuration.getSetting('WindowsLayout')
        print 'from settings windowsLayout = ', windowsLayoutDict

        # # first we convert window keys to integers 
        # int windowsLayoutDict.keys()

        # import time
        # time.sleep(2)

        # # first will check if window with id 0 was saved if, not than we are closing main window which should be open at this point
        # if str(0) not in windowsLayoutDict.keys():
        #     self.closeActiveSubWindowSlot()

        # first restore main window with id 0 - this window is the only window open at this point and it is open by default when simulation is started
        # that's why we have to treat it in a special way but only when we determine that windowsLayoutDict is not empty
        if len(windowsLayoutDict.keys()):
            try:
                windowDataDict0 = windowsLayoutDict[str(0)] # inside windowsLayoutDict windows are labeled using ints represented as strings

                from Graphics.GraphicsWindowData import GraphicsWindowData

                gwd = GraphicsWindowData()

                gwd.fromDict(windowDataDict0)

                if gwd.winType == GRAPHICS_WINDOW_LABEL:

                    graphicsWindow = self.lastActiveRealWindow
                    gfw = graphicsWindow.widget()
                    # gfw = self.lastActiveWindow
                    #
                    # graphicsWindow = self.findMDISubWindowForWidget(gfw)

                    graphicsWindow.resize(gwd.winSize)
                    graphicsWindow.move(gwd.winPosition)

                    gfw.applyGraphicsWindowData(gwd)



            except KeyError:
                # in case there is no main window with id 0 in the settings we kill the main window

                graphicsWindow = self.lastActiveRealWindow
                graphicsWindow.close()
                self.win_inventory.remove_from_inventory(graphicsWindow)

                # gfw = self.lastActiveWindow
                # graphicsWindow = self.findMDISubWindowForWidget(gfw)
                # graphicsWindow.close()
                # self.removeWindowFromRegistry(graphicsWindow)

                pass

        # restore graphics windows first
        for windowId, windowDataDict in windowsLayoutDict.iteritems():
            if windowId == str(0):
                continue

            # gfw = self.findMDISubWindowForWidget(self.lastActiveWindow)
            from Graphics.GraphicsWindowData import GraphicsWindowData

            gwd = GraphicsWindowData()

            gwd.fromDict(windowDataDict)

            if gwd.winType != GRAPHICS_WINDOW_LABEL:
                continue

            if gwd.sceneName not in self.fieldTypes.keys():
                continue  # we only create window for a sceneNames (e.g. fieldNames) that exist in the simulation


            graphicsWindow = self.addNewGraphicsWindow()
            gfw = graphicsWindow.widget()

            # mdiWindow = self.findMDISubWindowForWidget(gfw)
            graphicsWindow.resize(gwd.winSize)
            graphicsWindow.move(gwd.winPosition)

            gfw.applyGraphicsWindowData(gwd)

            print 'self.lastActiveWindow=',self.lastActiveWindow
            print 'gwd.winSize=',gwd.winSize
            # time.sleep(2)


            # # MDIFIX
            # if windowId != str(0):  # window with id one is created by addVTKWindowToWorkspaceo at this point there is already first window in the simulation
            #     self.addNewGraphicsWindow()


            # gfw = self.lastActiveWindow
            #
            # mdiWindow = self.findMDISubWindowForWidget(gfw)
            # mdiWindow.resize(gwd.winSize)
            # mdiWindow.move(gwd.winPosition)
            #
            # gfw.applyGraphicsWindowData(gwd)
            #
            # print 'self.lastActiveWindow=',self.lastActiveWindow
            # print 'gwd.winSize=',gwd.winSize
            # time.sleep(2)
            #

            # mdiWindow.setFixedSize(gwd.winSize)

        print ' PLOT WINDOW MANAGER  WINDOW LIST = ', self.plotManager.plotWindowList


        # import time
        # time.sleep(2)

        # for windowId, windowDataDict in windowsLayoutDict.iteritems():
        # if windowId != str(1):
        # # gfw = self.findMDISubWindowForWidget(self.lastActiveWindow)
        # from Graphics.GraphicsWindowData import GraphicsWindowData
        # gwd = GraphicsWindowData()
        # gwd.fromDict(windowDataDict)
        # if gwd.sceneName not in  self.fieldTypes.keys():
        # continue # we only create window for a scenNames (e.g. fieldNames) that exist in the simulation

        # self.addNewGraphicsWindow()
        # gfw = self.lastActiveWindow
        # gfw.applyGraphicsWindowData(gwd)

        # mdiWindow = self.findMDISubWindowForWidget (gfw)
        # # mdiWindow.setFixedSize(gwd.winSize)
        # mdiWindow.resize(gwd.winSize)
        # mdiWindow.move(gwd.winPosition)

    def setFieldTypesCML(self):
        # Add cell field
        self.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  #"CellField"

        self.fieldComboBox.clear()
        self.fieldComboBox.addItem("-- Field Type --")
        self.fieldComboBox.addItem("Cell_Field")

        for fieldName in self.simulation.fieldsUsed.keys():
            if fieldName != "Cell_Field":
                self.fieldTypes[fieldName] = self.simulation.fieldsUsed[fieldName]
                self.fieldComboBox.addItem(fieldName)


    def setFieldTypes(self):
        # Add cell field
        #        self.fieldTypes = {}
        simObj = self.mysim()
        if not simObj: return

        self.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  #"CellField" 

        # Add concentration fields How? I don't care how I got it at this time
        # print self.mysim.getPotts()
        concFieldNameVec = simObj.getConcentrationFieldNameVector()
        # # # concFieldNameVec = self.mysim.getConcentrationFieldNameVector()

        #putting concentration fields from simulator
        for fieldName in concFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this conc field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[1]


        #inserting extra scalar fields managed from Python script
        scalarFieldNameVec = self.fieldStorage.getScalarFieldNameVector()
        for fieldName in scalarFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this scalar field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[2]


        #inserting extra scalar fields cell levee managed from Python script
        scalarFieldCellLevelNameVec = self.fieldStorage.getScalarFieldCellLevelNameVector()
        for fieldName in scalarFieldCellLevelNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this scalar field (cell leve): ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[3]


        #inserting extra vector fields  managed from Python script
        vectorFieldNameVec = self.fieldStorage.getVectorFieldNameVector()
        for fieldName in vectorFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this vector field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[4]


        #inserting extra vector fields  cell level managed from Python script
        vectorFieldCellLevelNameVec = self.fieldStorage.getVectorFieldCellLevelNameVector()
        for fieldName in vectorFieldCellLevelNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this vector field (cell level): ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[5]

        #inserting custom visualization
        visDict = CompuCellSetup.customVisStorage.visDataDict

        for visName in visDict:
            self.fieldTypes[visName] = FIELD_TYPES[6]


    def showDisplayWidgets(self):
        #        print MODULENAME,' showDisplayWidgets'

        # This block of code simply checks to see if some plugins assoc'd with Vis are defined
        #        import CompuCellSetup, XMLUtils
        import XMLUtils
        #        print MODULENAME,' dir(XMLUtils)= ',dir(XMLUtils)
        if CompuCellSetup.cc3dXML2ObjConverter != None:
            self.pluginCOMDefined = False
            self.pluginFPPDefined = False

            #            print MODULENAME, 'CompuCellSetup.cc3dXML2ObjConverter != None; check FPP and COM plugins'
            self.root_element = CompuCellSetup.cc3dXML2ObjConverter.root
            elms = self.root_element.getElements("Plugin")
            elmList = XMLUtils.CC3DXMLListPy(elms)
            for elm in elmList:
                #                print 'dir(elm) = ',dir(elm)
                #                print "Element = ",elm.name  # -> Plugin
                pluginName = elm.getAttribute("Name")
                print "   pluginName = ", pluginName  # e.g. CellType, Contact, etc
                if pluginName == "FocalPointPlasticity":
                    #                    print '    yes, FPP is definded, enabling Vis menu item'
                    self.pluginFPPDefined = True
                    self.pluginCOMDefined = True  # if FPP is defined, COM will (implicitly) be defined

                if pluginName == "CenterOfMass":
                    self.pluginCOMDefined = True

                #            print MODULENAME,'showDisplayWidgets(): FPP= ',self.pluginFPPDefined, ', COM=',self.pluginCOMDefined

            # If appropriate, disable/enable Vis menu options
            if not self.pluginFPPDefined:
                self.FPPLinksAct.setEnabled(False)
                self.FPPLinksAct.setChecked(False)
                Configuration.setSetting("FPPLinksOn", False)

            #                self.FPPLinksColorAct.setEnabled(False)
            #                self.FPPLinksColorAct.setChecked(False)
            #                Configuration.setSetting("FPPLinksColorOn",False)
            else:
                #                print '    yes, FPP is definded, enabling Vis menu item'
                self.FPPLinksAct.setEnabled(True)
            #                self.FPPLinksColorAct.setEnabled(True)

            if not self.pluginCOMDefined:
                self.cellGlyphsAct.setEnabled(False)
                self.cellGlyphsAct.setChecked(False)
                Configuration.setSetting("CellGlyphsOn", False)
            else:
                self.cellGlyphsAct.setEnabled(True)

        #------------------
        if not self.mainGraphicsWindow: return

        self.mainGraphicsWindow.setStatusBar(self.__statusBar)

        #        self.mainGraphicsWindow._switchDim(True)   # rwh

        # if not self.keepOldTabs:
        #     pass
        #self.mainGraphicsWindow.setZoomItems(self.zitems)   # Set zoomFixed parameters
        self.mainGraphicsWindow.setZoomItems(self.zitems)  # Set zoomFixed parameters

        #if self.cellsAct.isChecked():          # Set "Cells" check box
        #    self.mainGraphicsWindow.showCells()

        if self.borderAct.isChecked():  # Vis menu "Cell Borders" check box
            self.mainGraphicsWindow.showBorder()
        else:
            self.mainGraphicsWindow.hideBorder()

        if self.clusterBorderAct.isChecked():  # Vis menu "Cluster Borders" check box
            self.mainGraphicsWindow.showClusterBorder()

        #---------------------
        if self.cellGlyphsAct.isChecked():  # Vis menu "Cell Glyphs"
            self.mainGraphicsWindow.showCellGlyphs()
        #            if not self.pluginCOMDefined:
        #                print "CenterOfMass plugin was NOT defined - toggling off Vis Cell Glyphs menu item"
        #                self.cellGlyphsAct.setChecked(False)
        #                self.cellGlyphsAct.setEnabled(False)
        #                Configuration.setSetting("CellGlyphsOn",False)
        #            else:
        #                self.cellGlyphsAct.setEnabled(True)
        #                self.mainGraphicsWindow.showCellGlyphs()

        #---------------------
        if self.FPPLinksAct.isChecked():  # Vis menu "FPP (Focal Point Plasticity) Links"
            self.mainGraphicsWindow.showFPPLinks()
        #            if not self.pluginFPPDefined:
        #                print "showDisplayWidgets(): FocalPointPlasticity plugin was NOT defined - toggling off Vis FPP Links menu item"
        #                self.FPPLinksAct.setChecked(False)
        #                self.FPPLinksAct.setEnabled(False)
        #                Configuration.setSetting("FPPLinksOn",False)
        #            else:
        #                self.FPPLinksAct.setEnabled(True)
        #                self.mainGraphicsWindow.showFPPLinks()

        #---------------------
        # if self.contourAct.isChecked():         
        # self.mainGraphicsWindow.showContours(True)
        # else:
        # self.mainGraphicsWindow.showContours(False)

        self.mainGraphicsWindow.setPlane(PLANES[0], 0)
        self.mainGraphicsWindow.currentDrawingObject.setPlane(PLANES[0], 0)

    def setParams(self):
        self.__paramsChanged()

    def __paramsChanged(self):
        #        print MODULENAME,'  __paramsChanged():  do a bunch of Config--.getSetting'
        self.__updateScreen = Configuration.getSetting("ScreenUpdateFrequency")
        self.__imageOutput = Configuration.getSetting("ImageOutputOn")
        self.__shotFrequency = Configuration.getSetting("SaveImageFrequency")
        self.__latticeOutputFlag = Configuration.getSetting("LatticeOutputOn")
        self.__latticeOutputFrequency = Configuration.getSetting("SaveLatticeFrequency")
        self.__projectLocation = str(Configuration.getSetting("ProjectLocation"))
        self.__outputLocation = str(Configuration.getSetting("OutputLocation"))

        # test if the sneaky user changed the output location
        #        prevOutputDir = self.__outputDirectory

        self.__outputDirectory = str(Configuration.getSetting("OutputLocation"))
        if Configuration.getSetting("OutputToProjectOn"):
            self.__outputDirectory = str(Configuration.getSetting("ProjectLocation"))

        #        print MODULENAME, '__paramsChanged(),  prevOutputDir, __outputDirectory= ', self.prevOutputDir, self.__outputDirectory

        if (
                    self.__imageOutput or self.__latticeOutputFlag) and self.mysim:  # has user requested output and is there a valid sim?
            if self.screenshotDirectoryName == "":  # haven't created any yet
                #                print MODULENAME, '__paramsChanged(), screenshotDirName empty;  calling createOutputDirs'
                self.createOutputDirs()
            elif self.prevOutputDir != self.__outputDirectory:  # test if the sneaky user changed the output location
                #                print MODULENAME, '__paramsChanged(),  prevOutput != Output;  calling createOutputDirs'
                self.createOutputDirs()

                # NOTE: if self.mysim == None (i.e. sim hasn't begun yet), then createOutputDirs() should be called in __loadSim

            #        print MODULENAME, '__paramsChanged(), screenshotDirName= ',self.screenshotDirectoryName # e.g. screenshotDirName= .../CC3DWorkspace2/cellsort_2D_xml_08_04_2011
            #        print MODULENAME, ' __paramsChanged(),  self.__outputDirectory, type()=',self.__outputDirectory,type(self.__outputDirectory)

        if self.simulation:
            self.updateSimPrefs()
        #            self.simulation.screenUpdateFrequency = self.__updateScreen
        #            self.simulation.screenshotFrequency = self.__shotFrequency

    def setZoomItems(self, zitems):
        self.zitems = zitems

    def zoomIn(self):
        if self.mainGraphicsWindow is not None:
            self.activeWindow().zoomIn()
            # self.mainGraphicsWindow.zoomIn()
            # print "Zoom in from TabView"

    def zoomOut(self):
        if self.mainGraphicsWindow is not None:
            self.activeWindow().zoomOut()
            # self.mainGraphicsWindow.zoomOut()

    def zoomFixed(self, val):
        if self.mainGraphicsWindow is not None:
            self.activeWindow().zoomFixed(val)
            # self.mainGraphicsWindow.zoomFixed(val)

    # # File name should be passed    
    def takeShot(self):
        if self.screenshotManager is not None:
            if self.threeDRB.isChecked():
                camera = self.mainGraphicsWindow.ren.GetActiveCamera()
                # print "CAMERA SETTINGS =",camera
                self.screenshotManager.add3DScreenshot(self.__fieldType[0], self.__fieldType[1], camera)
            else:
                planePositionTupple = self.mainGraphicsWindow.getPlane()
                # print "planePositionTupple=",planePositionTupple
                self.screenshotManager.add2DScreenshot(self.__fieldType[0], self.__fieldType[1], planePositionTupple[0],
                                                       planePositionTupple[1])

    def prepareSimulationView(self):
        if self.__fileName != "":
            file = QFile(self.__fileName)
            if file is not None:
                if self.mainGraphicsWindow is None:
                    # print "NO SIM TAB HERE"
                    self.showSimView(file)
                    # print 'ADDED SIM TAB'
                else:
                    # print "SIM TAB IITIALIZED"
                    # print 'file=',self.__fileName
                    # if self.simulation.dimensionChange():
                    # sys.exit()

                    self.__closeSim()
                    print 'BEFORE showSimView'
                    self.showSimView(file)
                    print 'AFTER showSimView'

        self.drawingAreaPrepared = True
        self.updateActiveWindowVisFlags()  # needed in case switching from one sim to another (e.g. 1st has FPP, 2nd doesn't)

    def __openLDSFile(self, fileName=None):
        filter = "Lattice Description Summary file  (*.dml )"  # self._getOpenFileFilter()

        #        defaultDir = str(Configuration.getSetting('OutputLocation'))
        defaultDir = self.__outputDirectory

        if not os.path.exists(defaultDir):
            defaultDir = os.getcwd()

        self.__fileName = QFileDialog.getOpenFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Open Lattice Description Summary file"),
            defaultDir,
            filter
            )
        # converting Qstring to python string    and normalizing path        
        self.__fileName = os.path.abspath(str(self.__fileName))
        from os.path import basename
        # setting text for main window (self.__parent) title bar 
        self.__parent.setWindowTitle(self.trUtf8(basename(self.__fileName) + " - CompuCell3D Player"))

        # Shall we inform the user?  Nah, screw 'em.
        #        msg = QMessageBox.warning(self, "Message","Toggling off image & lattice output in Preferences",
        #                          QMessageBox.Ok ,
        #                          QMessageBox.Ok)
        Configuration.setSetting("ImageOutputOn", False)
        Configuration.setSetting("LatticeOutputOn", False)


    def __openRecentSim(self):
        if self.simulationIsRunning:
            return

        action = self.sender()
        if isinstance(action, QAction):
            self.__fileName = str(action.data().toString())
        from os.path import basename
        # setting text for main window (self.__parent) title bar 
        self.__parent.setWindowTitle(self.trUtf8(basename(self.__fileName) + " - CompuCell3D Player"))

        import CompuCellSetup

        self.__fileName = os.path.abspath(self.__fileName)
        CompuCellSetup.simulationFileName = self.__fileName
        Configuration.setSetting("RecentFile", self.__fileName)
        Configuration.setSetting("RecentSimulations",
                                 self.__fileName)  #  each loaded simulation has to be passed to a function which updates list of recent files
        # Configuration.addItemToStrlist(item = os.path.abspath(self.__fileName),strListName = 'RecentSimulations',maxLength = Configuration.getSetting('NumberOfRecentSimulations'))

    def __openSim(self, fileName=None):
        # """
        # Public slot to open some files.

        # @param prog name of file to be opened (string or QString)
        # """

        # set the cwd of the dialog based on the following search criteria:
        #     1: Directory of currently active editor
        #     2: Directory of currently active project
        #     3: CWD  

        filter = "CompuCell3D simulation (*.cc3d *.xml *.py)"  # self._getOpenFileFilter()

        self.__screenshotDescriptionFileName = ""  # make screenshotDescriptionFile empty string

        defaultDir = str(Configuration.getSetting('ProjectLocation'))

        if not os.path.exists(defaultDir):
            defaultDir = os.getcwd()

        self.__fileName = QFileDialog.getOpenFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Open Simulation File"),
            defaultDir,
            filter
            )
        # converting Qstring to python string and normalizing path   
        self.__fileName = os.path.abspath(str(self.__fileName))

        print '__openSim: self.__fileName=', self.__fileName

        from os.path import basename
        # setting text for main window (self.__parent) title bar 
        self.__parent.setWindowTitle(self.trUtf8(basename(self.__fileName) + " - CompuCell3D Player"))

        import CompuCellSetup

        CompuCellSetup.simulationFileName = self.__fileName

        Configuration.setSetting("RecentFile", self.__fileName)
        Configuration.setSetting("RecentSimulations",
                                 self.__fileName)  #  each loaded simulation has to be passed to a function which updates list of recent files
        # Configuration.addItemToStrlist(item = os.path.abspath(self.__fileName),strListName = 'RecentSimulations',maxLength = Configuration.getSetting('NumberOfRecentSimulations'))


    def __saveSim(self):
        fullSimFileName = os.path.abspath(self.__fileName)
        simFilePath = os.path.dirname(fullSimFileName)

        filter = "CompuCell3D Simulation File (CC3DML) (*.xml )"  # self._getOpenFileFilter()
        cc3dmlFileName = QFileDialog.getSaveFileName( \
            self.ui,
            QApplication.translate('ViewManager', "CompuCell3D Simulation File (CC3DML)"),
            simFilePath,
            filter
            )

        #        import CompuCellSetup
        CompuCellSetup.cc3dXML2ObjConverter.root.saveXML(str(cc3dmlFileName))

    def __openScrDesc(self):
        filter = "Screenshot description file (*.sdfml)"  # self._getOpenFileFilter()
        self.__screenshotDescriptionFileName = QFileDialog.getOpenFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Open Screenshot Description File"),
            os.getcwd(),
            filter
            )

    def __saveScrDesc(self):
        # print "THIS IS __saveScrDesc"
        filter = "Screenshot Description File (*.sdfml )"  # self._getOpenFileFilter()
        self.screenshotDescriptionFileName = QFileDialog.getSaveFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Save Screenshot Description File"),
            os.getcwd(),
            filter
            )
        if self.screenshotManager:
            self.screenshotManager.writeScreenshotDescriptionFile(self.screenshotDescriptionFileName)

        print "self.screenshotDescriptionFileName=", self.screenshotDescriptionFileName

    def __closeSim(self):
        print "INSIDE closeSim"

    # Sets the attribute self.movieSupport
    def __setMovieSupport(self):
        self.movieSupport = False  # Is there vtkMPEG2Writer class in vtk module?
        vtkmod = inspect.getmembers(vtk, inspect.isclass)
        for i in range(len(vtkmod)):
            if vtkmod[i][0] == "vtkMPEG2Writer":
                self.movieSupport = True
                self.movieAct.setEnabled(True)
                return

        self.movieAct.setEnabled(False)

    def __checkMovieSupport(self, checked):
        if self.movieAct.isEnabled():
            if checked and self.movieSupport:
                # The ONLY place where the self.movieAct is checked!
                self.movieAct.setChecked(True)
            elif not self.movieSupport:
                self.movieAct.setChecked(False)
                QMessageBox.warning(self, "Movie Support Failed",
                                    "Sorry, your VTK library does not support \nmovie generation!",
                                    QMessageBox.Ok)


    def __checkCells(self, checked):

        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.updateActiveWindowVisFlags()
        if self.cellsAct.isEnabled():

            # MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                # if graphicsWidget.is_screenshot_widget: continue

                # self.updateActiveWindowVisFlags(graphicsWidget)

                try:
                    if checked:
                        print 'SHOWING CELLS ACTION'
                        graphicsWidget.showCells()
                        Configuration.setSetting('CellsOn',True)
                        self.cellsAct.setChecked(True)
                        win.activateWindow()
                    else:
                        print 'HIDING CELLS ACTION'
                        graphicsWidget.hideCells()
                        Configuration.setSetting('CellsOn',False)
                        self.cellsAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass
                self.updateActiveWindowVisFlags(graphicsWidget)


            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showCells()
            #             print 'SHOWING CELLS ACTION'
            #             Configuration.setSetting('CellsOn',True)
            #             self.cellsAct.setChecked(True)
            #         else:
            #             window.hideCells()
            #             Configuration.setSetting('CellsOn',False)
            #             print 'HIDING CELLS ACTION'
            #             self.cellsAct.setChecked(False)
            #     except AttributeError, e:
            #         pass
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkBorder(self, checked):
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        #        print '======== SimpleTabView.py:  __checkBorder(): checked =',checked
        #        print '             self.graphicsWindowDict=',self.graphicsWindowDict
        self.updateActiveWindowVisFlags()

        if self.borderAct.isEnabled():

            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()

                try:
                    if checked:
                        graphicsWidget.showBorder()
                        self.borderAct.setChecked(True)
                        win.activateWindow()
                    else:
                        graphicsWidget.hideBorder()
                        self.borderAct.setChecked(False)
                        win.activateWindow()
                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showBorder()
            #             self.borderAct.setChecked(True)
            #         else:
            #             window.hideBorder()
            #             self.borderAct.setChecked(False)
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkClusterBorder(self, checked):
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        #        print '======== SimpleTabView.py:  __checkClusterBorder: checked =',checked

        self.updateActiveWindowVisFlags()
        if self.clusterBorderAct.isEnabled():
            #MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                try:
                    if checked:
                        graphicsWidget.showClusterBorder()
                        self.clusterBorderAct.setChecked(True)
                        win.activateWindow()

                    else:
                        graphicsWidget.hideClusterBorder()
                        self.clusterBorderAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showClusterBorder()
            #             self.clusterBorderAct.setChecked(True)
            #
            #         else:
            #             window.hideClusterBorder()
            #             self.clusterBorderAct.setChecked(False)
            #
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkCellGlyphs(self, checked):
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.updateActiveWindowVisFlags()

        if self.cellGlyphsAct.isEnabled():
            if not self.pluginCOMDefined:
                QMessageBox.warning(self, "Message",
                                    "Warning: You have not defined a CenterOfMass plugin",
                                    QMessageBox.Ok)
                self.cellGlyphsAct.setChecked(False)
                Configuration.setSetting("CellGlyphsOn", False)

                self.simulation.drawMutex.unlock()
                return

            #MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                try:
                    if checked:
                        graphicsWidget.showCellGlyphs()
                        self.cellGlyphsAct.setChecked(True)
                        win.activateWindow()
                    else:
                        graphicsWidget.hideCellGlyphs()
                        self.cellGlyphsAct.setChecked(False)
                        win.activateWindow()
                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showCellGlyphs()
            #             self.cellGlyphsAct.setChecked(True)
            #
            #         else:
            #             window.hideCellGlyphs()
            #             self.cellGlyphsAct.setChecked(False)
            #
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkFPPLinks(self, checked):
        #        print MODULENAME,'  __checkFPPLinks, checked=',checked
        #        if checked and self.FPPLinksColorAct.isChecked():
        #            self.FPPLinksColorAct.setChecked(False)
        #            self.__checkFPPLinksColor(False)

        Configuration.setSetting("FPPLinksOn", checked)
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.updateActiveWindowVisFlags()

        if self.FPPLinksAct.isEnabled():

            if not self.pluginFPPDefined:
                QMessageBox.warning(self, "Message",
                                    "Warning: You have not defined a FocalPointPlasticity plugin",
                                    QMessageBox.Ok)
                self.FPPLinksAct.setChecked(False)
                Configuration.setSetting("FPPLinksOn", False)

                self.simulation.drawMutex.unlock()
                return

            #MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()

                try:
                    if checked:
                        graphicsWidget.showFPPLinks()
                        self.FPPLinksAct.setChecked(True)
                        win.activateWindow()
                    else:
                        graphicsWidget.hideFPPLinks()
                        self.FPPLinksAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showFPPLinks()
            #             self.FPPLinksAct.setChecked(True)
            #
            #         else:
            #             window.hideFPPLinks()
            #             self.FPPLinksAct.setChecked(False)
            #
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkFPPLinksColor(self, checked):
        #        print MODULENAME,'  __checkFPPLinksColor, checked=',checked
        if checked and self.FPPLinksAct.isChecked():
            self.FPPLinksAct.setChecked(False)
            self.__checkFPPLinks(False)
        #            if self.mainGraphicsWindow is not None:
        #                self.mainGraphicsWindow.hideFPPLinks()

        Configuration.setSetting("FPPLinksColorOn", checked)
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.updateActiveWindowVisFlags()

        if self.FPPLinksColorAct.isEnabled():

            if self.lastActiveWindow is not None:
                #                    Check for FPP plugin - improve to not even allow glyphs if no CoM
                #                    print '---- dir(self.simulation) =', dir(self.simulation)
                #                    print MODULENAME,'---- CoM = ',self.mainGraphicsWindow.drawModel2D.currentDrawingParameters.bsd.sim.getCC3DModuleData("Plugin","CenterOfMass")
                #                    print 'dir(self.mainGraphicsWindow)=',dir(self.mainGraphicsWindow)
                if not self.pluginFPPDefined:
                    QMessageBox.warning(self, "Message",
                                        "Warning: You have not defined a FocalPointPlasticity plugin",
                                        QMessageBox.Ok)
                    self.FPPLinksColorAct.setChecked(False)
                    Configuration.setSetting("FPPLinksColorOn", False)

                    self.simulation.drawMutex.unlock()
                    return

            #MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                try:
                    if checked:
                        graphicsWidget.showFPPLinksColor()
                        self.FPPLinksColorAct.setChecked(True)
                        win.activateWindow()
                    else:
                        graphicsWidget.hideFPPLinksColor()
                        self.FPPLinksColorAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showFPPLinksColor()
            #             self.FPPLinksColorAct.setChecked(True)
            #         else:
            #             window.hideFPPLinksColor()
            #             self.FPPLinksColorAct.setChecked(False)
            #
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)

        self.simulation.drawMutex.unlock()


    def __checkContour(self, checked):
        if self.contourAct.isEnabled():

            #MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                try:
                    if checked:
                        graphicsWidget.showContours(True)
                        self.contourAct.setChecked(True)
                        win.activateWindow()
                    else:
                        graphicsWidget.showContours(False)
                        self.contourAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass

                self.updateActiveWindowVisFlags(graphicsWidget)

            # for windowName, window in self.graphicsWindowDict.iteritems():
            #     try:
            #         if checked:
            #             window.showContours(True)
            #             self.contourAct.setChecked(True)
            #         else:
            #             windos.showContours(False)
            #             self.contourAct.setChecked(False)
            #
            #     except AttributeError, e:
            #         pass
            #
            #     self.updateActiveWindowVisFlags(window)


    def __checkLimits(self, checked):
        pass

    def __checkCC3DOutput(self, checked):
        Configuration.setSetting("CC3DOutputOn", checked)

    def __showConfigDialog(self, pageName=None):
        """
        Private slot to set the configurations.
        @param pageName name of the configuration page to show (string or QString)
        """
        activeFieldNamesList = []
        for idx in range(len(self.fieldTypes)):
            fieldName = self.fieldTypes.keys()[idx]
            if fieldName != 'Cell_Field':  # rwh: dangerous to hard code this field name
                # self.dlg.fieldComboBox.addItem(fieldName)   # this is where we set the combobox of field names in Prefs
                activeFieldNamesList.append(str(fieldName))

        Configuration.setUsedFieldNames(activeFieldNamesList)

        dlg = ConfigurationDialog(self, 'Configuration', True)
        self.dlg = dlg  # rwh: to allow enable/disable widgets in Preferences


        #        print '   dir(self.dlg)=',dir(self.dlg)
        if len(self.fieldTypes) < 2:
            self.dlg.tab_field.setEnabled(False)
        else:
            self.dlg.tab_field.setEnabled(True)

        self.dlg.fieldComboBox.clear()

        # print 'activeFieldNamesList=',activeFieldNamesList
        # print 'self.dlg.fieldComboBox.count()=',self.dlg.fieldComboBox.count()

        # import time
        # time.sleep(2)        
        for fieldName in activeFieldNamesList:
            # print 'fieldName=',fieldName
            self.dlg.fieldComboBox.addItem(fieldName)  # this is where we set the combobox of field names in Prefs

            # print 'self.dlg.fieldComboBox.count()=',self.dlg.fieldComboBox.count()


            # # # activeFieldNamesList = []

            # # # for idx in range(len(self.fieldTypes) ):
            # # # fieldName = self.fieldTypes.keys()[idx]
            # # # if fieldName != 'Cell_Field':  # rwh: dangerous to hard code this field name
            # # # self.dlg.fieldComboBox.addItem(fieldName)   # this is where we set the combobox of field names in Prefs
            # # # activeFieldNamesList.append(str(fieldName))

        # # # Configuration.setUsedFieldNames (activeFieldNamesList)

        self.connect(dlg, SIGNAL('configsChanged'), self.__configsChanged)
        dlg.show()
        #        dlg.showConfigurationPageByName("default") #showConfigurationDefaultPage()#

        dlg.exec_()
        QApplication.processEvents()

        if dlg.result() == QDialog.Accepted:
            # Saves changes from all configuration pages!
            #            dlg.setPreferences()
            Configuration.syncPreferences()
            self.__configsChanged()  # Explicitly calling signal 'configsChanged'

    def __generatePIFFromCurrentSnapshot(self):
        if self.__viewManagerType == "CMLResultReplay":
            self.__generatePIFFromVTK()
        else:
            self.__generatePIFFromRunningSimulation()

    def __generatePIFFromRunningSimulation(self):
        if self.pauseAct.isEnabled():
            self.__pauseSim()

        fullSimFileName = os.path.abspath(self.__fileName)
        simFilePath = os.path.dirname(fullSimFileName)

        filter = "Choose PIF File Name (*.piff *.txt )"  # self._getOpenFileFilter()
        pifFileName = QFileDialog.getSaveFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Save PIF File As ..."),
            simFilePath,
            filter
            )
        self.simulation.generatePIFFromRunningSimulation(str(pifFileName))

    def __generatePIFFromVTK(self):
        if self.pauseAct.isEnabled():
            self.__pauseSim()

        fullSimFileName = os.path.abspath(self.__fileName)
        simFilePath = os.path.dirname(fullSimFileName)

        filter = "Choose PIF File Name (*.piff *.txt )"  # self._getOpenFileFilter()
        pifFileName = QFileDialog.getSaveFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Save PIF File As ..."),
            simFilePath,
            filter
            )

        self.simulation.generatePIFFromVTK(self.simulation.currentFileName, str(pifFileName))


    def __configsChanged(self):
        """
        Private slot to handle a change of the preferences.
        """
        self.emit(SIGNAL('configsChanged'))

    def setModelEditor(self, modelEditor):
        self.__modelEditor = modelEditor

    # Class that checks if it is safe to close the simulation view
    def isSafeToCloseSim(self):
        msg = QMessageBox.warning(self, "Close Simulation", \
                                  "Are you sure you want to close the Simulation?", \
                                  QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)

        if msg == QMessageBox.Ok:
            return True
        else:
            return False

    def __createStatusBar(self):
        self.__statusBar = self.__parent.statusBar()
        self.mcSteps = QLabel()
        self.mcSteps.setAutoFillBackground(True)
        mcp = QPalette()
        mcp.setColor(QPalette.Window, QColor("white"))  # WindowText
        mcp.setColor(QPalette.WindowText, QColor("blue"))
        self.mcSteps.setPalette(mcp)

        self.conSteps = QLabel()
        self.conSteps.setAutoFillBackground(True)
        cp = QPalette()
        cp.setColor(QPalette.Window, QColor("white"))  # WindowText
        cp.setColor(QPalette.WindowText, QColor("blue"))
        self.conSteps.setPalette(cp)

        self.warnings = QLabel()
        self.warnings.setAutoFillBackground(True)
        cp = QPalette()
        cp.setColor(QPalette.Window, QColor("white"))  # WindowText
        cp.setColor(QPalette.WindowText, QColor("red"))
        self.warnings.setPalette(cp)


        self.__statusBar.addWidget(self.mcSteps)
        self.__statusBar.addWidget(self.conSteps)
        self.__statusBar.addWidget(self.warnings)

        # def loadCustomPlayerSettings(self,_root_element):
        # import XMLUtils
        # import CC3DXML
        # from XMLUtils import dictionaryToMapStrStr as d2mss
        # playerSettingsElement = _root_element.getFirstElement("Plugin",d2mss({"Name":"PlayerSettings"}))

    # #        print '--------------------------------\n'
    # #        print 'type(playerSettingsElement)=',type(playerSettingsElement)
    # #        print 'dir(playerSettingsElement)=',dir(playerSettingsElement)
    # #        print 'playerSettingsElement.getNumberOfChildren()=',playerSettingsElement.getNumberOfChildren()
    # if playerSettingsElement:
    # winList = XMLUtils.CC3DXMLListPy(playerSettingsElement.getElements("MainWindow"))
    # for myWin in winList:
    # attrKeys = myWin.getAttributes().keys()  # ['CameraClippingRange', 'CameraDistance', 'CameraFocalPoint', 'CameraPos', 'CameraViewUp', 'WindowNumber']
    # #                print '------ MainWindow: attrKeys=',attrKeys
    # self.setWindowView(myWin,attrKeys)
    # #            self.mainGraphicsWindow._xyChecked(True)

    # winList = XMLUtils.CC3DXMLListPy(playerSettingsElement.getElements("NewWindow"))
    # for myWin in winList:
    # self.addNewGraphicsWindow()   #rwh
    # #                camera3D = self.lastActiveWindow.getCamera3D()

    # #                print 'w.getAttributes()=',w.getAttributes()
    # #                print 'dir(w.getAttributes())=',dir(w.getAttributes())
    # attrKeys = myWin.getAttributes().keys()  # ['CameraClippingRange', 'CameraDistance', 'CameraFocalPoint', 'CameraPos', 'CameraViewUp', 'WindowNumber']
    # self.setWindowView(myWin,attrKeys)


    # visualControlElement = playerSettingsElement.getFirstElement("VisualControl")
    # if visualControlElement:
    # #                print 'type(visualControlElement)=',type(visualControlElement)
    # #                print 'visualControlElement=',visualControlElement
    # #                print 'dir(visualControlElement)=',dir(visualControlElement)
    # #                print 'visualControlElement.getName()=',visualControlElement.getName()
    # #                print 'visualControlElement.attributes=',visualControlElement.attributes
    # #                print 'visualControlElement.getNumberOfChildren()=',visualControlElement.getNumberOfChildren()
    # #                print 'visualControlElement.getElements()=',visualControlElement.getElements()
    # #                print 'visualControlElement.getAttributes()=',visualControlElement.getAttributes()
    # validAttrs = ("ScreenUpdateFrequency",  "NoOutput","ImageOutput", "ScreenshotFrequency","ImageFrequency",
    # "LatticeOutput","LatticeFrequency")

    # for vcAttr in XMLUtils.XMLAttributeList(visualControlElement):
    # if str(vcAttr[0]) not in validAttrs:
    # print "\n-------\nERROR in loadCustomPlayerSettings:  VisualControl attribute '",vcAttr[0],"' is invalid"
    # print 'Valid attributes are ',validAttrs
    # print '--------\n'

    # #  NOTE: do NOT do an if-elif block here!
    # if visualControlElement.findAttribute("ScreenUpdateFrequency"):
    # screenUpdateFrequency = visualControlElement.getAttributeAsUInt("ScreenUpdateFrequency")
    # Configuration.setSetting("ScreenUpdateFrequency",screenUpdateFrequency)

    # if visualControlElement.findAttribute("NoOutput"):  # trying to deprecate
    # noOutput = visualControlElement.getAttributeAsBool("NoOutput")
    # imageOut = not noOutput
    # Configuration.setSetting("ImageOutputOn",imageOut)
    # if visualControlElement.findAttribute("ImageOutput"):   # replaces previous
    # imageOut = visualControlElement.getAttributeAsBool("ImageOutput")
    # Configuration.setSetting("ImageOutputOn",imageOut)

    # if visualControlElement.findAttribute("ScreenshotFrequency"):  # trying to deprecate
    # scrFreq = visualControlElement.getAttributeAsUInt("ScreenshotFrequency")
    # Configuration.setSetting("SaveImageFrequency",scrFreq)
    # if visualControlElement.findAttribute("ImageFrequency"):  # replaces previous
    # scrFreq = visualControlElement.getAttributeAsUInt("ImageFrequency")
    # Configuration.setSetting("SaveImageFrequency",scrFreq)

    # if visualControlElement.findAttribute("LatticeOutput"):
    # latticeOut = visualControlElement.getAttributeAsBool("LatticeOutput")
    # Configuration.setSetting("LatticeOutputOn",latticeOut)
    # if visualControlElement.findAttribute("LatticeFrequency"):
    # latticeFreq = visualControlElement.getAttributeAsUInt("LatticeFrequency")
    # Configuration.setSetting("SaveLatticeFrequency",latticeFreq)

    # # if visualControlElement.findAttribute("ClosePlayerAfterSimulationDone"):
    # # closePlayerAfterSimulationDone=visualControlElement.getAttributeAsBool("ClosePlayerAfterSimulationDone")
    # # Configuration.setSetting("ClosePlayerAfterSimulationDone",closePlayerAfterSimulationDone)
    # self.__paramsChanged()

    # borderElement=playerSettingsElement.getFirstElement("Border")
    # if borderElement:
    # if borderElement.findAttribute("BorderColor"):
    # # print "borderElement"
    # borderColor=borderElement.getAttribute("BorderColor")
    # Configuration.setSetting("Border", QColor(borderColor))

    # if borderElement.findAttribute("BorderOn"):
    # borderOn=borderElement.getAttributeAsBool("BorderOn")
    # Configuration.setSetting("BordersOn", borderOn)
    # if borderOn:
    # self.borderAct.setChecked(True)
    # else:
    # self.borderAct.setChecked(False)

    # cellColorsList = XMLUtils.CC3DXMLListPy(playerSettingsElement.getElements("Cell"))
    # #            typeColorMap = {}
    # typeColorMap = Configuration.getSetting("TypeColorMap")  # start out with the given (default) cell type colormap
    # #            print MODULENAME,'  loadCustomPlayerSettings, typeColorMap =',typeColorMap
    # cellColorsListLength = 0
    # for cellElement in cellColorsList:
    # cellColorsListLength += 1

    # if cellColorsListLength:
    # #                print MODULENAME,'-----  cellColorsListLength=',cellColorsListLength
    # #                import pdb; pdb.set_trace()
    # for cellElement in cellColorsList:
    # cellType = cellElement.getAttributeAsUInt("Type")
    # #                    print 'type(cellType), cellType =',type(cellType),cellType
    # cellColor = cellElement.getAttribute("Color")
    # #                    print 'type(cellColor), cellColor =',type(cellColor),cellColor
    # if cellColor[0] == '#':   # handle a hex specification (Michael Rountree likes to do this)
    # r,g,b = cellColor[1:3], cellColor[3:5], cellColor[5:7]
    # r,g,b = [int(n, 16) for n in (r, g, b)]
    # #                        print '   type(r)=',type(r)
    # #                        print '   r,g,b=',r,g,b
    # typeColorMap[cellType] = QColor(r,g,b)
    # else:
    # cellColor = string.lower(cellColor)
    # typeColorMap[cellType] = QColor(cellColor)
    # # print "GOT CUSTOM COLORS"
    # # for cellType in typeColorMap.keys():
    # # print "typeColorMap=",typeColorMap
    # #                Configuration.setSetting("CustomTypeColorMap", typeColorMap)
    # Configuration.setSetting("TypeColorMap", typeColorMap)
    # for windowName,window in self.graphicsWindowDict.items():
    # window.populateLookupTable()


    # # self.mainGraphicsWindow.populateLookupTable()
    # # self.graphics3D.populateLookupTable()

    # typesInvisibleIn3DElement = playerSettingsElement.getFirstElement("TypesInvisibleIn3D")
    # if typesInvisibleIn3DElement:
    # print MODULENAME,' type(typesInvisibleIn3DElement.getAttribute("Types")) = ',type(typesInvisibleIn3DElement.getAttribute("Types"))
    # print MODULENAME,' typesInvisibleIn3DElement.getAttribute("Types") = ',typesInvisibleIn3DElement.getAttribute("Types")
    # Configuration.setSetting("Types3DInvisible", typesInvisibleIn3DElement.getAttribute("Types"))

    # self.saveSettings = True # by default we will save settings each time we exit player

    # settingsElement = playerSettingsElement.getFirstElement("Settings")
    # if settingsElement:
    # self.saveSettings = settingsElement.getAttributeAsBool("SaveSettings")

    # projection2DElement = playerSettingsElement.getFirstElement("Project2D")
    # if projection2DElement:
    # if projection2DElement.findAttribute("XYProj"):
    # zPos = projection2DElement.getAttributeAsUInt("XYProj")
    # print MODULENAME,'  loadCustomPlayerSettings(): XYProj, zPos =',zPos  #rwh
    # if zPos >= self.xySB.minimum() and zPos <= self.xySB.maximum():
    # self.mainGraphicsWindow._xyChecked(True)
    # self.mainGraphicsWindow._projSpinBoxChanged(zPos)

    # elif projection2DElement.findAttribute("XZProj"):
    # yPos = projection2DElement.getAttributeAsUInt("XZProj")
    # if yPos >= self.xzSB.minimum() and yPos <= self.xzSB.maximum():
    # self.mainGraphicsWindow._xzChecked(True)
    # self.mainGraphicsWindow._projSpinBoxChanged(yPos)

    # elif projection2DElement.findAttribute("YZProj"):
    # xPos = projection2DElement.getAttributeAsUInt("YZProj")
    # if xPos >= self.yzSB.minimum() and xPos <= self.yzSB.maximum():
    # self.mainGraphicsWindow._yzChecked(True)
    # self.mainGraphicsWindow._projSpinBoxChanged(xPos)
    # else:   # rwh: would like to deprecate this and duplicate the above syntax for 3D windows
    # view3DElement = playerSettingsElement.getFirstElement("View3D")
    # if view3DElement:
    # cameraCippingRange=None
    # cameraFocalPoint=None
    # cameraPosition=None
    # cameraViewUp=None

    # clippingRangeElement=view3DElement.getFirstElement("CameraClippingRange")
    # if clippingRangeElement:
    # cameraClippingRange=[float(clippingRangeElement.getAttribute("Min")),float(clippingRangeElement.getAttribute("Max"))]

    # focalPointElement=view3DElement.getFirstElement("CameraFocalPoint")
    # if focalPointElement:
    # cameraFocalPoint=[float(focalPointElement.getAttribute("x")),float(focalPointElement.getAttribute("y")),float(focalPointElement.getAttribute("z"))]

    # positionElement=view3DElement.getFirstElement("CameraPosition")
    # if positionElement:
    # cameraPosition=[float(positionElement.getAttribute("x")),float(positionElement.getAttribute("y")),float(positionElement.getAttribute("z"))]

    # viewUpElement=view3DElement.getFirstElement("CameraViewUp")
    # if viewUpElement:
    # cameraViewUp=[float(viewUpElement.getAttribute("x")),float(viewUpElement.getAttribute("y")),float(viewUpElement.getAttribute("z"))]

    # camera3D=self.mainGraphicsWindow.getCamera3D()
    # if cameraCippingRange:
    # camera3D.SetClippingRange(cameraCippingRange)
    # if  cameraFocalPoint:
    # camera3D.SetFocalPoint(cameraFocalPoint)
    # if  cameraPosition:
    # camera3D.SetPosition(cameraPosition)
    # if  cameraViewUp:
    # camera3D.SetViewUp(cameraViewUp)

    # self.mainGraphicsWindow._switchDim(True)
        