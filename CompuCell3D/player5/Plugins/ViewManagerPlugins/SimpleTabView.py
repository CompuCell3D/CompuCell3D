from __future__ import with_statement
# enabling with statement in python 2.5

# -*- coding: utf-8 -*-
import os, sys
import re
import inspect
import string
import time
from collections import OrderedDict
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtXml import *

from enums import *

from Messaging import stdMsg, dbgMsg, pd, errMsg, setDebugging

# setDebugging(1)

FIELD_TYPES = (
    "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel", "CustomVis")
PLANES = ("xy", "xz", "yz")

MODULENAME = '---- SimpleTabView.py: '

from PyQt5.QtCore import QCoreApplication

# from ViewManager.ViewManager import ViewManager
from ViewManager.SimpleViewManager import SimpleViewManager
from  Graphics.GraphicsFrameWidget import GraphicsFrameWidget

# from Utilities.QVTKRenderWidget import QVTKRenderWidget
from Utilities.SimModel import SimModel
from Configuration.ConfigurationDialog import ConfigurationDialog
import Configuration
import DefaultSettingsData as settings_data


import DefaultData

from Simulation.CMLResultReader import CMLResultReader
from Simulation.SimulationThread import SimulationThread
# from Simulation.SimulationThread1 import SimulationThread1

import ScreenshotManager
import vtk

# turning off vtkWindows console output
vtk.vtkObject.GlobalWarningDisplayOff()

from RollbackImporter import RollbackImporter

try:
    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)
    import CompuCellSetup
except:
    print 'STView: sys.path=', sys.path

# *********** TODO
# 1. add example with simplified plots
# 2. ADD WEAKREF TO PLOT FRAME WIDGET< PLOT INTERFACE CARTESIAN ETC...
# 4. CHECK IF IT IS NECESSARY TO FIX CLOSE EVENTS AND REMOVE GRAPHICS WIDGET PLOT WIDGET FROM ANY TYPE OF REGISTRIES -
# for QDockWindows this is taken care of , for MDI have to implement automatic removal from registries
# 7. get rid of multiple calls to pde from twedit++
# 9. Add parameter annotation self.lengthConstraintPlugin.setLengthConstraintData(cell,20,20)
# 10. figure out how to use named attributes for swig generated functions- quick way is to extend plugin object
# with python call which in turn calls swig annotated fcn
# 11. update mitosis generation add clone attributes to twedit



# from MainAreaMdi import MainArea
if Configuration.getSetting('FloatingWindows'):
    from MainArea import MainArea
else:
    from MainAreaMdi import MainArea


# class SimpleTabView(QMdiArea, SimpleViewManager):
class SimpleTabView(MainArea, SimpleViewManager):
    configsChanged = pyqtSignal()

    def __init__(self, parent):

        self.__parent = parent  # QMainWindow -> UI.UserInterface
        self.UI = parent

        SimpleViewManager.__init__(self, parent)
        MainArea.__init__(self, stv=self, ui=parent)

        self.__createStatusBar()
        self.__setConnects()

        # holds ptr (stored as long int) to original cerr stream buffer
        self.cerrStreamBufOrig = None

        # turning off vtk debug output. This requires small modification to the vtk code itself.
        # Files affected vtkOutputWindow.h vtkOutputWindow.cxx vtkWin32OutputWindow.h vtkWin32OutputWindow.cxx
        if hasattr(vtk.vtkOutputWindow, "setOutputToWindowFlag"):
            vtkOutput = vtk.vtkOutputWindow.GetInstance()
            vtkOutput.setOutputToWindowFlag(False)

        self.rollbackImporter = None

        from PlotManagerSetup import createPlotManager

        # stores parsed command line arguments
        self.cml_args = None

        self.useVTKPlots = False
        # object responsible for creating/managing plot windows so they're accessible from steppable level
        # TODO FIX IT
        # self.plotManager = None
        self.plotManager = createPlotManager(self, self.useVTKPlots)

        self.fieldTypes = {}

        self.pluginTab = None
        self.mysim = None

        self.simulation = None  # gets assigned to SimulationThread down in prepareForNewSimulation()
        self.screenshotManager = None
        self.zitems = []
        self.__fileName = ""  # simulation model filename
        self.__windowsXMLFileName = ""

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

        self.screenshotNumberOfDigits = 10  # this determines how many digits screenshot number of screenshot file name should have

        self.graphicsWindowVisDict = OrderedDict()  # stores visualization settings for each open window

        # self.lastActiveWindow = None
        self.lastPositionMainGraphicsWindow = None
        self.newWindowDefaultPlane = None

        self.cc3dSimulationDataHandler = None

        # for more information on QSignalMapper see Mark Summerfield book "Rapid GUI Development with PyQt"
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mapped.connect(self.setActiveSubWindowCustomSlot)
        # self.connect(self.windowMapper, SIGNAL("mapped(QWidget*)"), self.setActiveSubWindowCustomSlot)


        self.prepareForNewSimulation(_forceGenericInitialization=True)
        #        print MODULENAME,'__init__:   after prepareForNewSimulation(),  self.mysim = ',self.mysim

        self.setParams()
        # self.keepOldTabs = False  #this flag sets if tabs should be removed before creating new one or not
        self.mainGraphicsWidget = None  # vs.  lastActiveWindow

        # determine if some relevant plugins are defined in the model
        self.pluginFPPDefined = False  # FocalPointPlasticity
        self.pluginCOMDefined = False  # CenterOfMass
        # is there a better way to check for plugins being defined?
        # mainGraphicsWindow.drawModel2D.currentDrawingParameters.bsd.sim.getCC3DModuleData("Plugin","FocalPointPlasticity"):

        # Note: we cannot check the plugins here as CompuCellSetup.cc3dXML2ObjConverter.root is not defined

        # nextSimulation holds the name of the file that will be inserted as a new simulation to run after current simulation gets stopped
        self.nextSimulation = ""
        self.dlg = None

        # parameter scan variables
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

        # Here we are checking for new version - notice we use check interval in order not to perform version checks
        # too often. Default check interval is 7 days
        self.check_version(check_interval=7)

    def getSimFileName(self):
        '''
        Returns active cc3d project filename
        :return: str
        '''

        return self.__fileName

    def getWindowsXMLFileName(self):
        '''
        Deprecated - returns windows XML file name
        :return: str
        '''
        return self.__windowsXMLFileName

    def updateRecentFileMenu(self):
        '''
        Updates recent simulations File menu - called on demand only
        :return: None
        '''
        menusDict = self.__parent.getMenusDictionary()
        rencentSimulationsMenu = menusDict["recentSimulations"]
        rencentSimulationsMenu.clear()
        recentSimulations = Configuration.getSetting("RecentSimulations")

        simCounter = 1
        for simulationFileName in recentSimulations:
            actionText = self.tr("&%1 %2").format(simCounter, simulationFileName)
            # action=rencentSimulationsMenu.addAction(actionText)
            action = QAction("&%d %s " % (simCounter, simulationFileName), self)
            rencentSimulationsMenu.addAction(action)
            action.setData(QVariant(simulationFileName))
            # self.connect(action, SIGNAL("triggered()"), self.__openRecentSim)
            action.triggered.connect(self.__openRecentSim)

            simCounter += 1
        return

    def setActiveSubWindowCustomSlot(self, window):
        '''
        Activates window
        :param window: QDockWidget or QMdiWindow instance
        :return:None
        '''

        # print 'setActiveSubWindowCustomSlot = ', window
        self.lastActiveRealWindow = window
        # self.lastClickedRealWindow = window
        self.lastActiveRealWindow.activateWindow()

    def updateActiveWindowVisFlags(self, window=None):
        '''
        Updates graphics visualization dictionary - checks if border, cells fpp links etc should be drawn
        :param window: QDockWidget or QMdiWindow instance - but only for graphics windows
        :return: None
        '''



        try:
            if window:
                dictKey = window.winId().__int__()
            else:
                dictKey = self.lastActiveRealWindow.widget().winId().__int__()
        except StandardError:
            print MODULENAME, 'updateActiveWindowVisFlags():  Could not find any open windows. Ignoring request'
            return

        self.graphicsWindowVisDict[dictKey] = (self.cellsAct.isChecked(), self.borderAct.isChecked(), \
                                               self.clusterBorderAct.isChecked(), self.cellGlyphsAct.isChecked(),
                                               self.FPPLinksAct.isChecked())

    def updateWindowMenu(self):
        '''
        Invoked whenever 'Window' menu is clicked. It does NOT modify lastActiveWindow directly (setActiveSubWindowCustomSlot does)
        :return:None
        '''

        menusDict = self.__parent.getMenusDictionary()
        windowMenu = menusDict["window"]
        windowMenu.clear()
        windowMenu.addAction(self.newGraphicsWindowAct)
        # windowMenu.addAction(self.newPlotWindowAct)
        if self.MDI_ON:
            windowMenu.addAction(self.tileAct)
            windowMenu.addAction(self.cascadeAct)
            windowMenu.addAction(self.minimizeAllGraphicsWindowsAct)
            windowMenu.addAction(self.restoreAllGraphicsWindowsAct)
        windowMenu.addSeparator()

        windowMenu.addAction(self.closeActiveWindowAct)
        # windowMenu.addAction(self.closeAdditionalGraphicsWindowsAct)
        windowMenu.addSeparator()

        # adding graphics windows
        counter = 0

        # for windowName in self.graphicsWindowDict.keys():
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            # for win in self.win_inventory.values():
            graphicsWidget = win.widget()

            if not graphicsWidget:  # happens with screenshot widget after simulation closes
                continue

            if graphicsWidget.is_screenshot_widget:
                continue

            # actionText = self.tr("&%1. %2").format(counter + 1, win.windowTitle())
            actionText = str("&{0}. {1}").format(counter + 1, win.windowTitle())

            action = windowMenu.addAction(actionText)
            action.setCheckable(True)
            # myFlag = self.lastActiveRealWindow == graphicsWidget
            myFlag = self.lastActiveRealWindow == win
            action.setChecked(myFlag)

            # todo
            # self.connect(action, SIGNAL("triggered()"), self.windowMapper, SLOT("map()"))
            action.triggered.connect(self.windowMapper.map)

            self.windowMapper.setMapping(action, win)
            counter += 1

        for winId, win in self.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            # actionText = self.tr("&%1. %2").arg(counter + 1).arg(win.windowTitle())
            actionText = self.tr("&{0}. {1}").format(counter + 1, win.windowTitle())

            action = windowMenu.addAction(actionText)
            action.setCheckable(True)
            # myFlag = self.lastActiveRealWindow == graphicsWidget
            myFlag = self.lastActiveRealWindow == win
            action.setChecked(myFlag)

            # self.connect(action, SIGNAL("triggered()"), self.windowMapper, SLOT("map()"))
            action.triggered.connect(self.windowMapper.map)
            self.windowMapper.setMapping(action, win)
            counter += 1

    def addNewGraphicsWindow(self):
        '''
        callback method to create additional ("Aux") graphics windows
        :return: None
        '''
        print MODULENAME, '--------- addNewGraphicsWindow() '

        if not self.simulationIsRunning:
            return
        self.simulation.drawMutex.lock()

        newWindow = GraphicsFrameWidget(parent=None, originatingWidget=self)

        newWindow.setZoomItems(self.zitems)  # Set zoomFixed parameters

        newWindow.hide()

        self.configsChanged.connect(newWindow.draw2D.configsChanged)
        self.configsChanged.connect(newWindow.draw3D.configsChanged)

        # self.connect(self, SIGNAL('configsChanged'), newWindow.draw2D.configsChanged)
        # self.connect(self, SIGNAL('configsChanged'), newWindow.draw3D.configsChanged)

        newWindow.readSettings()  # Graphics/MVCDrawViewBase.py
        # setting up plane tuple based on window number 1
        # plane=self.windowDict[1].getPlane()
        # newWindow.setPlane(plane[0],plane[1])

        # each new window is painted in 2D mode xy projection with z coordinate set to fieldDim.z/2
        self.newWindowDefaultPlane = ("XY", self.basicSimulationData.fieldDim.z / 2)
        newWindow.setPlane(self.newWindowDefaultPlane[0], self.newWindowDefaultPlane[1])

        newWindow.currentDrawingObject.setPlane(self.newWindowDefaultPlane[0], self.newWindowDefaultPlane[1])

        # self.simulation.setGraphicsWidget(self.mainGraphicsWindow)
        # self.mdiWindowDict[self.windowCounter] = self.addSubWindow(newWindow)
        mdiWindow = self.addSubWindow(newWindow)

        # MDIFIX
        self.lastActiveRealWindow = mdiWindow

        # this happens when during restoration graphics window with id 0 had to be closed
        if self.mainGraphicsWidget is None:
            self.mainGraphicsWidget = mdiWindow.widget()

        self.updateActiveWindowVisFlags()

        newWindow.show()

        self.simulation.drawMutex.unlock()

        newWindow.setConnects(self)  # in GraphicsFrameWidget
        newWindow.setInitialCrossSection(self.basicSimulationData)
        newWindow.setFieldTypesComboBox(self.fieldTypes)

        suggested_win_pos = self.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdiWindow.move(suggested_win_pos)

        return mdiWindow

    def addVTKWindowToWorkspace(self):  #
        '''
        just called one time, for initial graphics window  (vs. addNewGraphicsWindow())
        :return: None
        '''

        self.mainGraphicsWidget = GraphicsFrameWidget(parent=None, originatingWidget=self)

        # we make sure that first graphics window is positioned in the left upper corner
        # NOTE: we have to perform move prior to calling addSubWindow. or else we will get distorted window
        if self.lastPositionMainGraphicsWindow is not None:
            self.mainGraphicsWidget.move(self.lastPositionMainGraphicsWindow)
        else:
            self.lastPositionMainGraphicsWindow = self.mainGraphicsWidget.pos()

        self.mainGraphicsWidget.show()


        # todo ok
        # self.mainGraphicsWidget.setShown(False)

        # self.mainGraphicsWidget.hide()
        # return

        self.configsChanged.connect(self.mainGraphicsWidget.draw2D.configsChanged)
        self.configsChanged.connect(self.mainGraphicsWidget.draw3D.configsChanged)
        # self.connect(self, SIGNAL('configsChanged'), self.mainGraphicsWidget.draw2D.configsChanged)
        # self.connect(self, SIGNAL('configsChanged'), self.mainGraphicsWidget.draw3D.configsChanged)
        self.mainGraphicsWidget.readSettings()
        self.simulation.setGraphicsWidget(self.mainGraphicsWidget)

        mdiSubWindow = self.addSubWindow(self.mainGraphicsWidget)

        self.mainMdiSubWindow = mdiSubWindow
        self.mainGraphicsWidget.show()
        self.mainGraphicsWidget.setConnects(self)

        self.lastActiveRealWindow = mdiSubWindow
        # return OK drawing

        # MDIFIX
        self.setActiveSubWindowCustomSlot(
            self.lastActiveRealWindow)  # rwh: do this to "check" this in the "Window" menu

        self.updateWindowMenu()
        self.updateActiveWindowVisFlags()
        # print self.graphicsWindowVisDict


        suggested_win_pos = self.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdiSubWindow.move(suggested_win_pos)

    def minimizeAllGraphicsWindows(self):
        '''
        Minimizes all graphics windows. Used ony with MDI window layout
        :return:None
        '''
        if not self.MDI_ON: return

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showMinimized()

    def restoreAllGraphicsWindows(self):
        '''
        Restores all graphics windows. Used ony with MDI window layout
        :return:None
        '''
        if not self.MDI_ON: return

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showNormal()

    def closeActiveSubWindowSlot(self):
        '''
        This method is called whenever a user closes a graphics window - it is a slot for closeActiveWindow action
        :return:None
        '''

        # print '\n\n\n BEFORE  closeActiveSubWindowSlot self.subWindowList().size()=', len(self.subWindowList())

        activeWindow = self.activeSubWindow()

        if not activeWindow: return

        activeWindow.close()

        self.updateWindowMenu()

    def processCommandLineOptions(self, cml_args):  #
        # command line parsing needs to be fixed - it takes place in two places now...
        '''
        Called from compucell3d.pyw  - parses the command line (rf. player5/compucell3d.pyw now). initializes SimpleTabView member variables
        :param cml_args: object returned by: opts, args = getopt.getopt
        :return:
        '''
        import CompuCellSetup
        CompuCellSetup.cml_args = cml_args

        self.cml_args = cml_args

        self.__screenshotDescriptionFileName = ""
        self.customScreenshotDirectoryName = ""
        startSimulation = False

        currentDir = ""
        port = -1
        # TODO IMPLEMENT CML PARSING HERE

        self.__prefsFile = "cc3d_default"  # default name of QSettings .ini file (in ~/.config/Biocomplexity on *nix)

        if cml_args.input:
            self.__fileName = cml_args.input
            startSimulation = True

        if cml_args.screenshotDescription:
            self.__screenshotDescriptionFileName = cml_args.screenshotDescription

        self.__imageOutput = not cml_args.noOutput

        if cml_args.screenshotOutputDir:
            self.customScreenshotDirectoryName = cml_args.screenshotOutputDir
            self.__imageOutput = True

        if cml_args.playerSettings:
            self.playerSettingsFileName = cml_args.playerSettings

        currentDir = cml_args.currentDir if cml_args.currentDir else ''
        if cml_args.windowSize:
            winSizes = cml_args.windowSize.split('x')
            # print MODULENAME, "  winSizes=", winSizes
            width = int(winSizes[0])
            height = int(winSizes[1])
            Configuration.setSetting("GraphicsWinWidth", width)
            Configuration.setSetting("GraphicsWinHeight", height)

        port = cml_args.port if cml_args.port else -1
        if cml_args.prefs:
            self.__prefsFile = cml_args.prefs
            # print MODULENAME, '---------  doing QSettings ---------  prefsFile=', self.__prefsFile
            Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Biocomplexity",
                                                 self.__prefsFile)
            Configuration.setSetting("PreferencesFile", self.__prefsFile)

        self.closePlayerAfterSimulationDone = cml_args.exitWhenDone

        if cml_args.guiScan:
            # when user uses gui to do parameter scan all we have to do is to set self.closePlayerAfterSimulationDone to True
            self.closePlayerAfterSimulationDone = True
            # we reset max number of consecutive runs to 1 because we want each simulation in parameter scan
            # initiated by the psrun.py script to be an independent run after which player5 gets closed and reopened again for the next run
            self.maxNumberOfConsecutiveRuns = 1

        if cml_args.maxNumberOfConsecutiveRuns:
            self.maxNumberOfConsecutiveRuns = cml_args.maxNumberOfConsecutiveRuns

        # for o, a in cml_args:
            # print "o=", o
            # print "a=", a
            # if o in ("-i"):  # input file (e.g.  .dml for pre-dumped vtk files)
            #     self.__fileName = a
            #     startSimulation = True
            # elif o in ("-h", "--help"):
            #     self.usage()
            #     sys.exit()
            # elif o in ("-s"):
            #     self.__screenshotDescriptionFileName = a
            # elif o in ("-o"):
            #     self.customScreenshotDirectoryName = a
            #     self.__imageOutput = True
            # elif o in ("-p"):
            #     print ' handling - (playerSettings, e.g. camera)... a = ', a
            #     self.playerSettingsFileName = a
            #     print MODULENAME, 'self.playerSettingsFileName= ', self.playerSettingsFileName
            # elif o in ("--noOutput"):
            #     self.__imageOutput = False
            # elif o in ("--currentDir"):
            #     currentDir = a
            #     print "currentDirectory=", currentDir
            # elif o in ("-w"):  # assume parameter is widthxheight smashed together, e.g. -w 500x300
            #     winSizes = a.split('x')
            #     print MODULENAME, "  winSizes=", winSizes
            #     width = int(winSizes[0])
            #     height = int(winSizes[1])
            #     Configuration.setSetting("GraphicsWinWidth", width)
            #     Configuration.setSetting("GraphicsWinHeight", height)

            # elif o in ("--port"):
            #     port = int(a)
            #     print "port=", port
            # elif o in ("--prefs"):
            #     self.__prefsFile = a
            #     print MODULENAME, '---------  doing QSettings ---------  prefsFile=', self.__prefsFile
            #     Configuration.mySettings = QSettings(QSettings.IniFormat, QSettings.UserScope, "Biocomplexity",
            #                                          self.__prefsFile)
            #     Configuration.setSetting("PreferencesFile", self.__prefsFile)
            #
            #     # elif o in ("--tweditPID"):
            #     # tweditPID=int(a)
            #     # print "tweditPID=",tweditPID

            # elif o in ("--exitWhenDone"):
            #     self.closePlayerAfterSimulationDone = True
            # elif o in (
            #         "--guiScan"):  # when user uses gui to do parameter scan all we have to do is to set self.closePlayerAfterSimulationDone to True
            #     self.closePlayerAfterSimulationDone = True
            #     # we reset max number of consecutive runs to 1 because we want each simulation in parameter scan
            #     # initiated by the psrun.py script to be an independent run after which player5 gets closed and reopened again for the next run
            #     self.maxNumberOfConsecutiveRuns = 1
            #
            #     pass
            # elif o in ("--maxNumberOfRuns"):
            #     self.maxNumberOfConsecutiveRuns = int(a)
            #
            #
            #     # elif o in ("--connectTwedit"):
            #     # connectTwedit=True
            # else:
            #     assert False, "unhandled option"

        # import UI.ErrorConsole
        # self.UI.console.getSyntaxErrorConsole().closeCC3D.connect(qApp.closeAllWindows)

        self.UI.console.getSyntaxErrorConsole().setPlayerMainWidget(self)

        self.UI.console.getSyntaxErrorConsole().closeCC3D.connect(qApp.closeAllWindows)

        # establishConnection starts twedit and hooks it up via sockets to player5
        self.tweditAct.triggered.connect(self.UI.console.getSyntaxErrorConsole().cc3dSender.establishConnection)

        #        print MODULENAME,"    self.UI.console=",self.UI.console
        if port != -1:
            self.UI.console.getSyntaxErrorConsole().cc3dSender.setServerPort(port)

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

            self.__parent.setWindowTitle(basename(self.__fileName) + " - CompuCell3D Player")

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

        if startSimulation:
            self.__runSim()

    def usage(self):
        '''
        Prints player5 command line usage guide
        :return:None
        '''
        print "\n--------------------------------------------------------"
        print "USAGE: ./compucell3d.sh -i <sim file (.cc3d or .xml or .py)> -s <ScreenshotDescriptionFile> -o <custom outputDirectory>"
        print "-w <widthxheight of graphics window>"
        print "--exitWhenDone   close the player5 after simulation is done"
        print "--noOutput   ensure that no screenshots are stored regardless of Player settings"
        print "--prefs    name of preferences file to use/save"
        print "-p    playerSettingsFileName (e.g. 3D camera settings)"
        print "-h or --help   print (this) help message"
        print "\ne.g.  compucell3d.sh -i Demos/cellsort_2D/cellsort_2D/cellsort_2D.cc3d -w 500x500 --prefs myCellSortPrefs"

    def setRecentSimulationFile(self, _fileName):
        self.__fileName = _fileName
        from os.path import basename

        self.__parent.setWindowTitle(basename(self.__fileName) + " - CompuCell3D Player")
        import CompuCellSetup

        CompuCellSetup.simulationFileName = self.__fileName

    def resetControlButtonsAndActions(self):
        '''
        Resets control buttons and actions - called either after simulation is done (__cleanAfterSimulation) or in prepareForNewSimulation
        :return:None
        '''
        self.runAct.setEnabled(True)
        self.stepAct.setEnabled(True)
        self.pauseAct.setEnabled(False)
        self.stopAct.setEnabled(False)
        self.openAct.setEnabled(True)
        self.openLDSAct.setEnabled(True)
        self.pifFromSimulationAct.setEnabled(False)
        self.pifFromVTKAct.setEnabled(False)

    def resetControlVariables(self):
        '''
        Resets control variables - called either after simulation is done (__cleanAfterSimulation) or in prepareForNewSimulation
        :return:None
        '''

        self.steppingThroughSimulation = False
        self.cmlHandlerCreated = False

        CompuCellSetup.simulationFileName = ""

        self.drawingAreaPrepared = False
        self.simulationIsRunning = False

        self.newDrawingUserRequest = False
        self.completedFirstMCS = False

    def prepareForNewSimulation(self, _forceGenericInitialization=False, _inStopFcn=False):
        """
        This function creates new instance of computational thread and sets various flags
        to initial values i.e. to a state before the beginnig of the simulations
        """
        self.resetControlButtonsAndActions()

        self.steppingThroughSimulation = False

        CompuCellSetup.viewManager = self
        CompuCellSetup.simulationFileName = ""

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
            # filling out basic simulation data
            self.basicSimulationData.fieldDim = self.simulation.fieldDim
            self.basicSimulationData.numberOfSteps = self.simulation.numberOfSteps

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

            self.simulation = SimulationThread(self)

            self.simulation.simulationInitializedSignal.connect(self.initializeSimulationViewWidget)
            self.simulation.steppablesStarted.connect(self.runSteppablePostStartPlayerPrep)
            self.simulation.simulationFinished.connect(self.handleSimulationFinished)
            self.simulation.completedStep.connect(self.handleCompletedStep)
            self.simulation.finishRequest.connect(self.handleFinishRequest)

            self.plotManager.initSignalAndSlots()

            import PlayerPython
            self.fieldStorage = PlayerPython.FieldStorage()
            self.fieldExtractor = PlayerPython.FieldExtractor()
            self.fieldExtractor.setFieldStorage(self.fieldStorage)

        self.simulation.setCallingWidget(self)

        self.resetControlVariables()

    def __setupArea(self):
        '''
        Closes all open windows (from previous simulation) and creates new VTK window for the new simulation
        :return:None
        '''
        self.close_all_windows()

        self.addVTKWindowToWorkspace()

    def handleErrorMessage(self, _errorType, _traceback_message):
        '''
        Callback function used to display any type of errors from the simulation script
        :param _errorType: str - error type
        :param _traceback_message: str - contains full Python traceback
        :return:
        '''
        msg = QMessageBox.warning(self, _errorType, \
                                  _traceback_message, \
                                  QMessageBox.Ok,
                                  QMessageBox.Ok)

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

    def handleErrorFormatted(self, _errorMessage):
        '''
        Pastes errorMessage directly into error console
        :param _errorMessage: str with error message
        :return:None
        '''
        CompuCellSetup.error_code = 1

        self.__cleanAfterSimulation()
        syntaxErrorConsole = self.UI.console.getSyntaxErrorConsole()

        syntaxErrorConsole.setText(_errorMessage)
        self.UI.console.bringUpSyntaxErrorConsole()

        if self.cml_args.testOutputDir:
            with open(os.path.join(self.cml_args.testOutputDir, 'error_output.txt'), 'w') as fout:
                fout.write('%s' % _errorMessage)

        print 'DUPA GOT FORMATTED ERROR'
        return

    def processIncommingSimulation(self, _fileName, _stopCurrentSim=False):
        '''
        Callback function used to start new simulation. Currently invoked indirectly from the twedit++ when users choose
        "Open In Player" option form the CC3D project in the project context menu
        :param _fileName: str - simulation file name - full path
        :param _stopCurrentSim: bool , flag indicating if current simulation needs to be stopped
        :return:None
        '''
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

            if startNewSimulation:
                self.__runSim()
        else:
            self.__fileName = _fileName
            self.nextSimulation = _fileName

        from os.path import basename

        self.__parent.setWindowTitle(basename(str(_fileName)) + " - CompuCell3D Player")

    def prepareXMLTreeView(self):
        '''
        Initializes model editor tree view of the CC3DML - Model editor is used for steering
        :return:None
        '''

        self.root_element = CompuCellSetup.cc3dXML2ObjConverter.root
        self.model = SimModel(self.root_element, self.__modelEditor)

        # hook in simulation thread class to XML model TreeView panel in the GUI - needed for steering
        self.simulation.setSimModel(self.model)

        # self.model.checkSanity()

        self.__modelEditor.setModel(self.model)
        #        print MODULENAME,' --------- prepareXMLTreeView(self):'
        #        import pdb; pdb.set_trace()
        self.model.setPrintFlag(True)


        # todo
        pass

        # self.root_element = CompuCellSetup.cc3dXML2ObjConverter.root
        # self.model = SimModel(self.root_element, self.__modelEditor)
        # self.simulation.setSimModel(
        #     self.model)  # hook in simulation thread class to XML model TreeView panel in the GUI - needed for steering
        #
        # # self.model.checkSanity()
        #
        # self.__modelEditor.setModel(self.model)
        # #        print MODULENAME,' --------- prepareXMLTreeView(self):'
        # #        import pdb; pdb.set_trace()
        # self.model.setPrintFlag(True)

    def prepareLatticeDataView(self):
        '''
        Initializes widget that displays vtk file names during vtk file replay mode in the Player
        :return:None
        '''
        ui = self.__parent

        ui.latticeDataModel.setLatticeDataFileList(self.simulation.ldsFileList)
        self.latticeDataModel = ui.latticeDataModel

        # this sets up the model and actually displays model data- so use this function when model is ready to be used

        ui.latticeDataModelTable.setModel(ui.latticeDataModel)

        ui.latticeDataModelTable.setParams()
        self.latticeDataModelTable = ui.latticeDataModelTable

    def __loadSim(self, file):
        '''
        Loads simulation
        :param file: str - full path to the CC3D simulation (usually .cc3d file or .dml vtk replay file path) .
        XML and python files are also acceptable options for the simulation but they are deprecated in favor of .cc3d
        :return:
        '''
        # resetting reference to SimulationDataHandler

        self.prepareForNewSimulation(_forceGenericInitialization=True)

        self.cc3dSimulationDataHandler = None

        fileName = str(self.__fileName)
        # print 'INSIDE LOADSIM file=',fileName
        #        print MODULENAME,"Load file ",fileName
        self.UI.console.bringUpOutputConsole()

        # have to connect error handler to the signal emited from self.simulation object
        # TODO changing signals
        self.simulation.errorOccured.connect(self.handleErrorMessage)
        self.simulation.errorFormatted.connect(self.handleErrorFormatted)

        # self.connect(self.simulation, SIGNAL("errorOccured(QString,QString)"), self.handleErrorMessage)
        # # self.connect(self.simulation,SIGNAL("errorOccuredDetailed(QString,QString,int,int,QString)"),self.handleErrorMessageDetailed)
        # self.connect(self.simulation, SIGNAL("errorFormatted(QString)"), self.handleErrorFormatted)

        # We need to create new SimulationPaths object for each new simulation.
        #        import CompuCellSetup
        CompuCellSetup.simulationPaths = CompuCellSetup.SimulationPaths()

        if re.match(".*\.xml$", fileName):  # If filename ends with .xml
            # print "GOT FILE ",fileName
            # self.prepareForNewSimulation()
            self.simulation.setRunUserPythonScriptFlag(True)
            CompuCellSetup.simulationPaths.setPlayerSimulationXMLFileName(fileName)
            pythonScriptName = CompuCellSetup.ExtractPythonScriptNameFromXML(fileName)

            if pythonScriptName != "":
                CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(pythonScriptName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

        elif re.match(".*\.py$", fileName):
            globals = {'simTabView': 20}
            locals = {}
            self.simulation.setRunUserPythonScriptFlag(True)

            # NOTE: extracting of xml file name from python script is done during script run time so we cannot use CompuCellSetup.simulationPaths.setXmlFileNameFromPython function here
            CompuCellSetup.simulationPaths.setPlayerSimulationPythonScriptName(self.__fileName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

        elif re.match(".*\.cc3d$", fileName):
            self.__loadCC3DFile(fileName)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

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
        Configuration.setSetting("RecentSimulations", os.path.abspath(
            self.__fileName))  # each loaded simulation has to be passed to a function which updates list of recent files

    def __loadCC3DFile(self, fileName):
        '''
        Loads .cc3d file . loads project-specific settings for the project if such exist or creates them based on the
        global settings stored in ~/.compucell3d. It internally invokes the data reader modules which reads the file
        and populate resources and file paths in CC3DSimulationDataHandler class object.
        :param fileName: str - .cc3d file name
        :return:None
        '''

        """
         CC3DSimulationDataHandler class holds the file paths of all the resources and has methods to read the 
        .cc3d file contents
        """
        import CC3DSimulationDataHandler
        self.cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(self)

        # Checking if the file is readable otherwise raising an error
        try:
            f = open(fileName, 'r')
            f.close()
        except IOError, e:
            msg = QMessageBox.warning(self, "Not A Valid Simulation File", \
                                      "Please make sure <b>%s</b> exists" % fileName, \
                                      QMessageBox.Ok)
            raise IOError("%s does not exist" % fileName)

        self.cc3dSimulationDataHandler.readCC3DFileFormat(fileName)

        # check if current CC3D version is greater or equal to the version (minimal required version) specified in the project
        import Version
        currentVersion = Version.getVersionAsString()
        currentVersionInt = currentVersion.replace('.', '')
        projectVersion = self.cc3dSimulationDataHandler.cc3dSimulationData.version
        projectVersionInt = projectVersion.replace('.', '')
        # print 'projectVersion=', projectVersion
        # print 'currentVersion=', currentVersion

        if int(projectVersionInt) > int(currentVersionInt):
            msg = QMessageBox.warning(self, "CompuCell3D Version Mismatch", \
                                      "Your CompuCell3D version <b>%s</b> might be too old for the project you are trying to run. The least version project requires is <b>%s</b>. You may run project at your own risk" % (
                                          currentVersion, projectVersion), \
                                      QMessageBox.Ok)

        # If project settings exists using the project settings
        if self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource:
            self.customSettingPath = self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource.path
            # print 'GOT CUSTOM SETTINGS RESOURCE = ', self.customSettingPath
            Configuration.initializeCustomSettings(self.customSettingPath)
            self.__paramsChanged()
        # Else creating a project settings file
        else:
            self.customSettingPath = os.path.abspath(
                os.path.join(self.cc3dSimulationDataHandler.cc3dSimulationData.basePath, 'Simulation',settings_data.SETTINGS_FILE_NAME))

            # self.customSettingPath = os.path.abspath(
            #     os.path.join(self.cc3dSimulationDataHandler.cc3dSimulationData.basePath, 'Simulation/_settings.xml'))
            # Configuration.writeCustomFile(self.customSettingPath)
            Configuration.writeSettingsForSingleSimulation(self.customSettingPath)

        # Checking for parameter scan resource
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

                    self.cc3dSimulationDataHandler.copySimulationDataFiles(customOutputPath)

                    # construct path to the just-copied .cc3d file
                    cc3dFileBaseName = os.path.basename(self.cc3dSimulationDataHandler.cc3dSimulationData.path)
                    cc3dFileFullName = os.path.join(customOutputPath, cc3dFileBaseName)

                    psu.replaceValuesInSimulationFiles(_pScanFileName=pScanFilePath, _simulationDir=customOutputPath)
                    # save parameter Scan spec file with incremented ityeration
                    psu.saveParameterScanState(_pScanFileName=pScanFilePath)

                    from os.path import basename

                    self.__parent.setWindowTitle('ParameterScan: ' +
                                                 basename(self.__fileName) + ' Iteration: ' + basename(
                        customOutputPath) + " - CompuCell3D Player")

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

            if self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":
                CompuCellSetup.simulationPaths.setPythonScriptNameFromXML(
                    self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)

        if self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript != "":
            CompuCellSetup.simulationPaths.setPlayerSimulationWindowsFileName(
                self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript)
            self.__windowsXMLFileName = self.cc3dSimulationDataHandler.cc3dSimulationData.windowScript

    def __setConnects(self):
        '''
        Sets up signal slot connections for actions
        :return:None
        '''
        # QShortcut(QKeySequence("Ctrl+p"), self, self.__dumpPlayerParams)  # Cmd-3 on Mac
        self.runAct.triggered.connect(self.__runSim)
        self.stepAct.triggered.connect(self.__stepSim)
        self.pauseAct.triggered.connect(self.__pauseSim)
        self.stopAct.triggered.connect(self.__simulationStop)

        self.serializeAct.triggered.connect(self.__simulationSerialize)
        self.restoreDefaultSettingsAct.triggered.connect(self.__restoreDefaultSettings)

        self.openAct.triggered.connect(self.__openSim)
        self.openLDSAct.triggered.connect(self.__openLDSFile)

        self.saveScreenshotDescriptionAct.triggered.connect(self.__saveScrDesc)
        self.openScreenshotDescriptionAct.triggered.connect(self.__openScrDesc)

        # qApp is a member of QtGui. closeAllWindows will cause closeEvent and closeEventSimpleTabView will be called
        self.exitAct.triggered.connect(qApp.closeAllWindows)

        self.cellsAct.triggered.connect(self.__checkCells)
        self.borderAct.triggered.connect(self.__checkBorder)
        self.clusterBorderAct.triggered.connect(self.__checkClusterBorder)
        self.cellGlyphsAct.triggered.connect(self.__checkCellGlyphs)
        self.FPPLinksAct.triggered.connect(self.__checkFPPLinks)

        self.limitsAct.triggered.connect(self.__checkLimits)
        self.configAct.triggered.connect(self.__showConfigDialog)
        self.cc3dOutputOnAct.triggered.connect(self.__checkCC3DOutput)
        self.resetCameraAct.triggered.connect(self.__resetCamera)
        self.zoomInAct.triggered.connect(self.zoomIn)
        self.zoomOutAct.triggered.connect(self.zoomOut)

        self.pifFromSimulationAct.triggered.connect(self.__generatePIFFromCurrentSnapshot)
        self.pifFromVTKAct.triggered.connect(self.__generatePIFFromVTK)

        # window menu actions
        self.newGraphicsWindowAct.triggered.connect(self.addNewGraphicsWindow)

        self.tileAct.triggered.connect(self.tileSubWindows)
        self.cascadeAct.triggered.connect(self.cascadeSubWindows)

        self.minimizeAllGraphicsWindowsAct.triggered.connect(self.minimizeAllGraphicsWindows)
        self.restoreAllGraphicsWindowsAct.triggered.connect(self.restoreAllGraphicsWindows)

        self.closeActiveWindowAct.triggered.connect(self.closeActiveSubWindowSlot)
        # self.closeAdditionalGraphicsWindowsAct, triggered self.removeAuxiliaryGraphicsWindows)

        self.configsChanged.connect(self.__paramsChanged)

    def setFieldType(self, _fieldTypeTuple):
        '''
        Called from GraphicsFrameWidget
        :param _fieldTypeTuple: tuple with field types
        :return:None
        '''
        self.__fieldType = _fieldTypeTuple

    def closeEventSimpleTabView(self, event=None):
        '''
        Handles player5 CloseEvent - called from closeEvent in UserInterface.py
        :param event: Qt CloseEvent
        :return:None
        '''

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

    def initializeSimulationViewWidgetCMLResultReplay(self):
        '''
        Initializes PLayer during VTK replay run mode
        :return:None
        '''
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
        self.cmlReplayManager.set_stay_in_current_step(True)

    def createOutputDirs(self):
        '''
        Creates Simulation output directory
        :return:None
        '''

        import CompuCellSetup
        #        import pdb; pdb.set_trace()
        if self.customScreenshotDirectoryName == "":
            (self.screenshotDirectoryName, self.baseScreenshotName) = \
                CompuCellSetup.makeSimDir(self.__fileName, self.__outputDirectory)

            CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName
            self.prevOutputDir = self.__outputDirectory

        else:
            # for parameter scan the directories are created in __loadCC3DFile
            if self.singleSimulation:

                (self.screenshotDirectoryName, self.baseScreenshotName) = \
                    self.makeCustomSimDir(self.customScreenshotDirectoryName, self.__fileName)

                CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName

            else:
                self.screenshotDirectoryName = self.parameterScanOutputDir

                pScanBaseFileName = os.path.basename(self.__fileName)
                pScanBaseFileName, extension = os.path.splitext(pScanBaseFileName)
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

        if (self.mysim == None):
            print MODULENAME, '\n\n\n createOutputDirs():  self.mysim is None!!!'  # bad, very bad

        CompuCellSetup.initCMLFieldHandler(self.mysim(), self.resultStorageDirectory,
                                           self.fieldStorage)  # also creates the /LatticeData dir

    def initializeSimulationViewWidgetRegular(self):
        '''
        Initializes Player during simualtion run mode
        :return:None
        '''

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

        if not self.cerrStreamBufOrig:  # get original cerr stream buffer - do it only once per session
            self.cerrStreamBufOrig = simObj.getCerrStreamBufOrig()

        # if Configuration.getVisualization("CC3DOutputOn"):
        if self.UI.viewmanager.cc3dOutputOnAct.isChecked():
            if Configuration.getSetting("UseInternalConsole"):
                # redirecting output from C++ to internal console
                import sip

                stdErrConsole = self.UI.console.getStdErrConsole()  # we use __stdout console (see UI/Consile.py) as main output console for both stdout and std err from C++ and Python - sort of internal system console
                stdErrConsole.clear()
                addr = sip.unwrapinstance(stdErrConsole)

                simObj.setOutputRedirectionTarget(addr)
                # redirecting Python output to internal console
                self.UI.useInternalConsoleForPythonOutput(True)
            else:
                # C++ output goes to system console
                # simObj.setOutputRedirectionTarget(-1)
                simObj.restoreCerrStreamBufOrig(self.cerrStreamBufOrig)
                # Python output goes to system console
                self.UI.enablePythonOutput(True)
        else:
            # silencing output from C++
            simObj.setOutputRedirectionTarget(0)
            # silencing output from Python
            self.UI.enablePythonOutput(False)

        self.basicSimulationData.fieldDim = self.fieldDim
        self.basicSimulationData.sim = simObj
        self.basicSimulationData.numberOfSteps = simObj.getNumSteps()

        self.fieldStorage.allocateCellField(self.fieldDim)

        self.fieldExtractor.init(simObj)

        self.screenshotNumberOfDigits = len(str(self.basicSimulationData.numberOfSteps))

        latticeTypeStr = CompuCellSetup.ExtractLatticeType()

        if latticeTypeStr in Configuration.LATTICE_TYPES.keys():
            self.latticeType = Configuration.LATTICE_TYPES[latticeTypeStr]
        else:
            self.latticeType = Configuration.LATTICE_TYPES["Square"]  # default choice

        self.prepareSimulationView()

        self.screenshotManager = ScreenshotManager.ScreenshotManager(self)

        if self.__screenshotDescriptionFileName != "":
            self.screenshotManager.readScreenshotDescriptionFile(self.__screenshotDescriptionFileName)

        if self.simulationIsStepping:
            # print "BEFORE STEPPING PAUSE REGULAR SIMULATION"
            self.__pauseSim()

        self.prepareXMLTreeView()

    def initializeSimulationViewWidget(self):
        '''
        Dispatch function - calls player5 initialization functions (initializeSimulationViewWidgetRegular or initializeSimulationViewWidgetCML) depending on the run mode
        :return:None
        '''
        # todo
        pass
        import CompuCellSetup

        CompuCellSetup.simulationFileName = self.__fileName
        self.close_all_windows()

        initializeSimulationViewWidgetFcn = getattr(self, "initializeSimulationViewWidget" + self.__viewManagerType)
        initializeSimulationViewWidgetFcn()

        #        print MODULENAME, 'initializeSimulationViewWidget(),  __imageOutput,__latticeOutputFlag,screenshotDirectoryName=', self.__imageOutput,self.__latticeOutputFlag,self.screenshotDirectoryName
        if (self.__imageOutput or self.__latticeOutputFlag) and self.screenshotDirectoryName == "":
            #            print MODULENAME, 'initializeSimulationViewWidget(),  calling createOutputDirs'
            self.createOutputDirs()

        # copy simulation files to output directory  for simgle simulation- copying of the simulations files for parameter scan is doen in the __loadCC3DFile
        if self.singleSimulation:
            if self.cc3dSimulationDataHandler and CompuCellSetup.screenshotDirectoryName != "":
                self.cc3dSimulationDataHandler.copySimulationDataFiles(CompuCellSetup.screenshotDirectoryName)

        # print MODULENAME, " initializeSimulationViewWidget():  before TRY ACQUIRE"
        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()
        # print MODULENAME, " initializeSimulationViewWidget():  AFTER RELEASE"

    #        import pdb; pdb.set_trace()

    def runSteppablePostStartPlayerPrep(self):
        '''
        Handler function runs after steppables executed start functions. Restores window layout for plot windows
        :return:None
        '''
        self.setFieldTypes()

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

        # restoring plots

        self.plotManager.restore_plots_layout()

    def extractAddressIntFromVtkObject(self, _vtkObj):
        '''
        Extracts memory address of vtk object
        :param _vtkObj: vtk object - e.g. vtk array
        :return: int (possible long int) representing the address of the vtk object
        '''
        return self.fieldExtractor.unmangleSWIGVktPtrAsLong(_vtkObj.__this__)

    def handleSimulationFinishedCMLResultReplay(self, _flag):
        '''
        callback - runs after CML replay mode finished. Cleans after vtk replay
        :param _flag: bool - not used at tyhe moment
        :return:None
        '''
        if CompuCellSetup.playerType == "CMLResultReplay":
            self.latticeDataModelTable.prepareToClose()

        # # # self.__stopSim()
        self.__cleanAfterSimulation()

    def launchNextParameterScanRun(self):
        '''
        launches next aprameter scan. Deprecated - parameter scans shuld be run from command line
        :return:None
        '''
        fileName = self.__fileName
        # when running parameter scan after simulatino finish we run again the same simulation file. When cc3d project with parameter scan gets opened 'next iteration' simulation is generatet and this
        # newly generated cc3d file is substituted instead of the "master" cc3d with parameter scan
        # From user stand point whan matters is that the only thing that user needs to worry abuot is the "master" .cc3d project and this is what is opened in the player5
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
        '''
        Callback - called after "regular" simulation finishes
        :param _flag:bool - unused
        :return:None
        '''
        print 'INSIDE handleSimulationFinishedRegular'
        self.__cleanAfterSimulation()

        if not self.singleSimulation:
            self.launchNextParameterScanRun()

    def handleSimulationFinished(self, _flag):
        '''
        dispatch function for simulation finished event
        :param _flag: bool - unused
        :return:
        '''
        handleSimulationFinishedFcn = getattr(self, "handleSimulationFinished" + self.__viewManagerType)
        handleSimulationFinishedFcn(_flag)

    def handleCompletedStepCMLResultReplay(self, _mcs):
        '''
        callback - runs after vtk replay step completed.
        :param _mcs: int - current Monte Carlo step
        :return:None
        '''
        self.simulation.drawMutex.lock()  # had to add synchronization here . without it I would get weird behavior in CML replay mode

        simulationDataIntAddr = self.extractAddressIntFromVtkObject(self.simulation.simulationData)

        self.fieldExtractor.setSimulationData(simulationDataIntAddr)
        self.__step = self.simulation.currentStep

        self.latticeDataModelTable.selectRow(
            self.simulation.stepCounter - 1)  # self.simulation.stepCounter is incremented by one before it reaches this function

        # there is additional locking inside draw to acccount for the fact that users may want to draw lattice on demand
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

            self.mainGraphicsWidget.takeSimShot(screenshotFileName)
            self.screenshotManager.outputScreenshots(self.screenshotDirectoryName, self.__step)

        self.simulation.drawMutex.unlock()

        # print '\n\n self.simulationIsStepping=',self.simulationIsStepping
        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

        self.cmlReplayManager.keepGoing()

    def handleCompletedStepRegular(self, _mcs):
        '''
        callback - runs after simulation step completed.
        :param _mcs: int - current Monte Carlo step
        :return:
        '''

        self.__drawField()

        self.simulation.drawMutex.lock()
        # will need to sync screenshots with simulation thread. Be sure before simulation thread writes new results all the screenshots are taken



        if self.__imageOutput and not (self.__step % self.__shotFrequency):  # dumping images? Check modulo MCS #
            mcsFormattedNumber = string.zfill(str(self.__step),
                                              self.screenshotNumberOfDigits)  # fills string wtih 0's up to self.screenshotNumberOfDigits width
            screenshotFileName = os.path.join(self.screenshotDirectoryName,
                                              self.baseScreenshotName + "_" + mcsFormattedNumber + ".png")
            if _mcs != 0:
                if self.mainGraphicsWidget:  # self.mainGraphicsWindow can be closed by the user
                    self.mainGraphicsWidget.takeSimShot(screenshotFileName)

            if Configuration.getSetting('DebugOutputPlayer'):
                print 'self.screenshotManager=', self.screenshotManager

            if self.screenshotManager:
                self.screenshotManager.outputScreenshots(self.screenshotDirectoryName, self.__step)
                if Configuration.getSetting('DebugOutputPlayer'):
                    print 'self.screenshotDirectoryName=', self.screenshotDirectoryName
                    # sys.exit()

                    #        if (CompuCellSetup.cmlFieldHandler is not None) and self.__latticeOutputFlag and (not self.__step % self.__latticeOutputFrequency):  #rwh
        if self.cmlHandlerCreated and self.__latticeOutputFlag and (
                not self.__step % self.__latticeOutputFrequency):  # rwh
            CompuCellSetup.cmlFieldHandler.writeFields(self.__step)

        self.simulation.drawMutex.unlock()

        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

    def handleCompletedStep(self, _mcs):
        '''
        Dispatch function for handleCompletedStep functions
        :param _mcs: int - current Monte Carlo step
        :return:None
        '''
        if not self.completedFirstMCS:
            self.completedFirstMCS = True

        self.__step = _mcs

        handleCompletedStepFcn = getattr(self, "handleCompletedStep" + self.__viewManagerType)

        handleCompletedStepFcn(_mcs)

    def handleFinishRequest(self, _flag):
        '''
        Ensures that all the tasks in the GUI thread that need simulator to be alive are completed before proceeding
        further with finalizing the simulation. For example SimpleTabView.py. function handleCompletedStepRegular
        may need a lot of time to output simulations fields and those fields need to have alive simulator otherwise
        accessing to destroyed field will lead to segmentation fault
        Saves Window layout into project settings
        :param _flag: bool - unused
        :return:None
        '''

        # we do not save windows layout for simulation replay
        if self.__viewManagerType != "CMLResultReplay":
            self.__saveWindowsLayout()

        self.simulation.drawMutex.lock()
        self.simulation.drawMutex.unlock()

        # this releases finish mutex which is a signal to simulation thread that is is OK to finish
        self.simulation.finishMutex.unlock()

    def init_simulation_control_vars(self):
        '''
        Sets several output-related variables in simulation thread
        :return:None
        '''
        self.simulation.screenUpdateFrequency = self.__updateScreen
        self.simulation.imageOutputFlag = self.__imageOutput
        self.simulation.screenshotFrequency = self.__shotFrequency
        self.simulation.latticeOutputFlag = self.__latticeOutputFlag
        self.simulation.latticeOutputFrequency = self.__latticeOutputFrequency

    def prepareSimulation(self):
        '''
        Prepares simulation - loads simulation, installs rollback importer - to unimport previously used modules
        :return:None
        '''
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

                xmlFileName = CompuCellSetup.simulationPaths.simulationXMLFileName
                print "Error in XML File", "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message
                self.handleErrorMessage("Error in XML File",
                                        "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message)
            except IOError, e:
                return

            self.init_simulation_control_vars()

            self.screenshotDirectoryName = ""

            if self.rollbackImporter:
                self.rollbackImporter.uninstall()

            self.rollbackImporter = RollbackImporter()

    def __runSim(self):
        '''
        Slot that actuallt runs the simulation
        :return:None
        '''

        self.simulation.screenUpdateFrequency = self.__updateScreen  # when we run simulation we ensure that self.simulation.screenUpdateFrequency is whatever is written in the settings

        if not self.drawingAreaPrepared:
            self.prepareSimulation()

        # print 'SIMULATION PREPARED self.__viewManagerType=',self.__viewManagerType
        if self.__viewManagerType == "CMLResultReplay":
            # print 'starting CMLREPLAY'
            import CompuCellSetup

            self.simulation.semPause.release()  # just in case

            # these flagg settings calls have to be executed before self.cmlReplayManager.keepGoing()
            self.simulationIsRunning = True
            self.simulationIsStepping = False

            self.cmlReplayManager.setRunState()
            self.cmlReplayManager.keepGoing()

            self.runAct.setEnabled(False)
            self.stepAct.setEnabled(True)
            self.stopAct.setEnabled(True)
            self.pauseAct.setEnabled(True)

            self.openAct.setEnabled(False)
            self.openLDSAct.setEnabled(False)

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
                self.pifFromSimulationAct.setEnabled(True)

                self.openAct.setEnabled(False)
                self.openLDSAct.setEnabled(False)

            if Configuration.getSetting("LatticeOutputOn") and not self.cmlHandlerCreated:
                import CompuCellSetup

                CompuCellSetup.createCMLFieldHandler()
                self.cmlHandlerCreated = True
                #            CompuCellSetup.initCMLFieldHandler(self.mysim,self.resultStorageDirectory,self.fieldStorage)

            self.steppingThroughSimulation = False

            if self.simulationIsStepping:
                self.simulationIsStepping = False
                self.init_simulation_control_vars()

            if not self.pauseAct.isEnabled() and self.simulationIsRunning:
                self.runAct.setEnabled(False)
                self.pauseAct.setEnabled(True)
                self.simulation.semPause.release()
                return

    def __stepSim(self):
        '''
        Slot that steps through simulation
        :return:None
        '''

        self.simulation.screenUpdateFrequency = 1  # when we step we need to ensure screenUpdateFrequency is 1

        if not self.drawingAreaPrepared:
            self.prepareSimulation()

        # print 'SIMULATION PREPARED self.__viewManagerType=',self.__viewManagerType
        if self.__viewManagerType == "CMLResultReplay":
            # print 'starting CMLREPLAY'
            import CompuCellSetup

            self.simulation.semPause.release()
            self.simulationIsRunning = True
            self.simulationIsStepping = True
            self.cmlReplayManager.setStepState()
            self.cmlReplayManager.step()

            self.stopAct.setEnabled(True)
            self.pauseAct.setEnabled(False)
            self.runAct.setEnabled(True)
            self.pifFromVTKAct.setEnabled(True)

            self.openAct.setEnabled(False)
            self.openLDSAct.setEnabled(False)
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
                    "LatticeOutputOn") and not self.cmlHandlerCreated:  # rwh
                CompuCellSetup.createCMLFieldHandler()
                self.cmlHandlerCreated = True  # rwh

                CompuCellSetup.initCMLFieldHandler(self.mysim, self.resultStorageDirectory, self.fieldStorage)
                CompuCellSetup.cmlFieldHandler.getInfoAboutFields()  # rwh

            if self.simulationIsRunning and self.simulationIsStepping:
                #            print MODULENAME,'  __stepSim() - 1:'
                self.pauseAct.setEnabled(False)
                self.simulation.semPause.release()
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)

                return

            # if Pause button is enabled
            elif self.simulationIsRunning and not self.simulationIsStepping and self.pauseAct.isEnabled():  # transition from running simulation
                #            print MODULENAME,'  __stepSim() - 2:'
                #            updateSimPrefs()   # should we call this and then reset screenUpdateFreq = 1 ?
                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.simulationIsStepping = True
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)
            # if Pause button is disabled, meaning the sim is paused:
            elif self.simulationIsRunning and not self.simulationIsStepping and not self.pauseAct.isEnabled():  # transition from paused simulation
                #            print MODULENAME,'  __stepSim() - 3:'
                #            updateSimPrefs()   # should we call this and then reset screenUpdateFreq = 1 ?
                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.simulationIsStepping = True

                return

            return

    def requestRedraw(self):
        '''
        Responds to request to redraw simulatin snapshots if the simulation is running
        :return:
        '''
        if self.simulationIsRunning or self.simulationIsStepping:
            self.__drawField()

    def drawFieldCMLResultReplay(self):
        '''
        Draws fields during vtk replay mode
        :return:None
        '''
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
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsFrame = win.widget()

                if graphicsFrame.is_screenshot_widget:
                    continue

                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()

                if not self.simulation.newFileBeingLoaded:  # this flag is used to prevent calling  draw function when new data is read from hard drive
                    graphicsFrame.drawFieldLocal(self.basicSimulationData)

                self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())

        self.simulation.drawMutex.unlock()
        self.simulation.readFileSem.release()

    def drawFieldRegular(self):
        '''
        Draws field during "regular" simulation
        :return:None
        '''
        if not self.simulationIsRunning:
            return

        if self.newDrawingUserRequest:
            self.newDrawingUserRequest = False
            if self.pauseAct.isEnabled():
                self.__pauseSim()
        self.simulation.drawMutex.lock()

        self.__step = self.simulation.getCurrentStep()

        if self.mysim:

            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsFrame = win.widget()

                if graphicsFrame.is_screenshot_widget:
                    continue

                # rwh: error if we try to invoke switchdim earlier
                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()

                graphicsFrame.drawFieldLocal(self.basicSimulationData)

                self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())  # show MCS in lower-left GUI

        self.simulation.drawMutex.unlock()

    def updateSimulationProperties(self):
        '''
        INitializes basic simulation data - fieldDim, number of steps etc.
        :return:bool - flag indicating if initialization of basic simulation data was successful
        '''
        fieldDim = None
        if self.__viewManagerType == "Regular":

            if not self.mysim:
                return

            simObj = self.mysim()
            if not simObj: return False

            fieldDim = simObj.getPotts().getCellFieldG().getDim()

            if fieldDim.x == self.fieldDim.x and fieldDim.y == self.fieldDim.y and fieldDim.z == self.fieldDim.z:
                return False

            self.fieldDim = fieldDim
            self.basicSimulationData.fieldDim = self.fieldDim
            self.basicSimulationData.sim = simObj
            self.basicSimulationData.numberOfSteps = simObj.getNumSteps()

            return True

        elif self.__viewManagerType == "CMLResultReplay":
            fieldDim = self.simulation.fieldDim
            if self.simulation.dimensionChange():
                self.simulation.resetDimensionChangeMonitoring()
                self.fieldDim = self.simulation.fieldDim
                self.basicSimulationData.fieldDim = self.fieldDim
                self.fieldExtractor.setFieldDim(self.basicSimulationData.fieldDim)
                return True

            return False

    def updateVisualization(self):
        '''
        Updates visualization properties - called e.g. after resizing of the lattice
        :return:None
        '''

        self.fieldStorage.allocateCellField(self.fieldDim)
        # this updates cross sections when dimensions change

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            win.widget().updateCrossSection(self.basicSimulationData)

        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsWidget = win.widget()
            graphicsWidget.resetAllCameras()

        # self.__drawField()

        if self.simulationIsRunning and not self.simulationIsStepping:
            self.__runSim()  # we are immediately restarting it after e.g. lattice resizing took place

    def _drawField(self):  # called from GraphicsFrameWidget.py
        '''
        Calls __drawField
        :return:None
        '''
        #        print MODULENAME,'   _drawField called'
        self.__drawField()

    def __drawField(self):
        '''
        Dispatch function to draw simulation snapshots
        :return:None
        '''

        self.displayWarning(
            '')  # here we are resetting previous warnings because draw functions may write their own warning

        __drawFieldFcn = getattr(self, "drawField" + self.__viewManagerType)

        propertiesUpdated = self.updateSimulationProperties()

        if propertiesUpdated:
            # __drawFieldFcn() # this call is actually unnecessary
            self.updateVisualization()  # for some reason cameras have to be initialized after drawing resized lattice and draw function has to be repeated

        __drawFieldFcn()

    def displayWarning(self, warning_text):
        '''
        Displays Warnings in the status bar
        :param warning_text: str - warning text
        :return:None
        '''
        self.warnings.setText(warning_text)

    def __updateStatusBar(self, step, conMinMax):
        '''
        Updates status bar
        :param step: int - current MCS
        :param conMinMax: two element list with min max valuaes for the concentration field
        :return:
        '''
        self.mcSteps.setText("MC Step: %s" % step)
        self.conSteps.setText("Min: %s Max: %s" % conMinMax)

    def __pauseSim(self):
        '''
        slot that pauses simulation
        :return:None
        '''
        if self.__viewManagerType == "CMLResultReplay":
            self.cmlReplayManager.setPauseState()

        self.simulation.semPause.acquire()
        self.runAct.setEnabled(True)
        self.pauseAct.setEnabled(False)

    def __saveWindowsLayout(self):
        '''
        Saves windows layout in the _settings.xml
        :return:None
        '''

        windowsLayout = {}

        for key, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            print 'key, win = ', (key, win)
            widget = win.widget()
            # if not widget.allowSaveLayout: continue
            if widget.is_screenshot_widget:
                continue

            gwd = widget.getGraphicsWindowData()
            # fill size and position of graphics windows data using mdiWidget, NOT the internal widget such as GraphicsFrameWidget - sizes and positions are base on MID widet settings
            gwd.winPosition = win.pos()
            gwd.winSize = win.size()

            # print 'getGraphicsWindowData=', gwd
            # print 'toDict=', gwd.toDict()

            windowsLayout[key] = gwd.toDict()

        # print 'AFTER self.fieldTypes = ', self.fieldTypes
        try:
            print self.plotManager.plotWindowList
        except AttributeError:
            print "plot manager does not have plotWindowList member"

        plotLayoutDict = self.plotManager.getPlotWindowsLayoutDict()
        # for key, gwd in plotLayoutDict.iteritems():
        #     print 'key=', key
        #     print 'gwd=', gwd

        # combining two layout dicts
        windowsLayoutCombined = windowsLayout.copy()
        windowsLayoutCombined.update(plotLayoutDict)
        # print 'windowsLayoutCombined=',windowsLayoutCombined
        Configuration.setSetting('WindowsLayout', windowsLayoutCombined)

    def __simulationStop(self):
        '''
        Slot that handles simulation stop
        :return:None
        '''
        # Once user requests explicit stop of the simulation we stop regardless whether this is parameter scan or not.
        # To stop parameter scan we reset variables used to seer parameter scan to their default (non-param scan) values

        self.runAgainFlag = False

        # we do not save windows layout for simulation replay
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
        '''
        Slot that handles request to serialize simulation
        :return:None
        '''
        # print self.simulation.restartManager
        currentStep = self.simulation.sim.getStep()
        if self.pauseAct.isEnabled():
            self.__pauseSim()
        self.simulation.restartManager.outputRestartFiles(currentStep, True)

    def __restoreDefaultSettings(self):
        '''
        Replaces existing simulation's settings with the default ones
        :return: None
        '''
        if not self.simulationIsRunning:  # works only for running simulation
            return

        # print 'Replacing settings'
        Configuration.replaceCustomSettingsWithDefaults()

    def quit(self, error_code=0):
        """Quit the application."""
        # self.closeEvent(None)
        # QtCore.QCoreApplication.instance().quit()
        print 'error_code = ', error_code
        QCoreApplication.instance().exit(error_code)
        print 'AFTER QtCore.QCoreApplication.instance()'


    def __cleanAfterSimulation(self, _exitCode=0):
        '''
        Cleans after simulation is done
        :param _exitCode: exit code from the simulation
        :return:None
        '''

        self.resetControlButtonsAndActions()
        self.resetControlVariables()

        self.fieldTypes = {}  # re-init (empty) the fieldTypes dict, otherwise get previous/bogus fields in graphics win field combobox

        # saving settings witht eh simulation
        if self.customSettingPath:
            Configuration.writeSettingsForSingleSimulation(self.customSettingPath)
            self.customSettingPath = ''

        Configuration.writeAllSettings()
        Configuration.initConfiguration()  # this flushes configuration

        if Configuration.getSetting("ClosePlayerAfterSimulationDone") or self.closePlayerAfterSimulationDone:
            Configuration.setSetting("RecentFile", os.path.abspath(self.__fileName))

            Configuration.setSetting("RecentSimulations", os.path.abspath(self.__fileName))

            # sys.exit(_exitCode)
            self.quit(CompuCellSetup.error_code)

        # in case there is pending simulation to be run we will put it a recent simulation so that it can be ready to run without going through open file dialog
        if self.nextSimulation != "":
            Configuration.setSetting("RecentSimulations", self.nextSimulation)
            self.nextSimulation = ""

        self.simulation.sim = None
        self.basicSimulationData.sim = None
        self.mysim = None

        if self.screenshotManager:
            self.screenshotManager.cleanup()

        self.screenshotManager = None

        CompuCellSetup.resetGlobals()
        # print 'AFTER __cleanupAfterSimulation'

        # self.close_all_windows()

    def close_all_windows(self):
        '''
        Closes all windows
        :return:None
        '''
        for win in self.win_inventory.values():
            self.win_inventory.remove_from_inventory(win)
            try:
                if not sys.platform.startswith('win'):
                    win.showNormal()
            except:
                pass
            win.close()

        self.win_inventory.set_counter(0)

    def __stopSim(self):
        '''
        stops simulation thread
        :return:None
        '''
        self.simulation.stop()
        self.simulation.wait()

    def makeCustomSimDir(self, _dirName, _simulationFileName):
        '''
        Creates custom simulation output directory
        :param _dirName: str - custom directory name
        :param _simulationFileName: current simulation file name
        :return: tupple (custom directory name, base file name for directory)
        '''
        fullFileName = os.path.abspath(_simulationFileName)
        (filePath, baseFileName) = os.path.split(fullFileName)
        baseFileNameForDirectory = baseFileName.replace('.', '_')
        if not os.path.isdir(_dirName):
            os.mkdir(_dirName)
            return (_dirName, baseFileNameForDirectory)
        else:
            return ("", "")

    # Shows the plugin view tab
    def showPluginView(self, pluginInfo):
        '''
        Shows PLugin information - deprecated
        :param pluginInfo:plugin information
        :return:None
        '''
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

    def setInitialCrossSection(self, _basicSimulationData):
        '''
        Initializes cross section bar for vtk graphics window
        :param _basicSimulationData: BasicSimulationData
        :return:None
        '''
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsFrame = win.widget()
            graphicsFrame.setInitialCrossSection(_basicSimulationData)

    def initGraphicsWidgetsFieldTypes(self):
        '''
        Initializes graphics field types for vtk graphics window
        :return:None
        '''
        for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphicsFrame = win.widget()
            graphicsFrame.setFieldTypesComboBox(self.fieldTypes)

    # Shows simulation view tab
    def showSimView(self, file):
        '''
        Shows Initial simulation view. calls function to restore windows layout
        :param file: str - file path - unused
        :return:None
        '''

        self.__setupArea()

        # Create self.mainGraphicsWindow
        self.__step = 0

        self.showDisplayWidgets()

        simObj = None
        if self.mysim:
            simObj = self.mysim()
            # if not simObj:return

        self.__fieldType = ("Cell_Field", FIELD_TYPES[0])

        if self.basicSimulationData.sim:
            cellField = simObj.getPotts().getCellFieldG()

            self.__drawField()

            # # Fields are available only after simulation is loaded
            self.setFieldTypes()
        else:

            self.__drawField()

            self.setFieldTypesCML()

        self.setInitialCrossSection(self.basicSimulationData)

        self.initGraphicsWidgetsFieldTypes()

        self.drawingAreaPrepared = True

        self.__restoreWindowsLayout()

    def __restoreWindowsLayout(self):
        '''
        Restores Windows layout
        :return:None
        '''

        windowsLayoutDict = Configuration.getSetting('WindowsLayout')
        print 'from settings windowsLayout = ', windowsLayoutDict

        # first restore main window with id 0 - this window is the only window open at this point and it is open by default when simulation is started
        # that's why we have to treat it in a special way but only when we determine that windowsLayoutDict is not empty
        if len(windowsLayoutDict.keys()):
            try:
                # windowDataDict0 = windowsLayoutDict[
                #     str(0)]  # inside windowsLayoutDict windows are labeled using ints represented as strings
                try:
                    # inside windowsLayoutDict windows are labeled using ints represented as strings
                    windowDataDict0 = windowsLayoutDict[str(0)]
                except KeyError:
                    try:
                        windowDataDict0 = windowsLayoutDict[0]
                    except KeyError:
                        raise KeyError('Could not find 0 in the keys of windowsLayoutDict')


                from Graphics.GraphicsWindowData import GraphicsWindowData

                gwd = GraphicsWindowData()

                gwd.fromDict(windowDataDict0)

                if gwd.winType == GRAPHICS_WINDOW_LABEL:
                    graphicsWindow = self.lastActiveRealWindow
                    gfw = graphicsWindow.widget()

                    graphicsWindow.resize(gwd.winSize)
                    graphicsWindow.move(gwd.winPosition)

                    gfw.applyGraphicsWindowData(gwd)

            except KeyError:
                # in case there is no main window with id 0 in the settings we kill the main window

                graphicsWindow = self.lastActiveRealWindow
                graphicsWindow.close()
                self.mainGraphicsWidget = None
                self.win_inventory.remove_from_inventory(graphicsWindow)

                pass

        # we make a sorted list of graphics windows. Graphics Window with lowest id assumes role of
        # mainGraphicsWindow (actually this should be called maingraphicsWidget)
        win_id_list = []
        for windowId, windowDataDict in windowsLayoutDict.iteritems():
            if windowId==0 or windowId=='0':
                continue

            from Graphics.GraphicsWindowData import GraphicsWindowData

            gwd = GraphicsWindowData()

            gwd.fromDict(windowDataDict)

            if gwd.winType != GRAPHICS_WINDOW_LABEL:
                continue
            try:
                win_id_list.append(int(windowId))
            except:
                pass

            win_id_list = sorted(win_id_list)

        # restore graphics windows first
        # for windowId, windowDataDict in windowsLayoutDict.iteritems():
        for win_id in win_id_list:
            windowId = None
            try:
                windowDataDict = windowsLayoutDict[win_id]
            except:

                windowId = str(win_id)
                windowDataDict = windowsLayoutDict[win_id]


            # if windowId == str(0) or win_id==0:
            #     continue

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

            graphicsWindow.resize(gwd.winSize)
            graphicsWindow.move(gwd.winPosition)

            gfw.applyGraphicsWindowData(gwd)


            # print ' PLOT WINDOW MANAGER  WINDOW LIST = ', self.plotManager.plotWindowList

    def setFieldTypesCML(self):
        '''
        initializes field types for VTK vidgets during vtk replay mode
        :return:None
        '''
        # Add cell field
        self.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  # "CellField"

        self.fieldComboBox.clear()
        self.fieldComboBox.addItem("-- Field Type --")
        self.fieldComboBox.addItem("Cell_Field")

        for fieldName in self.simulation.fieldsUsed.keys():
            if fieldName != "Cell_Field":
                self.fieldTypes[fieldName] = self.simulation.fieldsUsed[fieldName]
                self.fieldComboBox.addItem(fieldName)

    def setFieldTypes(self):
        '''
        initializes field types for VTK vidgets during regular simulation
        :return:None
        '''
        # Add cell field
        #        self.fieldTypes = {}
        simObj = self.mysim()
        if not simObj: return

        self.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  # "CellField"

        # Add concentration fields How? I don't care how I got it at this time

        concFieldNameVec = simObj.getConcentrationFieldNameVector()

        # putting concentration fields from simulator
        for fieldName in concFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this conc field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[1]

        # inserting extra scalar fields managed from Python script
        scalarFieldNameVec = self.fieldStorage.getScalarFieldNameVector()
        for fieldName in scalarFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this scalar field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[2]

        # inserting extra scalar fields cell levee managed from Python script
        scalarFieldCellLevelNameVec = self.fieldStorage.getScalarFieldCellLevelNameVector()
        for fieldName in scalarFieldCellLevelNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this scalar field (cell leve): ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[3]

        # inserting extra vector fields  managed from Python script
        vectorFieldNameVec = self.fieldStorage.getVectorFieldNameVector()
        for fieldName in vectorFieldNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this vector field: ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[4]

        # inserting extra vector fields  cell level managed from Python script
        vectorFieldCellLevelNameVec = self.fieldStorage.getVectorFieldCellLevelNameVector()
        for fieldName in vectorFieldCellLevelNameVec:
            #            print MODULENAME,"setFieldTypes():  Got this vector field (cell level): ",fieldName
            self.fieldTypes[fieldName] = FIELD_TYPES[5]

        # inserting custom visualization
        visDict = CompuCellSetup.customVisStorage.visDataDict

        for visName in visDict:
            self.fieldTypes[visName] = FIELD_TYPES[6]

    def showDisplayWidgets(self):
        '''
        Displays initial snapthos widgets - called from showSimView
        :return:None
        '''

        # This block of code simply checks to see if some plugins assoc'd with Vis are defined
        import XMLUtils
        if CompuCellSetup.cc3dXML2ObjConverter != None:
            self.pluginCOMDefined = False
            self.pluginFPPDefined = False

            self.root_element = CompuCellSetup.cc3dXML2ObjConverter.root
            elms = self.root_element.getElements("Plugin")
            elmList = XMLUtils.CC3DXMLListPy(elms)
            for elm in elmList:
                pluginName = elm.getAttribute("Name")
                print "   pluginName = ", pluginName  # e.g. CellType, Contact, etc
                if pluginName == "FocalPointPlasticity":
                    self.pluginFPPDefined = True
                    self.pluginCOMDefined = True  # if FPP is defined, COM will (implicitly) be defined

                if pluginName == "CenterOfMass":
                    self.pluginCOMDefined = True

            # If appropriate, disable/enable Vis menu options
            if not self.pluginFPPDefined:
                self.FPPLinksAct.setEnabled(False)
                self.FPPLinksAct.setChecked(False)
                Configuration.setSetting("FPPLinksOn", False)
            else:
                self.FPPLinksAct.setEnabled(True)

            if not self.pluginCOMDefined:
                self.cellGlyphsAct.setEnabled(False)
                self.cellGlyphsAct.setChecked(False)
                Configuration.setSetting("CellGlyphsOn", False)
            else:
                self.cellGlyphsAct.setEnabled(True)

        # ------------------
        if not self.mainGraphicsWidget: return

        self.mainGraphicsWidget.setStatusBar(self.__statusBar)

        self.mainGraphicsWidget.setZoomItems(self.zitems)  # Set zoomFixed parameters

        if self.borderAct.isChecked():  # Vis menu "Cell Borders" check box
            self.mainGraphicsWidget.showBorder()
        else:
            self.mainGraphicsWidget.hideBorder()

        if self.clusterBorderAct.isChecked():  # Vis menu "Cluster Borders" check box
            self.mainGraphicsWidget.showClusterBorder()

        # ---------------------
        if self.cellGlyphsAct.isChecked():  # Vis menu "Cell Glyphs"
            self.mainGraphicsWidget.showCellGlyphs()

        # ---------------------
        if self.FPPLinksAct.isChecked():  # Vis menu "FPP (Focal Point Plasticity) Links"
            self.mainGraphicsWidget.showFPPLinks()

        self.mainGraphicsWidget.setPlane(PLANES[0], 0)
        self.mainGraphicsWidget.currentDrawingObject.setPlane(PLANES[0], 0)

    def setParams(self):
        '''
        Calls __paramsChanged. Used from outside SimpleTabView
        :return:None
        '''
        self.__paramsChanged()

    def __paramsChanged(self):
        '''
        Slot linked to configsChanged signal - called after we hit 'OK' button on configuration dialog
        Also called during run initialization
        :return:None
        '''
        #        print MODULENAME,'  __paramsChanged():  do a bunch of Config--.getSetting'
        self.__updateScreen = Configuration.getSetting("ScreenUpdateFrequency")
        self.__imageOutput = Configuration.getSetting("ImageOutputOn")
        self.__shotFrequency = Configuration.getSetting("SaveImageFrequency")
        self.__latticeOutputFlag = Configuration.getSetting("LatticeOutputOn")
        self.__latticeOutputFrequency = Configuration.getSetting("SaveLatticeFrequency")
        self.__projectLocation = str(Configuration.getSetting("ProjectLocation"))
        self.__outputLocation = str(Configuration.getSetting("OutputLocation"))

        self.__outputDirectory = str(Configuration.getSetting("OutputLocation"))
        if Configuration.getSetting("OutputToProjectOn"):
            self.__outputDirectory = str(Configuration.getSetting("ProjectLocation"))

        # print MODULENAME, '__paramsChanged(),  prevOutputDir, __outputDirectory= ', self.prevOutputDir, self.__outputDirectory

        if (
                    self.__imageOutput or self.__latticeOutputFlag) and self.mysim:  # has user requested output and is there a valid sim?
            if self.screenshotDirectoryName == "":  # haven't created any yet
                #                print MODULENAME, '__paramsChanged(), screenshotDirName empty;  calling createOutputDirs'
                self.createOutputDirs()
            elif self.prevOutputDir != self.__outputDirectory:  # test if the sneaky user changed the output location
                #                print MODULENAME, '__paramsChanged(),  prevOutput != Output;  calling createOutputDirs'
                self.createOutputDirs()

                # NOTE: if self.mysim == None (i.e. sim hasn't begun yet), then createOutputDirs() should be called in __loadSim

        if self.simulation:
            self.init_simulation_control_vars()

    def setZoomItems(self, zitems):
        '''
        Deprecated - was used to set zoom items in the combo box. We do not use it any longer
        :param zitems: list of zoom items e.g. 25,50,100, 125 etc...
        :return:
        '''
        self.zitems = zitems

    def zoomIn(self):
        '''
        Slot called after user presses Zoom In button
        :return:None
        '''

        activeSubWindow = self.activeSubWindow()

        if not activeSubWindow:
            return

        import Graphics
        if isinstance(activeSubWindow.widget(), Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            activeSubWindow.widget().zoomIn()

    def zoomOut(self):
        '''
        Slot called after user presses Zoom Out button
        :return:None
        '''

        activeSubWindow = self.activeSubWindow()

        if not activeSubWindow:
            return

        import Graphics
        if isinstance(activeSubWindow.widget(), Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            activeSubWindow.widget().zoomOut()

    # # File name should be passed
    def takeShot(self):
        '''
        slot that adds screenshot configuration
        :return:None
        '''
        if self.screenshotManager is not None:
            if self.threeDRB.isChecked():
                camera = self.mainGraphicsWidget.ren.GetActiveCamera()
                # print "CAMERA SETTINGS =",camera
                self.screenshotManager.add3DScreenshot(self.__fieldType[0], self.__fieldType[1], camera)
            else:
                planePositionTupple = self.mainGraphicsWidget.getPlane()
                # print "planePositionTupple=",planePositionTupple
                self.screenshotManager.add2DScreenshot(self.__fieldType[0], self.__fieldType[1], planePositionTupple[0],
                                                       planePositionTupple[1])

    def prepareSimulationView(self):
        '''
        One of the initialization functions - prepares initial simulation view
        :return:
        '''
        if self.__fileName != "":
            file = QFile(self.__fileName)
            if file is not None:
                if self.mainGraphicsWidget is None:

                    self.showSimView(file)

                else:

                    # self.__closeSim()
                    # print 'BEFORE showSimView'
                    self.showSimView(file)
                    # print 'AFTER showSimView'

        self.drawingAreaPrepared = True
        self.updateActiveWindowVisFlags()  # needed in case switching from one sim to another (e.g. 1st has FPP, 2nd doesn't)

    def __openLDSFile(self, fileName=None):
        '''
        Opens Lattice Description File - for vtk replay mode
        :param fileName: str - .dml file name
        :return:none
        '''
        filter = "Lattice Description Summary file  (*.dml )"  # self._getOpenFileFilter()

        #        defaultDir = str(Configuration.getSetting('OutputLocation'))
        defaultDir = self.__outputDirectory

        if not os.path.exists(defaultDir):
            defaultDir = os.getcwd()

        self.fileName_tuple = QFileDialog.getOpenFileName(
            self.ui,
            QApplication.translate('ViewManager', "Open Lattice Description Summary file"),
            defaultDir,
            filter
        )

        self.__fileName = self.fileName_tuple[0]

        # converting Qstring to python string    and normalizing path
        self.__fileName = os.path.abspath(str(self.__fileName))
        from os.path import basename
        # setting text for main window (self.__parent) title bar
        self.__parent.setWindowTitle(basename(self.__fileName) + " - CompuCell3D Player")

        # Shall we inform the user?
        #        msg = QMessageBox.warning(self, "Message","Toggling off image & lattice output in Preferences",
        #                          QMessageBox.Ok ,
        #                          QMessageBox.Ok)
        Configuration.setSetting("ImageOutputOn", False)
        Configuration.setSetting("LatticeOutputOn", False)

    def __openRecentSim(self):
        '''
        Slot - opens recent simulation
        :return:None
        '''
        if self.simulationIsRunning:
            return

        action = self.sender()
        if isinstance(action, QAction):
            # self.__fileName = str(action.data().toString())
            self.__fileName = str(action.data())
        from os.path import basename
        # setting text for main window (self.__parent) title bar
        self.__parent.setWindowTitle(basename(self.__fileName) + " - CompuCell3D Player")

        import CompuCellSetup

        self.__fileName = os.path.abspath(self.__fileName)
        CompuCellSetup.simulationFileName = self.__fileName
        Configuration.setSetting("RecentFile", self.__fileName)
        #  each loaded simulation has to be passed to a function which updates list of recent files
        Configuration.setSetting("RecentSimulations", self.__fileName)

    def __openSim(self, fileName=None):
        '''
        This function is called when open file is triggered.
        Displays File open dialog to open new simulation
        :param fileName: str - unused
        :return:None
        '''

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
        # getOpenFilename may return tuple
        if isinstance(self.__fileName, tuple):
            self.__fileName = self.__fileName[0]

        # converting Qstring to python string and normalizing path
        self.__fileName = os.path.abspath(str(self.__fileName))

        print '__openSim: self.__fileName=', self.__fileName

        from os.path import basename
        # setting text for main window (self.__parent) title bar
        self.__parent.setWindowTitle(basename(self.__fileName) + " - CompuCell3D Player")

        """
        What is CompuCellSetup?
        It is located in ./core/pythonSetupScripts/CompuCellSetup.py
        
        """
        import CompuCellSetup
        CompuCellSetup.simulationFileName = self.__fileName

        # Add the current opening file to recent files and recent simulation
        Configuration.setSetting("RecentFile", self.__fileName)
        Configuration.setSetting("RecentSimulations",
                                 self.__fileName)  # each loaded simulation has to be passed to a function which updates list of recent files

    def __openScrDesc(self):
        '''
        Slot that opens screenshot description file
        :return:None
        '''

        preferred_dir = os.getcwd()

        if self.__fileName:
            preferred_dir = os.path.dirname(self.__fileName)

        filter = "Screenshot description file (*.sdfml)"  # self._getOpenFileFilter()
        screenshotDescriptionFileName_tuple = QFileDialog.getOpenFileName(
            self.ui,
            QApplication.translate('ViewManager', "Open Screenshot Description File"),
            preferred_dir,
            # os.getcwd(),
            filter
        )

        self.__screenshotDescriptionFileName = screenshotDescriptionFileName_tuple[0]

    def __saveScrDesc(self):
        '''
        Slot that opens file dialog to save screenshot description file
        :return:None
        '''
        # print "THIS IS __saveScrDesc"
        preferred_dir = os.getcwd()

        if self.__fileName:
            preferred_dir = os.path.dirname(self.__fileName)

        filter = "Screenshot Description File (*.sdfml )"  # self._getOpenFileFilter()
        screenshotDescriptionFileName_tuple = QFileDialog.getSaveFileName(
            self.ui,
            QApplication.translate('ViewManager', "Save Screenshot Description File"),
            preferred_dir,
            # os.getcwd(),
            filter
        )

        self.screenshotDescriptionFileName = screenshotDescriptionFileName_tuple[0]

        if self.screenshotManager:
            self.screenshotManager.writeScreenshotDescriptionFile(self.screenshotDescriptionFileName)

        print "self.screenshotDescriptionFileName=", self.screenshotDescriptionFileName

    # Sets the attribute self.movieSupport
    def __setMovieSupport(self):
        '''
        Experimental
        :return:
        '''
        self.movieSupport = False  # Is there vtkMPEG2Writer class in vtk module?
        vtkmod = inspect.getmembers(vtk, inspect.isclass)
        for i in range(len(vtkmod)):
            if vtkmod[i][0] == "vtkMPEG2Writer":
                self.movieSupport = True
                self.movieAct.setEnabled(True)
                return

        self.movieAct.setEnabled(False)

    def __checkMovieSupport(self, checked):
        '''
        Experimental
        :param checked:
        :return:
        '''
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
        '''
        Slot that triggers display of cells
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''

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
                        # print 'SHOWING CELLS ACTION'
                        graphicsWidget.showCells()
                        Configuration.setSetting('CellsOn', True)
                        self.cellsAct.setChecked(True)
                        win.activateWindow()
                    else:
                        # print 'HIDING CELLS ACTION'
                        graphicsWidget.hideCells()
                        Configuration.setSetting('CellsOn', False)
                        self.cellsAct.setChecked(False)
                        win.activateWindow()

                except AttributeError, e:
                    pass
                self.updateActiveWindowVisFlags(graphicsWidget)

        self.simulation.drawMutex.unlock()

    def __checkBorder(self, checked):
        '''
        Slot that triggers display of borders
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
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

        self.simulation.drawMutex.unlock()

    def __checkClusterBorder(self, checked):
        '''
        Slot that triggers display of cluster borders
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()

        self.updateActiveWindowVisFlags()
        if self.clusterBorderAct.isEnabled():
            # MDIFIX
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

        self.simulation.drawMutex.unlock()

    def __checkCellGlyphs(self, checked):
        '''
        Slot that triggers display of cell glyphs
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
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

            # MDIFIX
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

        self.simulation.drawMutex.unlock()

    def __checkFPPLinks(self, checked):
        '''
        Slot that triggers display of FPP links
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''

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

            # MDIFIX
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

        self.simulation.drawMutex.unlock()

    def __checkFPPLinksColor(self, checked):
        '''
        Slot that triggers display of colored FPP links
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        if checked and self.FPPLinksAct.isChecked():
            self.FPPLinksAct.setChecked(False)
            self.__checkFPPLinks(False)
        # if self.mainGraphicsWindow is not None:
        #                self.mainGraphicsWindow.hideFPPLinks()

        Configuration.setSetting("FPPLinksColorOn", checked)
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.updateActiveWindowVisFlags()

        if self.FPPLinksColorAct.isEnabled():

            # if self.lastActiveWindow is not None:
            # MDIFIX
            if self.lastActiveRealWindow is not None:
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

        self.simulation.drawMutex.unlock()

    def __checkContour(self, checked):
        '''
        Slot that triggers display of contours - may be deprecated
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        if self.contourAct.isEnabled():

            # MDIFIX
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

    def __checkLimits(self, checked):
        '''
        Placeholder function for concentration limits on/off - to be implemented
        :param checked:bool - flag determines if action is on or off
        :return:None
        '''
        pass

    def __resetCamera(self):
        '''
        Resets Camera for the current window
        :return:None
        '''
        # print 'INSIDE RESET CAMERA'
        activeSubWindow = self.activeSubWindow()
        # print 'activeSubWindow=', activeSubWindow
        if not activeSubWindow:
            return

        import Graphics
        if isinstance(activeSubWindow.widget(), Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            activeSubWindow.widget().resetCamera()

    def __checkCC3DOutput(self, checked):
        '''
        Slot that triggers display output information in the console -may not work properly without QT
        linked to the core cc3d code
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        Configuration.setSetting("CC3DOutputOn", checked)

    def __showConfigDialog(self, pageName=None):
        """
        Private slot to set the configurations.
        @param pageName name of the configuration page to show (string or QString)
        :return:None
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

        if len(self.fieldTypes) < 2:
            self.dlg.tab_field.setEnabled(False)
        else:
            self.dlg.tab_field.setEnabled(True)

        self.dlg.fieldComboBox.clear()

        for fieldName in activeFieldNamesList:
            self.dlg.fieldComboBox.addItem(fieldName)  # this is where we set the combobox of field names in Prefs

        # TODO - fix this - figure out if config dialog has configsChanged signal
        # self.connect(dlg, SIGNAL('configsChanged'), self.__configsChanged)
        # dlg.configsChanged.connect(self.__configsChanged)


        dlg.show()

        dlg.exec_()
        QApplication.processEvents()

        if dlg.result() == QDialog.Accepted:
            # Saves changes from all configuration pages!
            #            dlg.setPreferences()
            Configuration.syncPreferences()
            self.__configsChanged()  # Explicitly calling signal 'configsChanged'

    def __generatePIFFromCurrentSnapshot(self):
        '''
        Slot that generates PIFF file from current snapshot - calls either __generatePIFFromVTK or
        __generatePIFFromRunningSimulation depending on th4e running mode
        :return:None
        '''
        self.__pauseSim()
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
        pifFileName_selection = QFileDialog.getSaveFileName(
            self.ui,
            QApplication.translate('ViewManager', "Save PIF File As ..."),
            simFilePath,
            filter
        )

        # todo - have to recode C++ code to take unicode as filename...
        pifFileName = str(pifFileName_selection[0])
        self.simulation.generatePIFFromRunningSimulation(pifFileName)

    def __generatePIFFromVTK(self):
        '''
        Slot that generates PIFF file from current vtk replay snapshot - calls __generatePIFFromVTK
        :return:None
        '''

        if self.pauseAct.isEnabled():
            self.__pauseSim()

        fullSimFileName = os.path.abspath(self.__fileName)
        simFilePath = os.path.dirname(fullSimFileName)

        filter = "Choose PIF File Name (*.piff *.txt )"  # self._getOpenFileFilter()
        pifFileName_selection = QFileDialog.getSaveFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Save PIF File As ..."),
            simFilePath,
            filter
        )

        # todo - have to recode C++ code to take unicode as filename...
        pifFileName = str(pifFileName_selection[0])
        self.simulation.generatePIFFromVTK(self.simulation.currentFileName, pifFileName)

    def __configsChanged(self):
        """
        Private slot to handle a change of the preferences. Called after we hit Ok buttin on configuration dialog
        :return:None
        """
        self.configsChanged.emit()


    def setModelEditor(self, modelEditor):
        '''
        assigns model editor to a local variable - called from UserInterface.py
        :param modelEditor: model editor
        :return:
        '''
        self.__modelEditor = modelEditor

    def __createStatusBar(self):
        '''
        Creates Status bar layout
        :return:None
        '''

        self.__statusBar = self.__parent.statusBar()
        self.mcSteps = QLabel()
        self.mcSteps.setStyleSheet("QLabel { background-color : white; color : red; }")

        self.conSteps = QLabel()
        self.conSteps.setAutoFillBackground(True)
        self.conSteps.setStyleSheet("QLabel { background-color : white; color : blue; }")

        self.warnings = QLabel()
        self.warnings.setAutoFillBackground(True)
        self.warnings.setStyleSheet("QLabel { background-color : white; color : red; }")

        self.__statusBar.addWidget(self.mcSteps)
        self.__statusBar.addWidget(self.conSteps)
        self.__statusBar.addWidget(self.warnings)
