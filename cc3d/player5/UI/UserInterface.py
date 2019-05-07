'''
This file holds the UI elements for CompuCell3D Player. Class UserInterface is the MainWindow of the
CompuCell3D player invoked from compucell3d_new.py file.
'''

# FIXME: Make the Console as a Dock window
# FIXME: When you open the XML file the second time, it doesn't expand the tree
# TODO: Make the tooltip for the description column in Plugins.

import os
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtXml import *

import cc3d.player5.ViewManager as ViewManager
# import ViewManager
# from cc3d.player5.UI.ViewManager import  ViewManager

# from .ModelEditor import ModelEditor
from cc3d.player5.UI.ModelEditor import ModelEditor

# TODO
from cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView import SimpleTabView

# TODO
from cc3d.player5.UI.CPlugins import CPlugins
from .LatticeDataModelTable import LatticeDataModelTable

# TODO
from .Console import Console

# TODO
from cc3d.player5.Utilities.QVTKRenderWidget import QVTKRenderWidget
import vtk

from cc3d.player5.Utilities.SimModel import SimModel
from cc3d.player5.Utilities.CPluginsModel import CPluginsModel
from cc3d.player5.Utilities.LatticeDataModel import LatticeDataModel
from cc3d.player5.Utilities.SimDelegate import SimDelegate
from cc3d.player5 import Configuration
from cc3d.player5 import  DefaultData

cc3dApp = QCoreApplication.instance

gip = DefaultData.getIconPath


class NullDevice:
    def write(self, s):
        pass

    def flush(self):
        pass


class DockWidget(QDockWidget):
    def __init__(self, _parent):
        super(DockWidget, self).__init__(_parent)

        self.toggleFcn = None

    def setToggleFcn(self, fcn): self.toggleFcn = fcn

    def closeEvent(self, ev):
        print('DOCK WIDGET CLOSE EVENT')
        print('self.toggleFcn=', self.toggleFcn)

        if self.toggleFcn: self.toggleFcn(False)
        # Configuration.setSetting(str(self.objectName(), False)

"""
This class represents the MainWindow of CompuCell3D.
"""
class UserInterface(QMainWindow):

    appendStdoutSignal = pyqtSignal(str)
    appendStderrSignal = pyqtSignal(str)

    def __init__(self):
        QMainWindow.__init__(self)
        self.argv = None
        # self.resize(QSize(900, 650))



        QApplication.setWindowIcon(QIcon(gip("cc3d_64x64_logo.png")))
        self.setWindowIcon(QIcon(gip("cc3d_64x64_logo.png")))
        self.setWindowTitle("CompuCell3D Player")

        self.origStdout = sys.stdout
        self.origStderr = sys.stderr
        # Setting self.viewmanager and dock windows
        # TODO
        self.__createViewManager()
        self.__createLayout()

        # # Generate the redirection helpers
        self.stdout = Redirector(False)
        self.stderr = Redirector(True)
        # #TODO
        self.stderr.setStdErrConsole(self.console.getStdErrConsole())

        # Now setup the connections
        if Configuration.getSetting("UseInternalConsole"):
            self.stdout.appendStdout.connect(self.appendToStdout)
            self.stderr.appendStderr.connect(self.appendToStderr)
            self.appendStdoutSignal.connect(self.console.appendToStdout)
            self.appendStderrSignal.connect(self.console.appendToStderr)


            # self.connect(self.stdout, SIGNAL('appendStdout'), self.appendToStdout)
            # self.connect(self.stderr, SIGNAL('appendStderr'), self.appendToStderr)
            #
            # self.connect(self, SIGNAL('appendStdout'), self.console.appendToStdout)
            # self.connect(self, SIGNAL('appendStderr'), self.console.appendToStderr)

        # I don't know why I need this
        cc3dApp().registerObject("UserInterface", self)
        cc3dApp().registerObject("ViewManager", self.viewmanager)

        self.__initActions()  # Setup actions
        self.__initMenus()  # Setup menus

        # self.__createViewManager()
        self.__initToolbars()  # Setup toolbars
        self.__initStatusbar()  # Setup status bar

        # self.tabWidget=QTabWidget(self)
        # self.tabWidget=SimpleTabView(self)
        # self.setCentralWidget(self.tabWidget)



        # now redirect stdout and stderr
        if Configuration.getSetting("CC3DOutputOn"):

            if Configuration.getSetting("UseInternalConsole"):
                # redirecting Python output to internal console
                self.useInternalConsoleForPythonOutput(True)
            else:
                # Python output goes to system console
                self.enablePythonOutput(True)
        else:
            # silencing output from Python
            self.enablePythonOutput(False)
        # TODO
        if self.viewmanager.MDI_ON:  # configuration of MDI
            playerSizes = Configuration.getSetting("PlayerSizes")
            if playerSizes and playerSizes.size() > 0:
                self.resize(Configuration.getSetting("MainWindowSize"))
                self.move(Configuration.getSetting("MainWindowPosition"))
                self.restoreState(playerSizes)
            else:
                self.resize(Configuration.getSetting("MainWindowSize"))
                self.move(Configuration.getSetting("MainWindowPosition"))
        else:  # configuration of floating windows
            playerSizes = Configuration.getSetting("PlayerSizesFloating")

            if playerSizes and playerSizes.size() > 0:
                self.resize(Configuration.getSetting("MainWindowSizeFloating"))
                self.resize(self.size().width(),
                            20)  # resizing vertical dimension to be minimal - for PyQt5 we cannot use 0
                self.move(Configuration.getSetting("MainWindowPositionFloating"))
                self.restoreState(playerSizes)
            else:
                self.resize(Configuration.getSetting("MainWindowSizeFloating"))
                self.resize(self.size().width(),
                            20)  # resizing vertical dimension to be minimal - for PyQt5 we cannot use 0
                self.move(Configuration.getSetting("MainWindowPositionFloating"))

        # if playerSizes and playerSizes.size()>0:
        #     self.resize(Configuration.getSetting("MainWindowSize"))
        #     self.move(Configuration.getSetting("MainWindowPosition"))
        #     self.restoreState(playerSizes)
        # else:
        #     self.resize(Configuration.getSetting("MainWindowSize"))
        #     self.move(Configuration.getSetting("MainWindowPosition"))

        # MDIFIX
        floatingFlag = Configuration.getSetting('FloatingWindows')
        self.modelEditorDock.setFloating(floatingFlag)
        self.consoleDock.setFloating(floatingFlag)
        self.latticeDataDock.setFloating(floatingFlag)

    ##########################################################
    ## Below are slots to handle StdOut and StdErr
    ##########################################################

    def enablePythonOutput(self, _flag):
        if _flag:
            sys.stdout = self.origStdout
            sys.stderr = self.origStderr
        else:
            sys.stdout = NullDevice()
            sys.stderr = NullDevice()

    def useInternalConsoleForPythonOutput(self, _flag):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def setArgv(self, _argv):
        self.argv = _argv

    def appendToStdout(self, s):
        """
        Public slot to append text to the stdout log viewer tab.

        @param s output to be appended (string or QString)
        """
        self.showLogTab("stdout")
        self.appendStdoutSignal.emit(s)
        # self.emit(SIGNAL('appendStdout'), s)

    def appendToStderr(self, s):
        """
        Public slot to append text to the stderr log viewer tab.

        @param s output to be appended (string or QString)
        """
        self.showLogTab("stderr")
        self.appendStderrSignal.emit(s)
        # self.emit(SIGNAL('appendStderr'), s)

    def showLogTab(self, tabname):
        """
        Public method to show a particular Log-Viewer tab.

        @param tabname string naming the tab to be shown (string)
        """
        self.console.showLogTab(tabname)

        # I don't think I need to show the dock widget
        # self.consoleDock.show()
        # self.consoleDock.raise_()

    def getMenusDictionary(self):
        return self.__menus

    def __initMenus(self):

        self.__menus = {}
        mb = self.menuBar()

        # TODO
        (fileMenu, recentSimulationsMenu) = self.viewmanager.init_file_menu()

        self.__menus["file"] = fileMenu
        self.__menus["recentSimulations"] = recentSimulationsMenu

        self.__menus["recentSimulations"].aboutToShow.connect(self.viewmanager.update_recent_file_menu)
        # self.connect(self.__menus["recentSimulations"] , SIGNAL("aboutToShow()"), self.viewmanager.updateRecentFileMenu )

        mb.addMenu(self.__menus["file"])

        self.__menus["view"] = QMenu("&View", self)
        mb.addMenu(self.__menus["view"])

        self.__menus["view"].aboutToShow.connect(self.__showViewMenu)
        # self.connect(self.__menus["view"], SIGNAL('aboutToShow()'), self.__showViewMenu)

        self.__menus["toolbars"] = QMenu("&Toolbars", self.__menus["view"])
        self.__menus["toolbars"].setIcon(QIcon(gip("toolbars.png")))

        self.__menus["toolbars"].aboutToShow.connect(self.__showToolbarsMenu)
        self.__menus["toolbars"].triggered.connect(self.__TBMenuTriggered)

        # self.connect(self.__menus["toolbars"], SIGNAL('aboutToShow()'), self.__showToolbarsMenu)
        # self.connect(self.__menus["toolbars"], SIGNAL('triggered(QAction *)'), self.__TBMenuTriggered)

        self.__showViewMenu()

        self.__menus["simulation"] = self.viewmanager.init_sim_menu()
        mb.addMenu(self.__menus["simulation"])
        self.__menus["visualization"] = self.viewmanager.init_visual_menu()
        mb.addMenu(self.__menus["visualization"])
        self.__menus["tools"] = self.viewmanager.init_tools_menu()
        mb.addMenu(self.__menus["tools"])

        self.__menus["window"] = self.viewmanager.init_window_menu()
        mb.addMenu(self.__menus["window"])

        self.__menus["window"].aboutToShow.connect(self.viewmanager.update_window_menu)
        # self.connect(self.__menus["window"] , SIGNAL("aboutToShow()"), self.viewmanager.updateWindowMenu )

        self.__menus["help"] = self.viewmanager.init_help_menu()
        mb.addMenu(self.__menus["help"])

    def __initToolbars(self):
        # TODO
        pass
        simtb = self.viewmanager.init_sim_toolbar()
        # filetb = self.viewmanager.initFileToolbar(self.toolbarManager)
        filetb = self.viewmanager.init_file_toolbar()

        # viewtb = QToolBar("View", self)
        # viewtb.setIconSize(QSize(20, 18))
        # viewtb.setObjectName("ViewToolbar")
        # viewtb.setToolTip("View")
        # viewtb.addAction(self.zoomInAct)
        # viewtb.addAction(self.zoomOutAct)
        # #viewtb.addAction(self.zoomFixedAct)
        # viewtb.addWidget(self.zoomFixed)
        # viewtb.addWidget(QLabel("  "))
        # viewtb.addAction(self.screenshotAct)

        visualizationtb = self.viewmanager.init_visualization_toolbar()
        windowtb = self.viewmanager.init_window_toolbar()

        # cstb = self.viewmanager.initCrossSectionToolbar() #QToolBar("Cross Section", self) #
        # threeDAct = QAction(self)
        # threeDRB  = QRadioButton("3D")

        # cstb.insertWidget(threeDAct, threeDRB)
        # viewtb = self.viewmanager.initViewToolbar()

        self.addToolBar(simtb)
        self.addToolBar(filetb)
        # self.addToolBar(viewtb)
        self.addToolBar(visualizationtb)
        self.addToolBar(windowtb)
        # self.addToolBar(cstb)

        # just add new toolbars to the end of the list
        self.__toolbars = {}
        self.__toolbars["file"] = [filetb.windowTitle(), filetb]
        # self.__toolbars["view"] = [viewtb.windowTitle(), viewtb]
        self.__toolbars["simulation"] = [simtb.windowTitle(), simtb]
        # self.__toolbars["crossSection"] = [cstb.windowTitle(), cstb]

    def closeEvent(self, event=None):
        print("CALLING CLOSE EVENT FROM  SIMTAB")
        # TODO check the rest of the function


        if self.viewmanager.MDI_ON:
            Configuration.setSetting("PlayerSizes", self.saveState())
            Configuration.setSetting("MainWindowSize", self.size())
            Configuration.setSetting("MainWindowPosition", self.pos())

        else:
            Configuration.setSetting("PlayerSizesFloating", self.saveState())
            Configuration.setSetting("MainWindowSizeFloating", self.size())
            Configuration.setSetting("MainWindowPositionFloating", self.pos())

        self.viewmanager.closeEventSimpleTabView(event)

    def __initStatusbar(self):
        self.__statusBar = self.statusBar()
        self.__statusBar.setSizeGripEnabled(True)
        self.setStatusBar(self.__statusBar)
        # self.__statusBar.showMessage("Welcome to CompuCell3D")

    def __initActions(self):
        """
        Private method to define the user interface actions.
        """
        self.actions = []
        # self.zoomInAct = QAction(QIcon(gip("zoomIn.png")), "&Zoom In", self)
        # self.actions.append(self.zoomInAct) # Replaced "viewActions" by "actions":self.viewActions.append(self.zoomInAct)
        # #
        # self.zoomOutAct = QAction(QIcon(gip("zoomOut.png")), "&Zoom Out", self)
        # self.actions.append(self.zoomOutAct)

        # # Why do I need self.zoomFixedAct?
        # #self.zoomFixedAct = QAction(self)
        # self.zoomFixed  = QComboBox()
        # self.zoomFixed.setToolTip("Zoom Fixed")
        # self.zoomFixed.addItems(self.__zoomItems())
        # self.zoomFixed.setCurrentIndex(3)
        # self.zoomFixed.addAction(self.zoomFixedAct)
        # self.actions.append(self.zoomFixedAct)

        # self.screenshotAct = QAction(QIcon("player5/icons/screenshot.png"), "&Take Screenshot", self)
        # self.actions.append(self.screenshotAct)




        self.toolbarFileAct = QAction("&File", self)
        self.toolbarFileAct.setCheckable(True)
        self.toolbarFileAct.setChecked(True)
        self.actions.append(self.toolbarFileAct)

        self.toolbarViewAct = QAction("&View", self)
        self.toolbarViewAct.setCheckable(True)
        self.toolbarViewAct.setChecked(True)
        self.actions.append(self.toolbarViewAct)

        self.toolbarSimAct = QAction("&Simulation", self)
        self.toolbarSimAct.setCheckable(True)
        self.toolbarSimAct.setChecked(True)
        self.actions.append(self.toolbarSimAct)
        self.toolbarSimAct.setShortcut(Qt.CTRL + Qt.Key_M)

        self.modelAct = QAction("&Model Editor", self)
        self.modelAct.setCheckable(True)
        # self.modelAct.setChecked(True) #not self.projectBrowserDock.isHidden()
        if Configuration.getSetting('DisplayModelEditor'):
            self.modelAct.setChecked(True)
        # self.connect(self.modelAct, SIGNAL("triggered(bool)"), self.toggleModelEditor)
        self.modelAct.triggered.connect(self.toggleModelEditor)
        # self.connect(self.modelAct, SIGNAL("triggered(bool)"), self.toggleModelEditor)

        self.actions.append(self.modelAct)

        self.pluginsAct = QAction("&Plugins", self)
        self.pluginsAct.setCheckable(True)
        self.pluginsAct.setChecked(True)
        # self.connect(self.pluginsAct, SIGNAL("triggered()"), self.__toggleCPlugins)
        self.pluginsAct.triggered.connect(self.__toggleCPlugins)

        self.actions.append(self.pluginsAct)

        self.latticeDataAct = QAction("&Lattice Data", self)
        self.latticeDataAct.setCheckable(True)
        if Configuration.getSetting('DisplayLatticeData'):
            self.latticeDataAct.setChecked(True)
        # self.latticeDataAct.setChecked(False)
        # self.connect(self.latticeDataAct, SIGNAL("triggered(bool)"), self.toggleLatticeData)
        self.latticeDataAct.triggered.connect(self.toggleLatticeData)

        self.actions.append(self.latticeDataAct)

        # self.connect(self.zoomInAct, SIGNAL('triggered()'), self.viewmanager.zoomIn)
        # self.connect(self.zoomOutAct, SIGNAL('triggered()'), self.viewmanager.zoomOut)
        # self.connect(self.zoomFixed, SIGNAL('activated(int)'), self.viewmanager.zoomFixed)
        # self.connect(self.screenshotAct, SIGNAL('triggered()'), self.viewmanager.takeShot)

        # self.connect(self.newGraphicsWindowAct, SIGNAL('triggered()'), self.viewmanager.addVTKWindowToWorkspace)


        # Keep this code. There won't be plotting in 3.4.0 version
        """
        self.plotAct = QAction("&Plot", self)
        self.plotAct.setCheckable(True)
        self.plotAct.setChecked(True)
        self.actions.append(self.plotAct)
        """

        self.consoleAct = QAction("&Console", self)
        self.consoleAct.setCheckable(True)

        self.toggleConsole(Configuration.getSetting('DisplayConsole'))

        if Configuration.getSetting('DisplayConsole'):
            self.consoleAct.setChecked(True)

        # self.connect(self.consoleAct, SIGNAL("triggered(bool)"), self.toggleConsole)
        self.consoleAct.triggered.connect(self.toggleConsole)

        self.actions.append(self.consoleAct)

        # I don't need probably to initActions() here. So I moved it to constructor
        # self.viewmanager.initActions()

    def __zoomItems(self):
        items = QStringList()
        for i in range(len(self.zitems)):
            num = self.zitems[i] * 100
            items.append("%s%%" % int(num))

        return items

    def setupDisplay3D(self):
        self.ren = vtk.vtkRenderer()
        self.display3D.GetRenderWindow().AddRenderer(self.ren)

        cone = vtk.vtkConeSource()
        cone.SetResolution(8)

        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInput(cone.GetOutput())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)

        self.ren.AddActor(coneActor)

        # show the widget
        self.display3D.show()

    def __createViewManager(self):
        self.zitems = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0]
        self.viewmanager = SimpleTabView(self)  # ViewManager.factory(self, self)

        self.viewmanager.set_recent_simulation_file(str(Configuration.getSetting("RecentFile")))

        self.viewmanager.setZoomItems(self.zitems)

        # self.viewmanager.setOrientation(Qt.Vertical)
        self.setCentralWidget(self.viewmanager)

    def __createLayout(self):
        # Zoom items. The only place where the zoom items are specified!
        # self.zitems = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0]


        # Set up the model for the Model Editor
        self.modelEditorDock = self.__createDockWindow("ModelEditor")

        self.modelEditorDock.setToggleFcn(self.toggleModelEditor)
        modelEditor = ModelEditor(self.modelEditorDock)

        # TODO
        # self.model = SimModel(QDomDocument(), self.modelEditorDock) # Do I need parent self.modelEditorDock
        self.model = SimModel(None, self.modelEditorDock)  # Do I need parent self.modelEditorDock
        modelEditor.setModel(self.model)  # Set the default model
        modelEditor.setItemDelegate(SimDelegate(self))
        modelEditor.setParams()
        modelEditor.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.viewmanager.setModelEditor(modelEditor)  # Sets the Model Editor in the ViewManager
        self.__setupDockWindow(self.modelEditorDock, Qt.LeftDockWidgetArea, modelEditor,
                               "Model Editor")  # projectBrowser

        self.latticeDataDock = self.__createDockWindow("LatticeData")
        self.latticeDataDock.setToggleFcn(self.toggleLatticeData)
        self.latticeDataModelTable = LatticeDataModelTable(self.latticeDataDock, self.viewmanager)
        self.latticeDataModel = LatticeDataModel()
        self.latticeDataModelTable.setModel(self.latticeDataModel)

        # # self.cplugins.latticeDataModelTable()
        # #self.connect(self.cplugins, SIGNAL("doubleClicked(const QModelIndex &)"), self.__showPluginView)
        #
        self.__setupDockWindow(self.latticeDataDock, Qt.LeftDockWidgetArea, self.latticeDataModelTable,
                               "LatticeDataFiles")
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)

        # Set up the console
        self.consoleDock = self.__createDockWindow("Console")

        self.consoleDock.setToggleFcn(self.toggleConsole)

        self.console = Console(self.consoleDock)
        self.consoleDock.setWidget(self.console)
        # self.consoleDock.setWindowTitle("Console")
        self.__setupDockWindow(self.consoleDock, Qt.BottomDockWidgetArea, self.console, "Console")
        # self.viewmanager.addWidget(self.consoleDock)
        # self.viewmanager.setSizes([400, 50])
        # self.consoleDock.show()

        # rec = self.console.geometry()
        # rec.setHeight(300)
        # print rec.height()
        # self.console.setGeometry(rec)
        # """
        # Don't know why I need dockwindows
        self.dockwindows = {}
        # self.dockwindows[0] = (self.trUtf8('Model Editor'), self.modelEditorDock)

    def __createDockWindow(self, name):
        """
        Private method to create a dock window with common properties.

        @param name object name of the new dock window (string or QString)
        @return the generated dock window (QDockWindow)
        """
        # dock = QDockWidget(self)
        dock = DockWidget(self)
        dock.setObjectName(name)
        # dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))
        return dock

    def __setupDockWindow(self, dock, where, widget, caption):
        """
        Private method to configure the dock window created with __createDockWindow().

        @param dock the dock window (QDockWindow)
        @param where dock area to be docked to (Qt.DockWidgetArea)
        @param widget widget to be shown in the dock window (QWidget)
        @param caption caption of the dock window (string or QString)
        """
        if caption is None:
            caption = QString()
        self.addDockWidget(where, dock)
        dock.setWidget(widget)
        dock.setWindowTitle(caption)
        dock.show()

    def toggleModelEditor(self, flag):
        """
        Private slot to handle the toggle of the Model Editor window.
        """
        self.modelAct.setChecked(flag)

        Configuration.setSetting('DisplayModelEditor', flag)
        self.__toggleWindowFlag(self.modelEditorDock, flag)

        # self.__toggleWindow(self.modelEditorDock)

    def __toggleCPlugins(self):
        """
        Private slot to handle the toggle of the Plugins window.
        """
        self.__toggleWindow(self.cpluginsDock)

    def toggleLatticeData(self, flag):
        """
        Private slot to handle the toggle of the Plugins window.
        """

        self.latticeDataAct.setChecked(flag)

        Configuration.setSetting('DisplayLatticeData', flag)
        self.__toggleWindowFlag(self.latticeDataDock, flag)

        # self.__toggleWindow(self.latticeDataDock)

    def __toggleWindow(self, w):
        """
        Private method to toggle a workspace editor window.

        @param w reference to the workspace editor window
        """
        if w.isHidden():
            w.show()
        else:
            w.hide()

    def __toggleWindowFlag(self, w, flag):
        """
        Private method to toggle a workspace editor window.

        @param w reference to the workspace editor window
        """

        print(' ')

        if flag:
            w.show()
        else:
            w.hide()

    def toggleConsole(self, flag):
        """
        Private slot to handle the toggle of the Log Viewer window.

        if self.layout == "DockWindows":
            self.__toggleWindow(self.logViewerDock)
        else:
            self.__toggleWindow(self.logViewer)
        """
        # self.__toggleWindow(self.consoleDock)

        # print ' TOGGLE CONSOLE FLAG = ', flag
        self.consoleAct.setChecked(flag)

        Configuration.setSetting('DisplayConsole', flag)
        # TODO
        self.__toggleWindowFlag(self.consoleDock, flag)

    def __showViewMenu(self):
        """
        Private slot to display the Window menu.
        """
        self.__menus["view"].clear()

        # Populate actions
        # self.__menus["view"].addAction(self.zoomInAct)
        # self.__menus["view"].addAction(self.zoomOutAct)
        # self.__menus["view"].addSeparator()
        # self.__menus["view"].addAction(self.screenshotAct)

        # self.__menus["view"].addSeparator()

        self.__menus["view"].addMenu(self.__menus["toolbars"])

        # self.__menus["view"].addSeparator()
        self.__menus["view"].addAction(self.modelAct)
        self.modelAct.setChecked(not self.modelEditorDock.isHidden())

        # # # self.__menus["view"].addAction(self.pluginsAct)
        # # # self.pluginsAct.setChecked(not self.cpluginsDock.isHidden())

        self.__menus["view"].addAction(self.latticeDataAct)
        self.latticeDataAct.setChecked(not self.latticeDataDock.isHidden())

        # Plotting action. Leave it here
        # self.__menus["view"].addAction(self.plotAct)
        self.__menus["view"].addAction(self.consoleAct)
        # self.consoleAct.setChecked(not self.consoleDock.isHidden())

    def __showToolbarsMenu(self):
        """
        Private slot to display the Toolbars menu.
        """
        self.__menus["toolbars"].clear()

        tbList = []
        for name, (text, tb) in list(self.__toolbars.items()):
            tbList.append((str(text), tb, name))

        tbList.sort()
        for text, tb, name in tbList:
            act = self.__menus["toolbars"].addAction(text)
            act.setCheckable(True)
            act.setData(QVariant(name))
            act.setChecked(not tb.isHidden())

        """
        #self.__menus["view"].addActions(self.viewProfileActGrp.actions())
        self.__menus["view"].addSeparator()

        # Set the options according to what is being displayed.
        self.__menus["window"].addAction(self.pbAct)
        if self.layout == "DockWindows":
            self.pbAct.setChecked(not self.projectBrowserDock.isHidden())
        else:
            self.pbAct.setChecked(not self.projectBrowser.isHidden())
        """

    def __TBMenuTriggered(self, act):
        """
        Private method to handle the toggle of a toolbar.

        @param act reference to the action that was triggered (QAction)
        """

        """
        if act == self.__toolbarsShowAllAct:
            for text, tb in self.__toolbars.values():
                tb.show()
            if self.__menus["toolbars"].isTearOffMenuVisible():
                self.__showToolb arsMenu()
        elif act == self.__toolbarsHideAllAct:
            for text, tb in self.__toolbars.values():
                tb.hide()
            if self.__menus["toolbars"].isTearOffMenuVisible():
                self.__showToolbarsMenu()
        else:
        """
        name = str(act.data().toString())
        if name:
            tb = self.__toolbars[name][1]
            if act.isChecked():
                tb.show()
            else:
                tb.hide()


class Redirector(QObject):
    """
    Helper class used to redirect stdout and stderr to the log window

    @signal appendStderr(string) emitted to write data to stderr logger
    @signal appendStdout(string) emitted to write data to stdout logger
    """
    appendStdout = pyqtSignal(str)
    appendStderr = pyqtSignal(str)

    def __init__(self, stderr):
        """
        Constructor

        @param stderr flag indicating stderr is being redirected
        """
        QObject.__init__(self)
        self.stderr = stderr
        self.buffer = ''
        self.stdErrConsole = None

    def setStdErrConsole(self, _stdErrConsole):
        self.stdErrConsole = _stdErrConsole

    def __nWrite(self, n):
        """
        Private method used to write data.

        @param n max numebr of bytes to write
        """
        if n:
            line = self.buffer[:n]
            if self.stderr:
                self.appendStderr.emit(line)
                # self.emit(SIGNAL('appendStderr'), line)
            else:
                self.appendStdout.emit(line)
                # self.emit(SIGNAL('appendStdout'), line)

            self.buffer = self.buffer[n:]

    def __bufferedWrite(self):
        """
        Private method returning number of characters to write.

        @return number of characters buffered or length of buffered line
        """
        return self.buffer.rfind('\n') + 1

    def flush(self):
        """
        Public method used to flush the buffered data.
        """
        self.__nWrite(len(self.buffer))

    def write(self, s):
        """
        Public method used to write data.

        @param s data to be written (it must support the str-method)
        """

        self.buffer = self.buffer + str(s)
        self.__nWrite(self.__bufferedWrite())
        # if self.stdErrConsole:
        # self.stdErrConsole.ensureCursorVisible()
        # else:
        # print "self.stdErrConsole=",self.stdErrConsole
