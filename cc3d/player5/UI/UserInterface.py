'''
This file holds the UI elements for CompuCell3D Player. Class UserInterface is the MainWindow of the
CompuCell3D player invoked from compucell3d_new.py file.
'''

# FIXME: Make the Console as a Dock window
# FIXME: When you open the XML file the second time, it doesn't expand the tree
# TODO: Make the tooltip for the description column in Plugins.

import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from cc3d.player5.UI.ModelEditor import ModelEditor
from cc3d.player5.Plugins.ViewManagerPlugins.SimpleTabView import SimpleTabView
from .LatticeDataModelTable import LatticeDataModelTable
from .CellTypeColorMapView import CellTypeColorMapView
from .CellTypeColorMapModel import CellTypeColorMapModel

from .Console import Console
from cc3d.player5.Utilities.SimModel import SimModel
from cc3d.player5.Utilities.LatticeDataModel import LatticeDataModel
from cc3d.player5.Utilities.SimDelegate import SimDelegate
from cc3d.player5 import Configuration
from cc3d.player5 import DefaultData

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
        features = QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        self.setFeatures(features)
        self.toggleFcn = None

    def setToggleFcn(self, fcn): self.toggleFcn = fcn

    def closeEvent(self, ev):
        print('DOCK WIDGET CLOSE EVENT')
        print('self.toggleFcn=', self.toggleFcn)

        if self.toggleFcn: self.toggleFcn(False)


class UserInterface(QMainWindow):
    appendStdoutSignal = pyqtSignal(str)
    appendStderrSignal = pyqtSignal(str)

    def __init__(self):
        QMainWindow.__init__(self)
        self.argv = None

        QApplication.setWindowIcon(QIcon(gip("cc3d_64x64_logo.png")))
        self.setWindowIcon(QIcon(gip("cc3d_64x64_logo.png")))
        self.setWindowTitle("CompuCell3D Player")

        self.origStdout = sys.stdout
        self.origStderr = sys.stderr

        self.__toolbars = {}

        # Setting self.viewmanager and dock windows
        self.__createViewManager()
        self.__createLayout()

        # # Generate the redirection helpers
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        # Now setup the connections
        if Configuration.getSetting("UseInternalConsole"):
            self.stdout = Redirector(False)
            self.stderr = Redirector(True)

            self.stdout.appendStdout.connect(self.appendToStdout)
            self.stderr.appendStderr.connect(self.appendToStderr)
            self.appendStdoutSignal.connect(self.console.appendToStdout)
            self.appendStderrSignal.connect(self.console.appendToStderr)

        # I don't know why I need this
        cc3dApp().registerObject("UserInterface", self)
        cc3dApp().registerObject("ViewManager", self.viewmanager)

        # Setup actions
        self.__initActions()
        # Setup menus
        self.__initMenus()

        # Setup toolbars
        self.__initToolbars()

        # Setup status bar
        self.__initStatusbar()

        # now redirect stdout and stderr
        if Configuration.getSetting("CC3DOutputOn"):

            if Configuration.getSetting("UseInternalConsole"):
                # redirecting Python output to internal console
                self.use_internal_console_for_python_output(True)
            else:
                # Python output goes to system console
                self.enable_python_output(True)
        else:
            # silencing output from Python
            self.enable_python_output(False)

        self.initialize_gui_geometry()

        floating_flag = Configuration.getSetting('FloatingWindows')

        floating_non_graphics_flag = False
        # we allow non-graphics windows to be docked. This improves
        # layout when we request floating grahics windows
        self.modelEditorDock.setFloating(floating_non_graphics_flag)
        self.consoleDock.setFloating(floating_non_graphics_flag)
        self.latticeDataDock.setFloating(floating_non_graphics_flag)

        if floating_flag:
            # in order to have all dock widgets expand (i.e. fill all available space)
            # we hide central widget when graphics windows are floating
            self.centralWidget().hide()

    def initialize_gui_geometry(self, allow_main_window_move:bool=True):
        """
        Initializes GUI geometry based on saved settings and based on current screen configuration
        :param allow_main_window_move: flag that specifies whether we may move main window according to settings or not
        We typically allow to moving of the main window at the GUI startup but not after loading simulation
        :return:
        """

        current_screen_geometry_settings = self.get_current_screen_geometry_settings()
        saved_screen_geometry_settings = Configuration.getSetting("ScreenGeometry")

        main_window_size = Configuration.getSetting("MainWindowSizeDefault")
        main_window_position = Configuration.getSetting("MainWindowPositionDefault")
        if self.viewmanager.MDI_ON:
            player_sizes = Configuration.getSetting("PlayerSizesDefault")
        else:
            player_sizes = Configuration.getSetting("PlayerSizesFloatingDefault")

        if current_screen_geometry_settings == saved_screen_geometry_settings:
            # this indicates that saved screen geometry is the same as current screen geometry and we will use
            # saved settings because we are working with same screen configuration so it is safe to restore
            if self.viewmanager.MDI_ON:
                # configuration of MDI
                main_window_size = Configuration.getSetting("MainWindowSize")
                main_window_position = Configuration.getSetting("MainWindowPosition")
                player_sizes = Configuration.getSetting("PlayerSizes")
            else:
                main_window_size = Configuration.getSetting("MainWindowSizeFloating")

                main_window_position = Configuration.getSetting("MainWindowPositionFloating")
                player_sizes = Configuration.getSetting("PlayerSizesFloating")

        self.resize(main_window_size)
        # we want main window to move only during initial opening of the GUI but not upon loading new simulation
        if allow_main_window_move:
            self.move(main_window_position)

        if player_sizes and player_sizes.size() > 0:
            self.restoreState(player_sizes)

    def save_ui_geometry(self):
        """
        Stores ui geometry settings . Called after user presses stop button
        :return:
        """
        if self.viewmanager.MDI_ON:
            Configuration.setSetting("PlayerSizes", self.saveState())
            Configuration.setSetting("MainWindowSize", self.size())
            Configuration.setSetting("MainWindowPosition", self.pos())

        else:
            Configuration.setSetting("PlayerSizesFloating", self.saveState())
            Configuration.setSetting("MainWindowSizeFloating", self.size())
            Configuration.setSetting("MainWindowPositionFloating", self.pos())

    def get_current_screen_geometry_settings(self):
        """
        Returns a list of screen coordinates that describe current screens arrangements. Covers multiple
        monitors
        :return:
        """
        geometry = []
        for screen in QApplication.screens():
            screen_rect = screen.availableGeometry()

            geometry += [screen_rect.x(), screen_rect.y(), screen_rect.width(), screen_rect.height()]

        return geometry

    def enable_python_output(self, _flag):

        if _flag:
            sys.stdout = self.origStdout
            sys.stderr = self.origStderr
        else:
            sys.stdout = NullDevice()
            sys.stderr = NullDevice()

    def use_internal_console_for_python_output(self, _flag):
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

    def appendToStderr(self, s):
        """
        Public slot to append text to the stderr log viewer tab.

        @param s output to be appended (string or QString)
        """
        self.showLogTab("stderr")
        self.appendStderrSignal.emit(s)

    def showLogTab(self, tabname):
        """
        Public method to show a particular Log-Viewer tab.

        @param tabname string naming the tab to be shown (string)
        """
        self.console.showLogTab(tabname)

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

        mb.addMenu(self.__menus["file"])

        self.__menus["view"] = QMenu("&View", self)
        mb.addMenu(self.__menus["view"])

        self.__menus["view"].aboutToShow.connect(self.__showViewMenu)

        self.__menus["toolbars"] = QMenu("&Toolbars", self.__menus["view"])
        self.__menus["toolbars"].setIcon(QIcon(gip("toolbars.png")))

        self.__menus["toolbars"].aboutToShow.connect(self.__showToolbarsMenu)
        self.__menus["toolbars"].triggered.connect(self.__TBMenuTriggered)

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

        self.__menus["help"] = self.viewmanager.init_help_menu()
        mb.addMenu(self.__menus["help"])

    def __initToolbars(self):

        sim_tb = self.viewmanager.init_sim_toolbar()
        file_tb = self.viewmanager.init_file_toolbar()

        visualization_tb = self.viewmanager.init_visualization_toolbar()
        window_tb = self.viewmanager.init_window_toolbar()

        self.addToolBar(sim_tb)
        self.addToolBar(file_tb)
        self.addToolBar(visualization_tb)
        self.addToolBar(window_tb)

        # just add new toolbars to the end of the list
        self.__toolbars["file"] = [file_tb.windowTitle(), file_tb]
        self.__toolbars["simulation"] = [sim_tb.windowTitle(), sim_tb]

    def closeEvent(self, event=None):

        Configuration.setSetting('ScreenGeometry', self.get_current_screen_geometry_settings())
        # this saves size and position of window when player is opened and closed without running simulation
        self.save_ui_geometry()

        self.viewmanager.closeEventSimpleTabView(event)

    def __initStatusbar(self):
        self.__statusBar = self.statusBar()
        self.__statusBar.setSizeGripEnabled(True)
        self.setStatusBar(self.__statusBar)

    def __initActions(self):
        """
        Private method to define the user interface actions.
        """
        self.actions = []

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

        if Configuration.getSetting('DisplayModelEditor'):
            self.modelAct.setChecked(True)

        self.modelAct.triggered.connect(self.toggleModelEditor)

        self.actions.append(self.modelAct)

        self.cell_type_color_map_act = QAction("Cell T&ype Color Map", self)
        self.cell_type_color_map_act.setCheckable(True)
        if Configuration.getSetting('DisplayCellTypeColorMap'):
            self.cell_type_color_map_act.setChecked(True)

        # if Configuration.getSetting('DisplayCellTypeColorMap'):
        #     self.self.cell_type_color_map_act.setChecked(True)

        self.cell_type_color_map_act.triggered.connect(self.toggle_cell_type_color_map_dock)

        self.actions.append(self.cell_type_color_map_act)

        self.pluginsAct = QAction("&Plugins", self)
        self.pluginsAct.setCheckable(True)
        self.pluginsAct.setChecked(True)
        self.pluginsAct.triggered.connect(self.__toggleCPlugins)

        self.actions.append(self.pluginsAct)

        self.latticeDataAct = QAction("&Lattice Data", self)
        self.latticeDataAct.setCheckable(True)
        if Configuration.getSetting('DisplayLatticeData'):
            self.latticeDataAct.setChecked(True)
        self.latticeDataAct.triggered.connect(self.toggleLatticeData)

        self.actions.append(self.latticeDataAct)

        self.consoleAct = QAction("&Console", self)
        self.consoleAct.setCheckable(True)

        self.toggleConsole(Configuration.getSetting('DisplayConsole'))

        if Configuration.getSetting('DisplayConsole'):
            self.consoleAct.setChecked(True)

        self.consoleAct.triggered.connect(self.toggleConsole)

        self.actions.append(self.consoleAct)

    def __zoomItems(self):
        items = []
        for i in range(len(self.zitems)):
            num = self.zitems[i] * 100
            items.append("%s%%" % int(num))

        return items

    def __createViewManager(self):
        self.zitems = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0]

        self.viewmanager = SimpleTabView(self)

        self.viewmanager.set_recent_simulation_file(str(Configuration.getSetting("RecentFile")))

        self.viewmanager.setZoomItems(self.zitems)

        self.setCentralWidget(self.viewmanager)
        self.setCentralWidget(self.viewmanager)

    def __createLayout(self):
        # Zoom items. The only place where the zoom items are specified!
        # Set up the model for the Model Editor
        self.modelEditorDock = self.__createDockWindow("ModelEditor")

        self.modelEditorDock.setToggleFcn(self.toggleModelEditor)
        model_editor = ModelEditor(self.modelEditorDock)

        self.model = SimModel(None, self.modelEditorDock)  # Do I need parent self.modelEditorDock
        model_editor.setModel(self.model)  # Set the default model
        model_editor.setItemDelegate(SimDelegate(self))
        model_editor.setParams()
        model_editor.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.viewmanager.setModelEditor(model_editor)  # Sets the Model Editor in the ViewManager
        self.__setupDockWindow(self.modelEditorDock, Qt.LeftDockWidgetArea, model_editor,
                               "Model Editor")  # projectBrowser

        self.latticeDataDock = self.__createDockWindow("LatticeData")
        self.latticeDataDock.setToggleFcn(self.toggleLatticeData)
        self.latticeDataModelTable = LatticeDataModelTable(self.latticeDataDock, self.viewmanager)
        self.latticeDataModel = LatticeDataModel()
        self.latticeDataModelTable.setModel(self.latticeDataModel)

        self.__setupDockWindow(self.latticeDataDock, Qt.LeftDockWidgetArea, self.latticeDataModelTable,
                               "LatticeDataFiles")
        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)


        self.cell_type_color_map_dock = self.__createDockWindow("CellTypeColorMapView")
        self.cell_type_color_map_dock.setToggleFcn(self.toggle_cell_type_color_map_dock)
        self.cell_type_color_map_model = CellTypeColorMapModel()
        self.cell_type_color_map_view = CellTypeColorMapView(parent=self.cell_type_color_map_dock, vm=self.viewmanager)
        # self.cell_type_color_map_view.setModel(self.cell_type_color_map_model)

        self.__setupDockWindow(self.cell_type_color_map_dock, Qt.LeftDockWidgetArea, self.cell_type_color_map_view,
                               "Cell Type Colors")

        self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)


        # Set up the console
        self.consoleDock = self.__createDockWindow("Console")

        self.consoleDock.setToggleFcn(self.toggleConsole)

        self.console = Console(self.consoleDock)
        self.consoleDock.setWidget(self.console)
        self.__setupDockWindow(self.consoleDock, Qt.BottomDockWidgetArea, self.console, "Console")

    def __createDockWindow(self, name):
        """
        Private method to create a dock window with common properties.

        @param name object name of the new dock window (string or QString)
        @return the generated dock window (QDockWindow)
        """
        dock = DockWidget(self)
        dock.setObjectName(name)

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
            caption = ""
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

    def toggle_cell_type_color_map_dock(self, flag):
        """

        :param flag:
        :return:
        """
        print ('toggle_cell_type_color_map_dock')
        self.cell_type_color_map_act.setChecked(flag)
        self.__toggleWindowFlag(self.cell_type_color_map_dock, flag)
        Configuration.setSetting('DisplayCellTypeColorMap', flag)

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
        self.consoleAct.setChecked(flag)

        Configuration.setSetting('DisplayConsole', flag)
        # TODO
        self.__toggleWindowFlag(self.consoleDock, flag)

    def __showViewMenu(self):
        """
        Private slot to display the Window menu.
        """
        self.__menus["view"].clear()

        self.__menus["view"].addMenu(self.__menus["toolbars"])

        self.__menus["view"].addAction(self.modelAct)
        self.modelAct.setChecked(not self.modelEditorDock.isHidden())

        # adding Cell Type Color Map to the menu
        self.__menus["view"].addAction(self.cell_type_color_map_act)
        self.cell_type_color_map_act.setChecked(not self.cell_type_color_map_dock.isHidden())

        self.__menus["view"].addAction(self.latticeDataAct)
        self.latticeDataAct.setChecked(not self.latticeDataDock.isHidden())

        # Plotting action. Leave it here
        self.__menus["view"].addAction(self.consoleAct)

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

    def __TBMenuTriggered(self, act):
        """
        Private method to handle the toggle of a toolbar.

        @param act reference to the action that was triggered (QAction)
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
            else:
                self.appendStdout.emit(line)

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
