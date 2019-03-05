# todo - change settings core name to _-settings_3.0.sqlite
# todo - check if cml replay or regular simulation always exits cleanly right now
# todo there is a segfault after several repeated opens of simulations

# enabling with statement in python 2.5 - CC3D was first implemented with Python 2.2 or 2.3. Those were the times...

# -*- coding: utf-8 -*-
import os
import argparse
import sys
import re
import inspect
from collections import OrderedDict
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
from cc3d.core.enums import *
from os.path import basename, dirname, join, exists
from cc3d.player5.ViewManager.SimpleViewManager import SimpleViewManager
from cc3d.player5.Graphics.GraphicsFrameWidget import GraphicsFrameWidget
from cc3d.player5.Utilities.SimModel import SimModel
from cc3d.player5.Configuration.ConfigurationDialog import ConfigurationDialog
import cc3d.player5.Configuration as Configuration
import cc3d.core.DefaultSettingsData as settings_data
from cc3d.core.BasicSimulationData import BasicSimulationData
import cc3d.player5.DefaultData as DefaultData
from cc3d.player5.Graphics.GraphicsWindowData import GraphicsWindowData
from cc3d.player5.Simulation.CMLResultReader import CMLResultReader
from cc3d.player5.Simulation.SimulationThread import SimulationThread
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object
from cc3d.core import XMLUtils
from .PlotManagerSetup import createPlotManager
from .WidgetManager import WidgetManager
import PlayerPython
from cc3d.core.CMLFieldHandler import CMLFieldHandler

from . import ScreenshotManager
import vtk

from .RollbackImporter import RollbackImporter

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.readers import readCC3DFile
import cc3d.Version as Version

FIELD_TYPES = (
    "CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel", "CustomVis")

PLANES = ("xy", "xz", "yz")

MODULENAME = '---- SimpleTabView.py: '

# turning off vtkWindows console output
vtk.vtkObject.GlobalWarningDisplayOff()

try:
    python_module_path = os.environ["PYTHON_MODULE_PATH"]
    appended = sys.path.count(python_module_path)
    if not appended:
        sys.path.append(python_module_path)
    from cc3d import CompuCellSetup
except:
    print('STView: sys.path=', sys.path)

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


if Configuration.getSetting('FloatingWindows'):
    from .MainArea import MainArea
else:
    from .MainAreaMdi import MainArea


class SimpleTabView(MainArea, SimpleViewManager):
    configsChanged = pyqtSignal()

    def __init__(self, parent):

        # QMainWindow -> UI.UserInterface
        self.__parent = parent
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
            vtk_output = vtk.vtkOutputWindow.GetInstance()
            vtk_output.setOutputToWindowFlag(False)

        self.rollbackImporter = None

        # stores parsed command line arguments
        self.cml_args = None

        # object responsible for creating/managing plot windows so they're accessible from steppable level

        self.plotManager = createPlotManager(self)

        self.widgetManager = WidgetManager(self)

        self.fieldTypes = {}

        self.pluginTab = None
        self.mysim = None

        # gets assigned to SimulationThread down in prepareForNewSimulation()
        self.simulation = None

        # object that manages screenshots
        self.screenshotManager = None

        self.zitems = []

        self.__sim_file_name = ""  # simulation model filename

        self.__fieldType = ("Cell_Field", FIELD_TYPES[0])


        self.simulationIsStepping = False
        self.simulationIsRunning = False

        self.playerSettingsFileName = ""
        self.resultStorageDirectory = ""
        self.prevOutputDir = ""

        self.latticeType = Configuration.LATTICE_TYPES["Square"]
        self.newDrawingUserRequest = False
        self.completedFirstMCS = False

        self.fieldStorage = None
        self.fieldExtractor = None

        self.customSettingPath = ''
        self.cmlHandlerCreated = False

        self.basicSimulationData = None
        self.saveSettings = True

        self.closePlayerAfterSimulationDone = False

        self.__outputDirectory = ""

        self.__viewManagerType = "Regular"

        self.graphicsWindowVisDict = OrderedDict()  # stores visualization settings for each open window

        # self.lastActiveWindow = None
        self.lastPositionMainGraphicsWindow = None
        self.newWindowDefaultPlane = None

        self.cc3dSimulationDataHandler = None

        # for more information on QSignalMapper see Mark Summerfield book "Rapid GUI Development with PyQt"
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mapped.connect(self.set_active_sub_window_custom_slot)

        self.prepare_for_new_simulation(force_generic_initialization=True)

        self.setParams()
        self.mainGraphicsWidget = None  # vs.  lastActiveWindow

        # determine if some relevant plugins are defined in the model
        self.pluginFPPDefined = False  # FocalPointPlasticity
        self.pluginCOMDefined = False  # CenterOfMass

        # Note: we cannot check the plugins here as CompuCellSetup.cc3dXML2ObjConverter.root is not defined

        # nextSimulation holds the name of the file that will be inserted as a new simulation
        # to run after current simulation gets stopped
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
        except:
            # if for whatever reason we cannot do it we stay with the default value
            pass

            # note that this variable will be the same as self.simulation when doing CMLReplay mode.
            # I keep it under diffferent name to keep track of the places in the code where
            # I am using SimulationThread API and where I use CMLResultReade replay part of the API

        # this means that further refactoring is needed but I leave it for now
        self.cmlReplayManager = None

        # Here we are checking for new version - notice we use check interval in order not to perform version checks
        # too often. Default check interval is 7 days
        self.check_version(check_interval=7)


    def update_recent_file_menu(self) -> None:
        """
        Updates recent simulations File menu - called on demand only
        :return: None

        :return:
        """
        menus_dict = self.__parent.getMenusDictionary()
        rencent_simulations_menu = menus_dict["recentSimulations"]
        rencent_simulations_menu.clear()
        recent_simulations = Configuration.getSetting("RecentSimulations")

        sim_counter = 1
        for simulationFileName in recent_simulations:
            action_text = self.tr("&%1 %2").format(sim_counter, simulationFileName)
            action = QAction("&%d %s " % (sim_counter, simulationFileName), self)
            rencent_simulations_menu.addAction(action)
            action.setData(QVariant(simulationFileName))
            action.triggered.connect(self.open_recent_sim)

            sim_counter += 1
        return

    def set_active_sub_window_custom_slot(self, window: object) -> None:
        """
        Activates window
        :param window: QDockWidget or QMdiWindow instance
        :return:None

        """

        self.lastActiveRealWindow = window
        self.lastActiveRealWindow.activateWindow()

    def update_active_window_vis_flags(self, window: object = None) -> None:
        """
        Updates graphics visualization dictionary - checks if border, cells fpp links etc should be drawn
        :param window: QDockWidget or QMdiWindow instance - but only for graphics windows
        :return: None
        """

        try:
            if window:
                dict_key = window.winId().__int__()
            else:
                dict_key = self.lastActiveRealWindow.widget().winId().__int__()
        except Exception:
            print('update_active_window_vis_flags():  Could not find any open windows. Ignoring request')
            return

        self.graphicsWindowVisDict[dict_key] = (self.cellsAct.isChecked(), self.borderAct.isChecked(),
                                                self.clusterBorderAct.isChecked(), self.cellGlyphsAct.isChecked(),
                                                self.FPPLinksAct.isChecked())

    def update_window_menu(self) -> None:
        """
        Invoked whenever 'Window' menu is clicked.
        It does NOT modify lastActiveWindow directly (setActiveSubWindowCustomSlot does)
        :return:None
        """

        menus_dict = self.__parent.getMenusDictionary()
        window_menu = menus_dict["window"]
        window_menu.clear()
        window_menu.addAction(self.newGraphicsWindowAct)
        window_menu.addAction(self.pythonSteeringPanelAct)

        if self.MDI_ON:
            window_menu.addAction(self.tileAct)
            window_menu.addAction(self.cascadeAct)
            window_menu.addAction(self.minimizeAllGraphicsWindowsAct)
            window_menu.addAction(self.restoreAllGraphicsWindowsAct)
        window_menu.addSeparator()

        window_menu.addAction(self.closeActiveWindowAct)
        window_menu.addSeparator()

        # adding graphics windows
        counter = 0
        for win_id, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):

            graphics_widget = win.widget()

            if not graphics_widget:
                # happens with screenshot widget after simulation closes
                continue

            if graphics_widget.is_screenshot_widget:
                continue

            action_text = str("&{0}. {1}").format(counter + 1, win.windowTitle())

            action = window_menu.addAction(action_text)
            action.setCheckable(True)
            my_flag = self.lastActiveRealWindow == win
            action.setChecked(my_flag)

            action.triggered.connect(self.windowMapper.map)

            self.windowMapper.setMapping(action, win)
            counter += 1

        for win_id, win in self.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            action_text = self.tr("&{0}. {1}").format(counter + 1, win.windowTitle())

            action = window_menu.addAction(action_text)
            action.setCheckable(True)
            my_flag = self.lastActiveRealWindow == win
            action.setChecked(my_flag)

            action.triggered.connect(self.windowMapper.map)
            self.windowMapper.setMapping(action, win)
            counter += 1

    def addPythonSteeringPanel(self):
        '''
        callback method to create Steering Panel window with sliders
        :return: {None or mdiWindow}
        '''
        if not self.simulationIsRunning:
            return

        print('THIS IS ADD STEERING PANEL')
        from steering.SteeringParam import SteeringParam
        from steering.SteeringPanelView import SteeringPanelView
        from steering.SteeringPanelModel import SteeringPanelModel
        from steering.SteeringEditorDelegate import SteeringEditorDelegate

        self.item_data = []
        self.item_data.append(SteeringParam(name='vol', val=25, min_val=0, max_val=100, widget_name='slider'))
        self.item_data.append(
            SteeringParam(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))

        self.item_data.append(
            SteeringParam(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,
                          widget_name='slider'))

        self.steering_window = QWidget()
        layout = QHBoxLayout()

        # cdf = get_data_frame()
        self.steering_model = SteeringPanelModel()
        self.steering_model.update(self.item_data)
        # model.update_type_conv_fcn(get_types())

        self.steering_table_view = SteeringPanelView()
        self.steering_table_view.setModel(self.steering_model)

        delegate = SteeringEditorDelegate()
        self.steering_table_view.setItemDelegate(delegate)

        layout.addWidget(self.steering_table_view)
        self.steering_window.setLayout(layout)
        self.steering_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        mdiWindow = self.addSteeringSubWindow(self.steering_window)

        # IMPORTANT show() method needs to be called AFTER creating MDI subwindow
        self.steering_window.show()

        return mdiWindow

    def handle_vis_field_created(self, field_name: str, field_type: int) -> None:
        """
        slot that handles new visualization field creation. This mechanism is necessary to handle fields
        created outside steppable constructor
        :param field_name:
        :param field_type:
        :return:
        """

        self.fieldTypes[field_name] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[field_type]

        for win_id, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            graphics_frame = win.widget()
            graphics_frame.update_field_types_combo_box(field_types=self.fieldTypes)

    def add_new_graphics_window(self) -> object:
        """
        callback method to create additional ("Aux") graphics windows
        :return: None

        """

        if not self.simulationIsRunning:
            return None
        self.simulation.drawMutex.lock()

        new_window = GraphicsFrameWidget(parent=None, originatingWidget=self)

        # prepares new window for drawing - mainly sets reference to fieldExtractor
        new_window.initialize_scene()

        new_window.setZoomItems(self.zitems)  # Set zoomFixed parameters

        new_window.hide()

        self.configsChanged.connect(new_window.configsChanged)
        self.configsChanged.connect(new_window.configsChanged)

        mdi_window = self.addSubWindow(new_window)

        # MDIFIX
        self.lastActiveRealWindow = mdi_window

        # this happens when during restoration graphics window with id 0 had to be closed
        if self.mainGraphicsWidget is None:
            self.mainGraphicsWidget = mdi_window.widget()

        self.update_active_window_vis_flags()

        new_window.show()

        self.simulation.drawMutex.unlock()

        new_window.setConnects(self)  # in GraphicsFrameWidget
        new_window.setInitialCrossSection(self.basicSimulationData)
        new_window.setFieldTypesComboBox(self.fieldTypes)

        suggested_win_pos = self.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdi_window.move(suggested_win_pos)

        return mdi_window

    def add_vtk_window_to_workspace(self) -> None:
        """
        called one time, for initial graphics window  (vs. addNewGraphicsWindow())
        :return:
        """

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

        self.configsChanged.connect(self.mainGraphicsWidget.configsChanged)
        self.configsChanged.connect(self.mainGraphicsWidget.configsChanged)

        self.simulation.setGraphicsWidget(self.mainGraphicsWidget)

        mdi_sub_window = self.addSubWindow(self.mainGraphicsWidget)

        self.mainMdiSubWindow = mdi_sub_window
        self.mainGraphicsWidget.show()
        self.mainGraphicsWidget.setConnects(self)

        self.lastActiveRealWindow = mdi_sub_window

        # MDIFIX
        self.set_active_sub_window_custom_slot(self.lastActiveRealWindow)

        self.update_window_menu()
        self.update_active_window_vis_flags()

        suggested_win_pos = self.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdi_sub_window.move(suggested_win_pos)

    def minimize_all_graphics_windows(self) -> None:
        """
        Minimizes all graphics windows. Used ony with MDI window layout
        :return:None

        """

        if not self.MDI_ON: return

        for win_id, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showMinimized()

    def restore_all_graphics_windows(self) -> None:
        """
        Restores all graphics windows. Used ony with MDI window layout
        :return:None
        """
        if not self.MDI_ON: return

        for win_id, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            if win.widget().is_screenshot_widget:
                continue
            win.showNormal()

    def close_active_sub_window_slot(self):
        """
        This method is called whenever a user closes a graphics window - it is a slot for closeActiveWindow action
        :return:None

        """

        active_window = self.activeSubWindow()

        if not active_window:
            return

        active_window.close()

        self.update_window_menu()

    def process_command_line_options(self, cml_args:argparse.Namespace)->None:
        """
        initializes player internal variables based on command line input.
        Also if user passes appropriate option this function may get simulation going directly from command line
        :param cml_args: {}
        :return:
        """

        persistent_globals = CompuCellSetup.persistent_globals
        self.cml_args = cml_args

        start_simulation = False

        if cml_args.input:
            self.__sim_file_name = cml_args.input
            start_simulation = True

        self.__imageOutput = not cml_args.noOutput

        if cml_args.screenshotOutputDir:
            persistent_globals.set_screenshot_dir(screenshot_dir=cml_args.screenshotOutputDir)
            self.__imageOutput = True

        if cml_args.playerSettings:
            self.playerSettingsFileName = cml_args.playerSettings

        current_dir = cml_args.currentDir if cml_args.currentDir else ''
        if cml_args.windowSize:
            winSizes = cml_args.windowSize.split('x')
            # print MODULENAME, "  winSizes=", winSizes
            width = int(winSizes[0])
            height = int(winSizes[1])
            Configuration.setSetting("GraphicsWinWidth", width)
            Configuration.setSetting("GraphicsWinHeight", height)

        port = cml_args.port if cml_args.port else -1

        self.closePlayerAfterSimulationDone = cml_args.exitWhenDone

        if cml_args.guiScan:
            # when user uses gui to do parameter scan all we have to do is
            # to set self.closePlayerAfterSimulationDone to True
            self.closePlayerAfterSimulationDone = True

            # we reset max number of consecutive runs to 1 because we want each simulation in parameter scan
            # initiated by the psrun.py script to be an independent run after which player5
            # gets closed and reopened again for the next run
            self.maxNumberOfConsecutiveRuns = 1

        if cml_args.maxNumberOfConsecutiveRuns:
            self.maxNumberOfConsecutiveRuns = cml_args.maxNumberOfConsecutiveRuns

        self.UI.console.getSyntaxErrorConsole().setPlayerMainWidget(self)

        self.UI.console.getSyntaxErrorConsole().closeCC3D.connect(qApp.closeAllWindows)

        # establishConnection starts twedit and hooks it up via sockets to player5
        self.tweditAct.triggered.connect(self.UI.console.getSyntaxErrorConsole().cc3dSender.establishConnection)

        if port != -1:
            self.UI.console.getSyntaxErrorConsole().cc3dSender.setServerPort(port)

        # checking if file path needs to be remapped to point to files in the directories
        # from which run script was called
        sim_file_full_name = os.path.join(current_dir, self.__sim_file_name)


        self.__sim_file_name = sim_file_full_name
        CompuCellSetup.persistent_globals.simulation_file_name = self.__sim_file_name

        if start_simulation:
            if not os.access(self.__sim_file_name, os.F_OK):
                raise FileNotFoundError("Could not find simulation file: " + self.__sim_file_name)

        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=self.__sim_file_name)

        if self.playerSettingsFileName != '':
            player_settings_full_file_name = os.path.abspath(self.playerSettingsFileName)

            # checking if such a file exists
            if os.access(player_settings_full_file_name, os.F_OK):

                self.playerSettingsFileName = player_settings_full_file_name
            else:
                raise FileNotFoundError(
                    "Could not find playerSettings file: " + self.playerSettingsFileName)

        if start_simulation:
            self.__runSim()

    def set_recent_simulation_file(self, file_name:str)->None:
        """
        sets recent simulation file name
        :param file_name: {str}
        :return: None
        """

        self.__sim_file_name = file_name
        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=self.__sim_file_name)

        CompuCellSetup.persistent_globals.simulation_file_name = self.__sim_file_name

    def reset_control_buttons_and_actions(self)->None:
        """

        Resets control buttons and actions - called either after simulation is done
        (__cleanAfterSimulation) or in prepareForNewSimulation
        :return:

        """

        self.runAct.setEnabled(True)
        self.stepAct.setEnabled(True)
        self.pauseAct.setEnabled(False)
        self.stopAct.setEnabled(False)
        self.openAct.setEnabled(True)
        self.openLDSAct.setEnabled(True)
        self.pifFromSimulationAct.setEnabled(False)
        self.pifFromVTKAct.setEnabled(False)

    def reset_control_variables(self)->None:
        """
        Resets control variables - called either after simulation is done (__cleanAfterSimulation) or in prepareForNewSimulation
        :return: None
        """

        self.steppingThroughSimulation = False

        self.cmlHandlerCreated = False

        # CompuCellSetup.persistent_globals.simulation_file_name = None

        self.drawingAreaPrepared = False
        self.simulationIsRunning = False

        self.newDrawingUserRequest = False
        self.completedFirstMCS = False

    def prepare_for_new_simulation(self, force_generic_initialization: bool = False, in_stop_fcn: bool = False) -> None:
        """
        This function creates new instance of computational thread and sets various flags
        to initial values i.e. to a state before the beginning of the simulations
        """

        persistent_globals = CompuCellSetup.persistent_globals

        self.reset_control_buttons_and_actions()

        self.steppingThroughSimulation = False

        CompuCellSetup.persistent_globals.view_manager = self

        self.basicSimulationData = BasicSimulationData()

        # this import has to be here not inside is statement to ensure
        # that during switching from playing one type of files to another
        # there is no "missing module" issue due to improper imports

        # from Simulation.CMLResultReader import CMLResultReader

        self.cmlHandlerCreated = False

        # this is used to perform generic preparation for new simulation ,
        # normally called after "stop".
        # If users decide to use *.dml  prepare simulation will be called again with False argument
        if force_generic_initialization:
            persistent_globals.player_type = "new"

        if persistent_globals.player_type == "CMLResultReplay":
            self.__viewManagerType = "CMLResultReplay"

            # note that this variable will be the same as self.simulation when doing CMLReplay mode.
            # I keep it under diffferent name to keep track of the places in the code where I am using
            # SimulationThread API and where I use CMLResultReade replay part of the API
            # this means that further refactoring is needed but I leave it for now
            self.cmlReplayManager = self.simulation = CMLResultReader(self)

            self.simulation.extract_lattice_description_info(self.__sim_file_name)

            # filling out basic simulation data
            self.basicSimulationData.fieldDim = self.simulation.fieldDim
            self.basicSimulationData.numberOfSteps = self.simulation.numberOfSteps

            self.cmlReplayManager.initial_data_read.connect(self.initializeSimulationViewWidget)
            self.cmlReplayManager.subsequent_data_read.connect(self.handleCompletedStep)
            self.cmlReplayManager.final_data_read.connect(self.handleSimulationFinished)

            self.fieldExtractor = PlayerPython.FieldExtractorCML()
            self.fieldExtractor.setFieldDim(self.basicSimulationData.fieldDim)

        else:
            self.__viewManagerType = "Regular"

            # have to reinitialize cmlFieldHandler to None
            CompuCellSetup.persistent_globals.cml_field_handler = None

            self.simulation = SimulationThread(self)

            self.simulation.simulationInitializedSignal.connect(self.initializeSimulationViewWidget)
            self.simulation.steppablesStarted.connect(self.runSteppablePostStartPlayerPrep)
            self.simulation.simulationFinished.connect(self.handleSimulationFinished)
            self.simulation.completedStep.connect(self.handleCompletedStep)
            self.simulation.finishRequest.connect(self.handleFinishRequest)

            self.plotManager.initSignalAndSlots()
            self.widgetManager.initSignalAndSlots()

            self.fieldStorage = PlayerPython.FieldStorage()
            self.fieldExtractor = PlayerPython.FieldExtractor()
            self.fieldExtractor.setFieldStorage(self.fieldStorage)

        self.simulation.setCallingWidget(self)

        self.reset_control_variables()

    def prepare_area_for_new_simulation(self)->None:
        """
        Closes all open windows (from previous simulation) and creates new VTK window for the new simulation
        :return:
        """

        self.close_all_windows()

        self.add_vtk_window_to_workspace()

    def popup_message(self, title:str, msg:str)->None:
        """
        displays popup message window
        :param title: {str} title
        :param msg: {str} message
        :return: None
        """
        msg = QMessageBox.warning(self,
                                  title,
                                  msg,
                                  QMessageBox.Ok,
                                  QMessageBox.Ok
                                  )

    def handleErrorMessage(self, _errorType, _traceback_message):
        '''
        Callback function used to display any type of errors from the simulation script
        :param _errorType: str - error type
        :param _traceback_message: str - contains full Python traceback
        :return:
        '''
        msg = QMessageBox.warning(self, _errorType,
                                  _traceback_message,
                                  QMessageBox.Ok,
                                  QMessageBox.Ok)

        import ParameterScanEnums

        if _errorType == 'Assertion Error' and _traceback_message.startswith(
                'Parameter Scan ERRORCODE=' + str(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)):
            self.__cleanAfterSimulation(_exitCode=ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)
        else:
            self.__cleanAfterSimulation()
            print('errorType=', _errorType)
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

        print('DUPA GOT FORMATTED ERROR')
        return

    def processIncommingSimulation(self, _fileName, _stopCurrentSim=False):
        '''
        Callback function used to start new simulation. Currently invoked indirectly from the twedit++ when users choose
        "Open In Player" option form the CC3D project in the project context menu
        :param _fileName: str - simulation file name - full path
        :param _stopCurrentSim: bool , flag indicating if current simulation needs to be stopped
        :return:None
        '''
        print("processIncommingSimulation = ", _fileName, ' _stopCurrentSim=', _stopCurrentSim)
        if _stopCurrentSim:
            startNewSimulation = False
            if not self.simulationIsRunning and not self.simulationIsStepping:
                startNewSimulation = True

            self.__stopSim()

            import os

            self.__sim_file_name = os.path.abspath(str(_fileName))  # normalizing path
            import CompuCellSetup

            CompuCellSetup.simulationFileName = self.__sim_file_name

            if startNewSimulation:
                self.__runSim()
        else:
            self.__sim_file_name = _fileName
            self.nextSimulation = _fileName

        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=str(_fileName))
        # self.__parent.setWindowTitle(basename(str(_fileName)) + " - CompuCell3D Player")

    def prepareXMLTreeView(self):
        '''
        Initializes model editor tree view of the CC3DML - Model editor is used for steering
        :return:None
        '''

        # todo 5 -  restore model view
        return
        self.root_element = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter
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
        """
        Loads simulation
        :param file: str - full path to the CC3D simulation (usually .cc3d file or .dml vtk replay file path) .
        XML and python files are also acceptable options for the simulation but they are deprecated in favor of .cc3d
        :param file:
        :return:
        """

        self.prepare_for_new_simulation(force_generic_initialization=True)

        self.cc3dSimulationDataHandler = None

        file_name = str(self.__sim_file_name)
        CompuCellSetup.persistent_globals.simulation_file_name = self.__sim_file_name
        self.UI.console.bringUpOutputConsole()

        # have to connect error handler to the signal emited from self.simulation object
        # TODO changing signals
        self.simulation.errorOccured.connect(self.handleErrorMessage)
        self.simulation.errorFormatted.connect(self.handleErrorFormatted)

        self.simulation.visFieldCreatedSignal.connect(self.handle_vis_field_created)

        if re.match(".*\.cc3d$", file_name):
            self.__loadCC3DFile(file_name)

            self.__parent.toggleLatticeData(False)
            self.__parent.toggleModelEditor(True)

        elif re.match(".*\.dml$", file_name):
            self.__loadDMLFile(file_name=file_name)

        Configuration.setSetting("RecentFile", os.path.abspath(self.__sim_file_name))

        # each loaded simulation has to be passed to a function which updates list of recent files
        Configuration.setSetting("RecentSimulations", os.path.abspath(self.__sim_file_name))

    def __loadDMLFile(self, file_name: str) -> None:
        """
        loads lattice descriotion file and initializes simulation result replay
        :param file_name:
        :return:
        """
        persistent_globals = CompuCellSetup.persistent_globals
        # Let's toggle these off (and not tell the user for now)
        # need to make it possible to save images from .dml/vtk files
        if Configuration.getSetting("LatticeOutputOn"):
            QMessageBox.warning(self, "Message",
                                "Warning: Turning OFF 'Save lattice...' in Preferences",
                                QMessageBox.Ok)
            print('-----------------------')
            print('  WARNING:  Turning OFF "Save lattice" in Preferences|Output')
            print('-----------------------')
            Configuration.setSetting("LatticeOutputOn", False)

        if Configuration.getSetting("CellGlyphsOn"):
            QMessageBox.warning(self, "Message",
                                "Warning: Turning OFF 'Vis->Cell Glyphs' ",
                                QMessageBox.Ok)
            print('-----------------------')
            print('  WARNING:  Turning OFF "Vis->Cell Glyphs"')
            print('-----------------------')
            Configuration.setSetting("CellGlyphsOn", False)
            #                self.graphicsWindowVisDict[self.lastActiveWindow.winId()][3] = False
            self.cellGlyphsAct.setChecked(False)

        if Configuration.getSetting("FPPLinksOn"):
            QMessageBox.warning(self, "Message",
                                "Warning: Turning OFF 'Vis->FPP Links' ",
                                QMessageBox.Ok)
            print('-----------------------')
            print('  WARNING:  Turning OFF "Vis->FPP Links"')
            print('-----------------------')
            Configuration.setSetting("FPPLinksOn", False)
            #                self.graphicsWindowVisDict[self.lastActiveWindow.winId()][4] = False
            self.FPPLinksAct.setChecked(False)

        # todo 5 - old code

        # CompuCellSetup.playerType = "CMLResultReplay"
        # CompuCellSetup.parseXML(file_name)
        #
        # self.prepareForNewSimulation()
        #
        # CompuCellSetup.simulationPaths.setSimulationResultDescriptionFile(file_name)

        persistent_globals.player_type = 'CMLResultReplay'
        persistent_globals.cc3d_xml_2_obj_converter = CompuCellSetup.parseXML(file_name)

        self.prepare_for_new_simulation()

        self.__parent.toggleLatticeData(True)
        self.__parent.toggleModelEditor(False)

        self.prepareLatticeDataView()

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
        # import CC3DSimulationDataHandler
        # self.cc3dSimulationDataHandler = CC3DSimulationDataHandler.CC3DSimulationDataHandler(self)

        # Checking if the file is readable otherwise raising an error
        try:
            f = open(fileName, 'r')
            f.close()
        except IOError as e:
            msg = QMessageBox.warning(self, "Not A Valid Simulation File", \
                                      "Please make sure <b>%s</b> exists" % fileName, \
                                      QMessageBox.Ok)

            raise IOError("%s does not exist" % fileName)

        self.cc3dSimulationDataHandler = readCC3DFile(fileName=fileName)
        # self.cc3dSimulationDataHandler.readCC3DFileFormat(fileName)

        # check if current CC3D version is greater or equal to the version (minimal required version) specified in the project

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

        self.customSettingPath = self.cc3dSimulationDataHandler.cc3dSimulationData.custom_settings_path
        Configuration.initializeCustomSettings(self.customSettingPath)
        self.__paramsChanged()

        # todo 5  - previouse code - see if it still make ssense
        # # If project settings exists using the project settings
        # if self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource:
        #     self.customSettingPath = self.cc3dSimulationDataHandler.cc3dSimulationData.playerSettingsResource.path
        #     # print 'GOT CUSTOM SETTINGS RESOURCE = ', self.customSettingPath
        #     Configuration.initializeCustomSettings(self.customSettingPath)
        #     self.__paramsChanged()

        if self.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript != "":
            self.simulation.setRunUserPythonScriptFlag(True)

        # Else creating a project settings file
        else:
            self.customSettingPath = os.path.abspath(
                os.path.join(self.cc3dSimulationDataHandler.cc3dSimulationData.basePath, 'Simulation',
                             settings_data.SETTINGS_FILE_NAME))

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

            # todo 5- refactor this for parameter scans
            # try:
            #     from FileLock import FileLock
            #     with FileLock(file_name=fileName, timeout=10, delay=0.05)  as flock:
            #
            #         self.singleSimulation = False
            #         self.parameterScanFile = self.cc3dSimulationDataHandler.cc3dSimulationData.parameterScanResource.path  # parameter scan file path
            #         pScanFilePath = self.parameterScanFile
            #         # We use separate ParameterScanUtils object to handle parameter scan
            #         from ParameterScanUtils import ParameterScanUtils
            #
            #         psu = ParameterScanUtils()
            #
            #         psu.readParameterScanSpecs(pScanFilePath)
            #
            #         paramScanSpecsDirName = os.path.dirname(pScanFilePath)
            #
            #         outputDir = str(Configuration.getSetting('OutputLocation'))
            #
            #         customOutputPath = psu.prepareParameterScanOutputDirs(_outputDirRoot=outputDir)
            #
            #         self.cc3dSimulationDataHandler.copySimulationDataFiles(customOutputPath)
            #
            #         # construct path to the just-copied .cc3d file
            #         cc3dFileBaseName = os.path.basename(self.cc3dSimulationDataHandler.cc3dSimulationData.path)
            #         cc3dFileFullName = os.path.join(customOutputPath, cc3dFileBaseName)
            #
            #         psu.replaceValuesInSimulationFiles(_pScanFileName=pScanFilePath, _simulationDir=customOutputPath)
            #         # save parameter Scan spec file with incremented ityeration
            #         psu.saveParameterScanState(_pScanFileName=pScanFilePath)
            #
            #         self.__parent.setWindowTitle('ParameterScan: ' +
            #                                      basename(self.__sim_file_name) + ' Iteration: ' + basename(
            #             customOutputPath) + " - CompuCell3D Player")
            #
            #         # read newly created .cc3d file
            #         self.cc3dSimulationDataHandler.readCC3DFileFormat(cc3dFileFullName)
            #
            #         # # setting simultaion output dir names
            #         self.customScreenshotDirectoryName = customOutputPath
            #         CompuCellSetup.screenshotDirectoryName = customOutputPath
            #         self.screenshotDirectoryName = customOutputPath
            #         self.parameterScanOutputDir = customOutputPath
            #         # print 'self.screenshotDirectoryName=',self.screenshotDirectoryName
            #
            # except AssertionError as e:  # propagating exception
            #     raise e

        else:
            self.singleSimulation = True


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
        self.pythonSteeringPanelAct.triggered.connect(self.addPythonSteeringPanel)
        self.newGraphicsWindowAct.triggered.connect(self.add_new_graphics_window)

        self.tileAct.triggered.connect(self.tileSubWindows)
        self.cascadeAct.triggered.connect(self.cascadeSubWindows)

        self.minimizeAllGraphicsWindowsAct.triggered.connect(self.minimize_all_graphics_windows)
        self.restoreAllGraphicsWindowsAct.triggered.connect(self.restore_all_graphics_windows)

        self.closeActiveWindowAct.triggered.connect(self.close_active_sub_window_slot)
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

    def read_screenshot_description_file(self, scr_file=None):
        """
        Reads screenshot_description file
        :param scr_file: {str} scr file
        :return: None
        """

        try:
            self.screenshotManager.read_screenshot_description_file(screenshot_fname=scr_file)
        except RuntimeError as e:
            # self.screenshotManager.screenshotDataDict = {}
            self.popup_message(
                title='Error Parsing Screenshot Description',
                msg=str(e))
        except:
            self.screenshotManager.screenshotDataDict = {}
            self.popup_message(
                title='Error Parsing Screenshot Description',
                msg='Could not parse '
                    'screenshot description file {}. Try '
                    'removing old screenshot file and generate new one. No screenshots will be taken'.format(
                    self.screenshotManager.get_screenshot_filename()))

    def initializeSimulationViewWidgetCMLResultReplay(self):
        '''
        Initializes PLayer during VTK replay run mode
        :return:None
        '''

        self.fieldDim = self.simulation.fieldDim
        self.mysim = self.simulation.sim

        # currentl;y not supported - likely race conditions/synchronization because it works
        # when slowly stepping through the code but crashes during actual run
        # opening screenshot description file

        latticeTypeStr = self.simulation.latticeType
        if latticeTypeStr in list(Configuration.LATTICE_TYPES.keys()):
            self.latticeType = Configuration.LATTICE_TYPES[latticeTypeStr]
        else:
            self.latticeType = Configuration.LATTICE_TYPES["Square"]  # default choice

        # simulationDataIntAddr = self.extractAddressIntFromVtkObject(self.simulation.simulationData)

        simulationDataIntAddr = extract_address_int_from_vtk_object(self.simulation.simulationData)
        self.fieldExtractor.setSimulationData(simulationDataIntAddr)

        # this flag is used to prevent calling  draw function
        # when new data is read from hard drive
        # at this moment new data has been read and is ready to be used
        self.simulation.newFileBeingLoaded = False

        # this fcn will draw initial lattice configuration so data has to be available by then
        # and appropriate pointers set - see line above
        self.prepareSimulationView()

        if self.simulationIsStepping:
            self.__pauseSim()

        self.screenshotManager = ScreenshotManager.ScreenshotManager(self)

        self.read_screenshot_description_file()


        self.cmlReplayManager.keep_going()
        self.cmlReplayManager.set_stay_in_current_step(True)

    def createOutputDirs(self):
        '''
        Creates Simulation output directory
        :return:None
        '''

        # todo 5 - restore it
        return
        #        import pdb; pdb.set_trace()
        if self.customScreenshotDirectoryName == "":
            (self.screenshotDirectoryName, self.baseScreenshotName) = \
                CompuCellSetup.makeSimDir(self.__sim_file_name, self.__outputDirectory)

            CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName
            self.prevOutputDir = self.__outputDirectory

        else:
            # for parameter scan the directories are created in __loadCC3DFile
            if self.singleSimulation:

                (self.screenshotDirectoryName, self.baseScreenshotName) = \
                    self.makeCustomSimDir(self.customScreenshotDirectoryName, self.__sim_file_name)

                CompuCellSetup.screenshotDirectoryName = self.screenshotDirectoryName

            else:
                self.screenshotDirectoryName = self.parameterScanOutputDir

                pScanBaseFileName = os.path.basename(self.__sim_file_name)
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
            print(MODULENAME, '\n\n\n createOutputDirs():  self.mysim is None!!!')  # bad, very bad

        CompuCellSetup.initCMLFieldHandler(self.mysim(), self.resultStorageDirectory,
                                           self.fieldStorage)  # also creates the /LatticeData dir

    def initializeSimulationViewWidgetRegular(self)->None:
        """
        Initializes Player during simualtion run mode
        :return:
        """

        sim = CompuCellSetup.persistent_globals.simulator
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

        lattice_type_str = CompuCellSetup.simulation_utils.extract_lattice_type()

        if lattice_type_str in list(Configuration.LATTICE_TYPES.keys()):
            self.latticeType = Configuration.LATTICE_TYPES[lattice_type_str]
        else:
            self.latticeType = Configuration.LATTICE_TYPES["Square"]  # default choice

        self.prepareSimulationView()

        self.screenshotManager = ScreenshotManager.ScreenshotManager(self)
        self.read_screenshot_description_file()

        if self.simulationIsStepping:
            self.__pauseSim()

        self.prepareXMLTreeView()

    def initializeSimulationViewWidget(self):
        """
        Dispatch function - calls player5 initialization functions
        (initializeSimulationViewWidgetRegular or initializeSimulationViewWidgetCML) depending on the run mode
        :return:None
        """

        self.close_all_windows()

        initializeSimulationViewWidgetFcn = getattr(self, "initializeSimulationViewWidget" + self.__viewManagerType)
        initializeSimulationViewWidgetFcn()

        # copy simulation files to output directory  for single simulation
        # copying of the simulations files for parameter scan is done in the __loadCC3DFile

        screenshot_directory = CompuCellSetup.persistent_globals.screenshot_directory
        if self.singleSimulation:
            if self.cc3dSimulationDataHandler and screenshot_directory is not None:
                self.cc3dSimulationDataHandler.copySimulationDataFiles(screenshot_directory)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

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

    def handleSimulationFinishedCMLResultReplay(self, _flag):
        '''
        callback - runs after CML replay mode finished. Cleans after vtk replay
        :param _flag: bool - not used at tyhe moment
        :return:None
        '''
        persistent_globals = CompuCellSetup.persistent_globals
        if persistent_globals.player_type == "CMLResultReplay":
            self.latticeDataModelTable.prepareToClose()

        # # # self.__stopSim()
        self.__cleanAfterSimulation()

    def launchNextParameterScanRun(self):
        '''
        launches next aprameter scan. Deprecated - parameter scans shuld be run from command line
        :return:None
        '''
        fileName = self.__sim_file_name
        # when running parameter scan after simulatino finish we run again the same simulation file. When cc3d project with parameter scan gets opened 'next iteration' simulation is generatet and this
        # newly generated cc3d file is substituted instead of the "master" cc3d with parameter scan
        # From user stand point whan matters is that the only thing that user needs to worry abuot is the "master" .cc3d project and this is what is opened in the player5
        self.consecutiveRunCounter += 1
        if self.consecutiveRunCounter >= self.maxNumberOfConsecutiveRuns:

            from ParameterScanUtils import getParameterScanCommandLineArgList
            from SystemUtils import getCC3DPlayerRunScriptPath

            print('getCC3DPlayerRunScriptPath=', getCC3DPlayerRunScriptPath())
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
        print('INSIDE handleSimulationFinishedRegular')
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

        # synchronization  to allow CMLReader to read data, and report results and then
        # once data is ready drawing can happen.
        self.simulation.drawMutex.lock()

        simulationDataIntAddr = extract_address_int_from_vtk_object(self.simulation.simulationData)

        self.fieldExtractor.setSimulationData(simulationDataIntAddr)
        self.__step = self.simulation.currentStep

        # self.simulation.stepCounter is incremented by one before it reaches this function
        self.latticeDataModelTable.selectRow(self.simulation.stepCounter - 1)

        # there is additional locking inside draw to account for the fact that users may want to draw lattice on demand
        # self.simulation.newFileBeingLoaded=False

        # had to add synchronization here . without it I would get weird behavior in CML replay mode
        self.simulation.drawMutex.unlock()

        # at this moment new data has been read and is ready to be used
        self.__drawField()
        self.simulation.drawMutex.lock()
        # will need to synchronize screenshots with simulation thread .
        # make sure that before simulation thread writes new results all the screenshots are taken

        if self.__imageOutput and not (self.__step % self.__shotFrequency):
            self.screenshotManager.output_screenshots(self.__step)

        self.simulation.drawMutex.unlock()

        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

        self.cmlReplayManager.keep_going()

    def handleCompletedStepRegular(self, mcs: int) -> None:
        """
        callback - runs after simulation step completed.
        :param mcs: {int} current Monte Carlo step
        :return:
        """

        # creating cml field handler in case lattice output is ON
        if Configuration.getSetting("LatticeOutputOn") and not self.cmlHandlerCreated:
            persistent_globals = CompuCellSetup.persistent_globals
            persistent_globals.cml_field_handler = CMLFieldHandler()
            persistent_globals.cml_field_handler.initialize(field_storage=self.fieldStorage)
            self.cmlHandlerCreated = True

        self.__drawField()
        self.simulation.drawMutex.lock()

        # will need to sync screenshots with simulation thread.
        # Be sure before simulation thread writes new results all the screenshots are taken

        if self.__imageOutput and not (self.__step % self.__shotFrequency):  # dumping images? Check modulo MCS #

            if self.screenshotManager:
                try:
                    self.screenshotManager.output_screenshots(mcs=self.__step)
                except KeyError:

                    self.screenshotManager.screenshotDataDict = {}
                    self.popup_message(
                        title='Error Processing Screnenshots',
                        msg='Could not output screenshots. It is likely that screenshot description file was generated '
                            'using incompatible code. '
                            'You may want to remove "screenshot_data" directory from your project '
                            'and use camera button to generate new screenshot file '
                            ' No screenshots will be taken'.format(self.screenshotManager.get_screenshot_filename()))

        if self.cmlHandlerCreated and self.__latticeOutputFlag and (not self.__step % self.__latticeOutputFrequency):
            CompuCellSetup.persistent_globals.cml_field_handler.write_fields(self.__step)

        self.simulation.drawMutex.unlock()

        if self.simulationIsStepping:
            self.__pauseSim()
            self.stepAct.setEnabled(True)

        self.simulation.sem.tryAcquire()
        self.simulation.sem.release()

    def handleCompletedStep(self, mcs:int)->None:
        """
        Dispatch function for handleCompletedStep functions

        :param mcs: int - current Monte Carlo step

        :return:None

        """

        if not self.completedFirstMCS:
            self.completedFirstMCS = True

        self.__step = mcs

        handle_completed_step_fcn = getattr(self, "handleCompletedStep" + self.__viewManagerType)
        handle_completed_step_fcn(mcs)

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
            if self.__sim_file_name == "":
                msg = QMessageBox.warning(self, "Not A Valid Simulation File", \
                                          "Please pick simulation file <b>File->OpenSimulation File ...</b>", \
                                          QMessageBox.Ok,
                                          QMessageBox.Ok)
                return False
            file = QFile(self.__sim_file_name)

            import xml

            try:
                self.__loadSim(file)
            except AssertionError as e:
                print("Assertion Error: ", str(e))

                self.handleErrorMessage("Assertion Error", str(e))
                import ParameterScanEnums

                if _errorType == 'Assertion Error' and _traceback_message.startswith(
                        'Parameter Scan ERRORCODE=' + str(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)):
                    #                     print 'Exiting inside prepare simulation '
                    sys.exit(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)

                return False
            except xml.parsers.expat.ExpatError as e:
                # todo 5 - fix this - simulationPaths does not exit
                xmlFileName = CompuCellSetup.simulationPaths.simulationXMLFileName
                print("Error in XML File", "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message)
                self.handleErrorMessage("Error in XML File",
                                        "File:\n " + xmlFileName + "\nhas the following problem\n" + e.message)
            except IOError:
                return False

            self.init_simulation_control_vars()

            if self.rollbackImporter:
                self.rollbackImporter.uninstall()

            self.rollbackImporter = RollbackImporter()

            return True

    def __runSim(self):
        '''
        Slot that actuallt runs the simulation
        :return:None
        '''

        print('INSIDE RUN SIM')
        # self.simulation.start()

        # when we run simulation we ensure that self.simulation.screenUpdateFrequency
        # is whatever is written in the settings
        self.simulation.screenUpdateFrequency = self.__updateScreen

        if not self.drawingAreaPrepared:
            prepare_flag = self.prepareSimulation()
            if prepare_flag:
                # todo 5 - self.drawingAreaPrepared is initialized elsewhere this is tmp placeholder and a hack
                self.drawingAreaPrepared = True
            else:
                return

        # print 'SIMULATION PREPARED self.__viewManagerType=',self.__viewManagerType
        if self.__viewManagerType == "CMLResultReplay":

            self.simulation.semPause.release()  # just in case

            # these flags settings calls have to be executed before self.cmlReplayManager.keepGoing()
            self.simulationIsRunning = True
            self.simulationIsStepping = False

            self.cmlReplayManager.set_run_state(state=RUN_STATE)

            self.cmlReplayManager.keep_going()

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

            prepare_flag = self.prepareSimulation()
            if prepare_flag:
                # todo 5 - self.drawingAreaPrepared is initialized elsewhere this is tmp placeholder and a hack
                self.drawingAreaPrepared = True
            else:
                return

        if self.__viewManagerType == "CMLResultReplay":

            self.simulation.semPause.release()
            self.simulationIsRunning = True
            self.simulationIsStepping = True

            # self.cmlReplayManager.setStepState()
            self.cmlReplayManager.set_run_state(state=STEP_STATE)

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
                # self.screenshotDirectoryName = ""

                self.runAct.setEnabled(True)
                self.pauseAct.setEnabled(False)
                self.stopAct.setEnabled(True)
                self.pifFromSimulationAct.setEnabled(True)
                self.openAct.setEnabled(False)
                self.openLDSAct.setEnabled(False)

                self.simulation.start()

            if self.simulationIsRunning and self.simulationIsStepping:
                #            print MODULENAME,'  __stepSim() - 1:'
                self.pauseAct.setEnabled(False)
                self.simulation.semPause.release()
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)

                return

            # if Pause button is enabled
            elif self.simulationIsRunning and not self.simulationIsStepping and self.pauseAct.isEnabled():
                # transition from running simulation

                self.simulation.screenUpdateFrequency = 1
                self.simulation.screenshotFrequency = self.__shotFrequency
                self.simulationIsStepping = True
                self.stepAct.setEnabled(False)
                self.pauseAct.setEnabled(False)

            # if Pause button is disabled, meaning the sim is paused:
            elif self.simulationIsRunning and not self.simulationIsStepping and not self.pauseAct.isEnabled():
                # transition from paused simulation
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

    def drawFieldCMLResultReplay(self) -> None:
        """
        Draws fields during vtk replay mode
        :return:
        """

        self.simulation.drawMutex.lock()
        self.simulation.readFileSem.acquire()

        if not self.simulationIsRunning:
            self.simulation.drawMutex.unlock()
            self.simulation.readFileSem.release()

            return

        if self.newDrawingUserRequest:
            self.newDrawingUserRequest = False
            if self.pauseAct.isEnabled():
                self.__pauseSim()

        self.simulation.drawMutex.unlock()
        self.simulation.readFileSem.release()

        self.simulation.drawMutex.lock()
        self.simulation.readFileSem.acquire()

        self.__step = self.simulation.getCurrentStep()

        if True:
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsFrame = win.widget()

                if graphicsFrame.is_screenshot_widget:
                    continue

                (currentPlane, currentPlanePos) = graphicsFrame.getPlane()

                # this flag is used to prevent calling  draw function when new data is read from hard drive
                # if not self.simulation.newFileBeingLoaded:
                if not self.simulation.data_ready:
                    # graphicsFrame.drawFieldLocal(self.basicSimulationData)
                    graphicsFrame.draw(self.basicSimulationData)
                # todo 5
                # self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())
                self.__updateStatusBar(self.__step)

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

                # todo 5
                # graphicsFrame.drawFieldLocal(self.basicSimulationData)
                graphicsFrame.draw(self.basicSimulationData)
                # todo 5
                # self.__updateStatusBar(self.__step, graphicsFrame.conMinMax())  # show MCS in lower-left GUI
                self.__updateStatusBar(self.__step)  # show MCS in lower-left GUI

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
            # graphicsWidget.resetAllCameras()
            graphicsWidget.reset_all_cameras(self.basicSimulationData)

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
            # for some reason cameras have to be initialized after drawing resized lattice and draw function has to be repeated
            self.updateVisualization()

        __drawFieldFcn()

    def displayWarning(self, warning_text):
        '''
        Displays Warnings in the status bar
        :param warning_text: str - warning text
        :return:None
        '''
        self.warnings.setText(warning_text)

    def __updateStatusBar(self, step):
        '''
        Updates status bar
        :param step: int - current MCS
        :return:
        '''
        self.mcSteps.setText("MC Step: %s" % step)
        # self.conSteps.setText("Min: %s Max: %s" % conMinMax)

    def __pauseSim(self):
        '''
        slot that pauses simulation
        :return:None
        '''
        if self.__viewManagerType == "CMLResultReplay":
            self.cmlReplayManager.set_run_state(state=PAUSE_STATE)

        self.simulation.semPause.acquire()
        self.runAct.setEnabled(True)
        self.pauseAct.setEnabled(False)

    def __saveWindowsLayout(self):
        """
        Saves windows layout in the _settings.xml
        :return:None
        """

        windowsLayout = {}

        for key, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
            print('key, win = ', (key, win))
            widget = win.widget()
            # if not widget.allowSaveLayout: continue
            if widget.is_screenshot_widget:
                continue

            gwd = widget.getGraphicsWindowData()

            # fill size and position of graphics windows data using mdiWidget,
            # NOT the internal widget such as GraphicsFrameWidget - sizes and positions are base on MID widet settings
            gwd.winPosition = win.pos()
            gwd.winSize = win.size()

            # print 'getGraphicsWindowData=', gwd
            # print 'toDict=', gwd.toDict()

            windowsLayout[key] = gwd.toDict()

        # print 'AFTER self.fieldTypes = ', self.fieldTypes
        try:
            print(self.plotManager.plotWindowList)
        except AttributeError:
            print("plot manager does not have plotWindowList member")

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
            self.cmlReplayManager.set_run_state(state=STOP_STATE)

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
        print('error_code = ', error_code)
        QCoreApplication.instance().exit(error_code)
        print('AFTER QtCore.QCoreApplication.instance()')

    def __cleanAfterSimulation(self, _exitCode=0):
        '''
        Cleans after simulation is done
        :param _exitCode: exit code from the simulation
        :return:None
        '''

        self.reset_control_buttons_and_actions()
        self.reset_control_variables()

        # re-init (empty) the fieldTypes dict, otherwise get previous/bogus fields in graphics win field combobox
        self.fieldTypes = {}

        # saving settings with the simulation
        if self.customSettingPath:
            Configuration.writeSettingsForSingleSimulation(self.customSettingPath)
            self.customSettingPath = ''

        Configuration.writeAllSettings()
        Configuration.initConfiguration()  # this flushes configuration

        if Configuration.getSetting("ClosePlayerAfterSimulationDone") or self.closePlayerAfterSimulationDone:
            Configuration.setSetting("RecentFile", os.path.abspath(self.__sim_file_name))

            Configuration.setSetting("RecentSimulations", os.path.abspath(self.__sim_file_name))

            # sys.exit(_exitCode)
            self.quit(CompuCellSetup.error_code)

        # in case there is pending simulation to be run we will put it a recent simulation
        # so that it can be ready to run without going through open file dialog
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
        for win in list(self.win_inventory.values()):
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

        self.prepare_area_for_new_simulation()

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

        # todo 5 -  reimplement it

        windowsLayoutDict = Configuration.getSetting('WindowsLayout')
        print('from settings windowsLayout = ', windowsLayoutDict)

        # first restore main window with id 0 - this window is the only window open at this point and it is open by default when simulation is started
        # that's why we have to treat it in a special way but only when we determine that windowsLayoutDict is not empty
        if len(list(windowsLayoutDict.keys())):
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

                # from Graphics.GraphicsWindowData import GraphicsWindowData

                gwd = GraphicsWindowData()

                gwd.fromDict(windowDataDict0)

                if gwd.winType == GRAPHICS_WINDOW_LABEL:
                    graphicsWindow = self.lastActiveRealWindow
                    gfw = graphicsWindow.widget()

                    graphicsWindow.resize(gwd.winSize)
                    graphicsWindow.move(gwd.winPosition)

                    gfw.apply_graphics_window_data(gwd)

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
        for windowId, windowDataDict in windowsLayoutDict.items():
            if windowId == 0 or windowId == '0':
                continue

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

            gwd = GraphicsWindowData()

            gwd.fromDict(windowDataDict)

            if gwd.winType != GRAPHICS_WINDOW_LABEL:
                continue

            if gwd.sceneName not in list(self.fieldTypes.keys()):
                continue  # we only create window for a sceneNames (e.g. fieldNames) that exist in the simulation

            graphicsWindow = self.add_new_graphics_window()
            gfw = graphicsWindow.widget()

            graphicsWindow.resize(gwd.winSize)
            graphicsWindow.move(gwd.winPosition)

            gfw.apply_graphics_window_data(gwd)

            # print ' PLOT WINDOW MANAGER  WINDOW LIST = ', self.plotManager.plotWindowList

    def setFieldTypesCML(self):
        '''
        initializes field types for VTK vidgets during vtk replay mode
        :return:None
        '''

        # Add cell field
        self.fieldTypes["Cell_Field"] = FIELD_TYPES[0]  # "CellField"

        self.fieldComboBox.clear()
        # self.fieldComboBox.addItem("-- Field Type --")
        self.fieldComboBox.addItem("Cell_Field")

        for fieldName in list(self.simulation.fieldsUsed.keys()):
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

        extra_field_registry = CompuCellSetup.persistent_globals.field_registry

        # inserting extra scalar fields managed from Python script
        field_dict = extra_field_registry.get_fields_to_create_dict()

        for field_name, field_adapter in field_dict.items():
            self.fieldTypes[field_name] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[field_adapter.field_type]

        # # inserting extra scalar fields managed from Python script
        # scalarFieldNameVec = self.fieldStorage.getScalarFieldNameVector()
        # for fieldName in scalarFieldNameVec:
        #     #            print MODULENAME,"setFieldTypes():  Got this scalar field: ",fieldName
        #     self.fieldTypes[fieldName] = FIELD_TYPES[2]
        #
        # # inserting extra scalar fields cell levee managed from Python script
        # scalarFieldCellLevelNameVec = self.fieldStorage.getScalarFieldCellLevelNameVector()
        # for fieldName in scalarFieldCellLevelNameVec:
        #     #            print MODULENAME,"setFieldTypes():  Got this scalar field (cell leve): ",fieldName
        #     self.fieldTypes[fieldName] = FIELD_TYPES[3]
        #
        # # inserting extra vector fields  managed from Python script
        # vectorFieldNameVec = self.fieldStorage.getVectorFieldNameVector()
        # for fieldName in vectorFieldNameVec:
        #     #            print MODULENAME,"setFieldTypes():  Got this vector field: ",fieldName
        #     self.fieldTypes[fieldName] = FIELD_TYPES[4]
        #
        # # inserting extra vector fields  cell level managed from Python script
        # vectorFieldCellLevelNameVec = self.fieldStorage.getVectorFieldCellLevelNameVector()
        # for fieldName in vectorFieldCellLevelNameVec:
        #     #            print MODULENAME,"setFieldTypes():  Got this vector field (cell level): ",fieldName
        #     self.fieldTypes[fieldName] = FIELD_TYPES[5]
        #
        # # todo 5 - handle custom visualizations
        # # # inserting custom visualization
        # # visDict = CompuCellSetup.customVisStorage.visDataDict
        # #
        # # for visName in visDict:
        # #     self.fieldTypes[visName] = FIELD_TYPES[6]

    def showDisplayWidgets(self):
        '''
        Displays initial snapthos widgets - called from showSimView
        :return:None
        '''

        # This block of code simply checks to see if some plugins assoc'd with Vis are defined
        # todo 5 - rework this - remove parsing away from the player
        # from cc3d.core import XMLUtils

        cc3d_xml_2_obj_converter = CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter
        if cc3d_xml_2_obj_converter != None:
            self.pluginCOMDefined = False
            self.pluginFPPDefined = False

            self.root_element = cc3d_xml_2_obj_converter.root
            elms = self.root_element.getElements("Plugin")
            elmList = XMLUtils.CC3DXMLListPy(elms)
            for elm in elmList:
                pluginName = elm.getAttribute("Name")
                print("   pluginName = ", pluginName)  # e.g. CellType, Contact, etc
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

        # todo 5 -  rework initialization
        self.mainGraphicsWidget.setStatusBar(self.__statusBar)

        self.mainGraphicsWidget.setZoomItems(self.zitems)  # Set zoomFixed parameters

        # if self.borderAct.isChecked():  # Vis menu "Cell Borders" check box
        #     self.mainGraphicsWidget.showBorder()
        # else:
        #     self.mainGraphicsWidget.hideBorder()
        #
        # if self.clusterBorderAct.isChecked():  # Vis menu "Cluster Borders" check box
        #     self.mainGraphicsWidget.showClusterBorder()
        #
        # # ---------------------
        # if self.cellGlyphsAct.isChecked():  # Vis menu "Cell Glyphs"
        #     self.mainGraphicsWidget.showCellGlyphs()
        #
        # # ---------------------
        # if self.FPPLinksAct.isChecked():  # Vis menu "FPP (Focal Point Plasticity) Links"
        #     self.mainGraphicsWidget.showFPPLinks()

        self.mainGraphicsWidget.setPlane(PLANES[0], 0)
        # todo - old code
        # self.mainGraphicsWidget.currentDrawingObject.setPlane(PLANES[0], 0)

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


        # todo 5 - write code that create screenshot outoput if parameters are changed

        # if (
        #         self.__imageOutput or self.__latticeOutputFlag) and self.mysim:  # has user requested output and is there a valid sim?
        #     if self.screenshotDirectoryName == "":  # haven't created any yet
        #         #                print MODULENAME, '__paramsChanged(), screenshotDirName empty;  calling createOutputDirs'
        #         self.createOutputDirs()
        #     elif self.prevOutputDir != self.__outputDirectory:  # test if the sneaky user changed the output location
        #         #                print MODULENAME, '__paramsChanged(),  prevOutput != Output;  calling createOutputDirs'
        #         self.createOutputDirs()
        #
        #         # NOTE: if self.mysim == None (i.e. sim hasn't begun yet), then createOutputDirs() should be called in __loadSim

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
        if self.__sim_file_name != "":
            file = QFile(self.__sim_file_name)
            if file is not None:
                if self.mainGraphicsWidget is None:

                    self.showSimView(file)

                else:

                    # self.__closeSim()
                    # print 'BEFORE showSimView'
                    self.showSimView(file)
                    # print 'AFTER showSimView'

        self.drawingAreaPrepared = True
        self.update_active_window_vis_flags()  # needed in case switching from one sim to another (e.g. 1st has FPP, 2nd doesn't)

    def set_title_window_from_sim_fname(self, widget, abs_sim_fname):
        """
        Sets window title based on current simulation full name
        :param widget: {QWidget}
        :param abs_sim_fname: {str} absolute simulation fname
        :return: None
        """
        title_to_display = join(basename(dirname(self.__sim_file_name)), basename(self.__sim_file_name))
        widget.setWindowTitle(title_to_display + " - CompuCell3D Player")

    def __openLDSFile(self, fileName=None):
        '''
        Opens Lattice Description File - for vtk replay mode
        :param fileName: str - .dml file name
        :return:none
        '''
        filter = "Lattice Description Summary file  (*.dml )"  # self._getOpenFileFilter()

        defaultDir = self.__outputDirectory

        if not os.path.exists(defaultDir):
            defaultDir = os.getcwd()

        self.fileName_tuple = QFileDialog.getOpenFileName(
            self.ui,
            QApplication.translate('ViewManager', "Open Lattice Description Summary file"),
            defaultDir,
            filter
        )

        self.__sim_file_name = self.fileName_tuple[0]

        self.__sim_file_name = os.path.abspath(str(self.__sim_file_name))

        # setting text for main window (self.__parent) title bar
        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=self.__sim_file_name)

        Configuration.setSetting("ImageOutputOn", False)
        Configuration.setSetting("LatticeOutputOn", False)

    def open_recent_sim(self) -> None:
        """
        Slot - opens recent simulation
        :return: None
        """
        if self.simulationIsRunning:
            return

        action = self.sender()
        if isinstance(action, QAction):
            self.__sim_file_name = str(action.data())

        # # opening screenshot description file
        # self.open_implicit_screenshot_descr_file()

        # setting text for main window (self.__parent) title bar
        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=self.__sim_file_name)

        persistent_globals = CompuCellSetup.persistent_globals

        self.__sim_file_name = os.path.abspath(self.__sim_file_name)
        persistent_globals.simulation_file_name = os.path.abspath(self.__sim_file_name)

        Configuration.setSetting("RecentFile", self.__sim_file_name)
        #  each loaded simulation has to be passed to a function which updates list of recent files
        Configuration.setSetting("RecentSimulations", self.__sim_file_name)

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

        defaultDir = str(Configuration.getSetting('ProjectLocation'))

        if not os.path.exists(defaultDir):
            defaultDir = os.getcwd()

        self.__sim_file_name = QFileDialog.getOpenFileName( \
            self.ui,
            QApplication.translate('ViewManager', "Open Simulation File"),
            defaultDir,
            filter
        )
        # getOpenFilename may return tuple
        if isinstance(self.__sim_file_name, tuple):
            self.__sim_file_name = self.__sim_file_name[0]

        # converting Qstring to python string and normalizing path
        self.__sim_file_name = os.path.abspath(str(self.__sim_file_name))

        if os.path.splitext(self.__sim_file_name)[1].lower() not in ['.cc3d', '.dml']:
            print('Not a .cc3d of .dml file. Ignoring ')
            self.__sim_file_name = ''
            return

        print('__openSim: self.__fileName=', self.__sim_file_name)

        # setting text for main window (self.__parent) title bar
        self.set_title_window_from_sim_fname(widget=self.__parent, abs_sim_fname=self.__sim_file_name)

        CompuCellSetup.persistent_globals.simulation_file_name = self.__sim_file_name

        # Add the current opening file to recent files and recent simulation
        Configuration.setSetting("RecentFile", self.__sim_file_name)

        # each loaded simulation has to be passed to a function which updates list of recent files
        Configuration.setSetting("RecentSimulations", self.__sim_file_name)

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
        self.update_active_window_vis_flags()
        if self.cellsAct.isEnabled():
            Configuration.setSetting('CellsOn', checked)

            # MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()

                graphicsWidget.draw(basic_simulation_data=self.basicSimulationData)

                self.update_active_window_vis_flags(graphicsWidget)

        self.simulation.drawMutex.unlock()

    def __checkBorder(self, checked):
        '''
        Slot that triggers display of borders
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.update_active_window_vis_flags()

        if self.borderAct.isEnabled():
            Configuration.setSetting('CellBordersOn', checked)

            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()

                graphicsWidget.draw(basic_simulation_data=self.basicSimulationData)

                self.update_active_window_vis_flags(graphicsWidget)

        self.simulation.drawMutex.unlock()

    def __checkClusterBorder(self, checked):
        '''
        Slot that triggers display of cluster borders
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()

        self.update_active_window_vis_flags()
        if self.clusterBorderAct.isEnabled():
            Configuration.setSetting('ClusterBordersOn', checked)

            # MDIFIX
            for winId, win in self.win_inventory.getWindowsItems(GRAPHICS_WINDOW_LABEL):
                graphicsWidget = win.widget()
                graphicsWidget.draw(basic_simulation_data=self.basicSimulationData)
                self.update_active_window_vis_flags(graphicsWidget)

        self.simulation.drawMutex.unlock()

    def __checkCellGlyphs(self, checked):
        '''
        Slot that triggers display of cell glyphs
        :param checked: bool - flag determines if action is on or off
        :return:None
        '''
        # Should be disabled when the simulation is not loaded!
        self.simulation.drawMutex.lock()
        self.update_active_window_vis_flags()

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
                graphicsWidget.draw(basic_simulation_data=self.basicSimulationData)

                self.update_active_window_vis_flags(graphicsWidget)

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
        self.update_active_window_vis_flags()

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
                graphicsWidget.draw(basic_simulation_data=self.basicSimulationData)
                self.update_active_window_vis_flags(graphicsWidget)

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
        self.update_active_window_vis_flags()

        if self.FPPLinksColorAct.isEnabled():

            # if self.lastActiveWindow is not None:
            # MDIFIX
            if self.lastActiveRealWindow is not None:
                # Check for FPP plugin - improve to not even allow glyphs if no CoM
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

                except AttributeError as e:
                    pass

                self.update_active_window_vis_flags(graphicsWidget)

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

                except AttributeError as e:
                    pass

                self.update_active_window_vis_flags(graphicsWidget)

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
            fieldName = list(self.fieldTypes.keys())[idx]
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

        fullSimFileName = os.path.abspath(self.__sim_file_name)
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

        fullSimFileName = os.path.abspath(self.__sim_file_name)
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
        self.simulation.generate_pif_from_vtk(self.simulation.currentFileName, pifFileName)

    def __configsChanged(self):
        """
        Private slot to handle a change of the preferences. Called after we hit Ok builtin on configuration dialog
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
