# todo - check simulation replay
# todo - resolve the issue of imports in the CompuCellSetup

from weakref import ref
import cc3d.player5.DefaultData as DefaultData
import cc3d.player5.Configuration as Configuration
import cc3d
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.core.GraphicsUtils.GraphicsFrame import GraphicsFrame
from .GraphicsWindowData import GraphicsWindowData
import cc3d.CompuCellSetup
import sys

platform = sys.platform
if platform == 'darwin':
    from cc3d.player5.Utilities.QVTKRenderWindowInteractor_mac import QVTKRenderWindowInteractor
else:
    from cc3d.player5.Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

MODULENAME = '---- GraphicsFrameWidget.py: '


class QGraphicsFrame(GraphicsFrame):
    """
    :class:`GraphicsFrame` implementation for vtk Qt
    """

    def __init__(self, parent: QtWidgets.QWidget, stv):
        """

        :param parent: parent widget
        :type parent: QtWidgets.QWidget
        :param stv: tab view containing directing actions
        :type stv: SimpleTabView
        """

        self.parent = ref(parent)
        self.stv = ref(stv) if stv is not None else stv

        pg = cc3d.CompuCellSetup.persistent_globals

        generic_drawer = GenericDrawer(boundary_strategy=pg.simulator.getBoundaryStrategy() if pg.simulator else None)
        generic_drawer.set_pixelized_cartesian_scene(pg.configuration.getSetting("PixelizedCartesianFields"))

        try:
            generic_drawer.set_field_extractor(field_extractor=pg.persistent_holder['field_extractor'])
        except KeyError:
            pass

        super().__init__(generic_drawer=GenericDrawer(pg.simulator.getBoundaryStrategy() if pg.simulator else None),
                         current_bsd=pg.screenshot_manager.bsd if pg.screenshot_manager is not None else None)

    def get_vtk_window(self):

        vtk_widget = QVTKRenderWindowInteractor(self.parent())

        vtk_widget.Initialize()
        vtk_widget.Start()

        return vtk_widget.GetRenderWindow(), vtk_widget

    def store_gui_vis_config(self, scr_data):
        if self.stv is None:
            return

        stv = self.stv()

        scr_data.cell_borders_on = stv.border_act.isChecked()
        scr_data.cells_on = stv.cells_act.isChecked()
        scr_data.cluster_borders_on = stv.cluster_border_act.isChecked()
        scr_data.cell_glyphs_on = stv.cell_glyphs_act.isChecked()
        scr_data.fpp_links_on = stv.fpp_links_act.isChecked()
        scr_data.lattice_axes_on = Configuration.getSetting('ShowHorizontalAxesLabels') or Configuration.getSetting(
            'ShowVerticalAxesLabels')
        scr_data.lattice_axes_labels_on = Configuration.getSetting("ShowAxes")
        scr_data.bounding_box_on = Configuration.getSetting("BoundingBoxOn")

        invisible_types = Configuration.getSetting("Types3DInvisible")
        invisible_types = invisible_types.strip()

        if invisible_types:
            scr_data.invisible_types = list([int(x) for x in invisible_types.split(',')])
        else:
            scr_data.invisible_types = []

    def zoom_in(self):
        self.vtkWidget.zoomIn()

    def zoom_out(self):
        self.vtkWidget.zoomOut()

    def set_drawing_style(self, _style):
        super().set_drawing_style(_style)

        style = _style.upper()
        if style == "2D":
            self.vtkWidget.setMouseInteractionSchemeTo2D()
        elif style == "3D":
            self.vtkWidget.setMouseInteractionSchemeTo3D()

    def close(self):
        self.vtkWidget.close()
        super().close()


class GraphicsFrameWidget(QtWidgets.QFrame):
    def __init__(self, parent=None, originatingWidget=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.proj_sb_act = QtWidgets.QAction(self)
        self.is_screenshot_widget = False
        self.qvtkWidget = QGraphicsFrame(self, originatingWidget)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # MDIFIX
        self.parentWidget = ref(originatingWidget)

        self.status_bar = None

        self.proj_combo_box_act = None
        self.proj_combo_box = None
        self.proj_spin_box = None
        self.field_combo_box_act = None
        self.field_combo_box = None
        self.screenshot_act = None

        self.lineEdit = QtWidgets.QLineEdit()

        self.init_cross_section_actions()
        self.cstb = self.init_cross_section_toolbar()

        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        layout.addWidget(self.cstb)
        layout.addWidget(self.qvtkWidget.vtkWidget)
        self.setLayout(layout)
        self.setMinimumSize(100, 100)  # needs to be defined to resize smaller than 400x400
        self.resize(600, 600)

    @property
    def plane(self):
        return self.qvtkWidget.plane

    @plane.setter
    def plane(self, _plane):
        self.qvtkWidget.plane = _plane

    @property
    def planePos(self):
        return self.qvtkWidget.planePos

    @planePos.setter
    def planePos(self, _planePos):
        self.qvtkWidget.planePos = _planePos

    @property
    def currentProjection(self):
        return self.qvtkWidget.currentProjection

    @currentProjection.setter
    def currentProjection(self, _currentProjection):
        self.qvtkWidget.currentProjection = _currentProjection

    @property
    def xyPlane(self):
        return self.qvtkWidget.xyPlane

    @xyPlane.setter
    def xyPlane(self, _xyPlane):
        self.qvtkWidget.xyPlane = _xyPlane

    @property
    def xzPlane(self):
        return self.qvtkWidget.xzPlane

    @xzPlane.setter
    def xzPlane(self, _xzPlane):
        self.qvtkWidget.xzPlane = _xzPlane

    @property
    def yzPlane(self):
        return self.qvtkWidget.yzPlane

    @yzPlane.setter
    def yzPlane(self, _yzPlane):
        self.qvtkWidget.yzPlane = _yzPlane

    @property
    def xyMaxPlane(self):
        return self.qvtkWidget.xyMaxPlane

    @xyMaxPlane.setter
    def xyMaxPlane(self, _xyMaxPlane):
        self.qvtkWidget.xyMaxPlane = _xyMaxPlane

    @property
    def xzMaxPlane(self):
        return self.qvtkWidget.xzMaxPlane

    @xzMaxPlane.setter
    def xzMaxPlane(self, _xzMaxPlane):
        self.qvtkWidget.xzMaxPlane = _xzMaxPlane
    @property
    def yzMaxPlane(self):
        return self.qvtkWidget.yzMaxPlane

    @yzMaxPlane.setter
    def yzMaxPlane(self, _yzMaxPlane):
        self.qvtkWidget.yzMaxPlane = _yzMaxPlane

    @property
    def draw3DFlag(self):
        return self.qvtkWidget.draw3DFlag

    @draw3DFlag.setter
    def draw3DFlag(self, _draw3DFlag):
        self.qvtkWidget.draw3DFlag = _draw3DFlag

    @property
    def fieldTypes(self):
        return self.qvtkWidget.fieldTypes

    @fieldTypes.setter
    def fieldTypes(self, _fieldTypes):
        self.qvtkWidget.fieldTypes = _fieldTypes

    @property
    def gd(self):
        return self.qvtkWidget.gd

    @gd.setter
    def gd(self, _gd):
        self.qvtkWidget.gd = _gd

    @property
    def current_screenshot_data(self):
        return self.qvtkWidget.current_screenshot_data

    @current_screenshot_data.setter
    def current_screenshot_data(self, _current_screenshot_data):
        self.qvtkWidget.current_screenshot_data = _current_screenshot_data

    @property
    def current_bsd(self):
        return self.qvtkWidget.current_bsd

    @current_bsd.setter
    def current_bsd(self, _current_bsd):
        self.qvtkWidget.current_bsd = _current_bsd

    @property
    def camera2D(self):
        return self.qvtkWidget.camera2D

    @camera2D.setter
    def camera2D(self, _camera2D):
        self.qvtkWidget.camera2D = _camera2D

    @property
    def camera3D(self):
        return self.qvtkWidget.camera3D

    @camera3D.setter
    def camera3D(self, _camera3D):
        self.qvtkWidget.camera3D = _camera3D

    @property
    def renWin(self):
        return self.qvtkWidget.renWin

    @renWin.setter
    def renWin(self, _renWin):
        self.qvtkWidget.renWin = _renWin

    def initialize_scene(self):
        """
        initialization function that sets up field extractor for the generic Drawer
        :return:
        """
        tvw = self.parentWidget()
        self.current_screenshot_data = self.compute_current_screenshot_data()

        self.qvtkWidget.init_field_types()
        self.qvtkWidget.initialize_extractor(field_extractor=tvw.fieldExtractor)

    def get_current_field_name_and_type(self):
        """
        Returns current field name and type
        :return: {tuple}
        """

        return self.qvtkWidget.get_current_field_name_and_type()

    def compute_current_screenshot_data(self):
        """
        Computes/populates Screenshot Description data based ont he current GUI configuration
        for the current window
        :return: {screenshotData}
        """

        return self.qvtkWidget.compute_current_screenshot_data()

    def store_gui_vis_config(self, scr_data):
        """
        Stores visualization settings such as cell borders, on/or cell on/off etc...

        :param scr_data: {instance of ScreenshotDescriptionData}
        :return: None
        """

        return self.qvtkWidget.store_gui_vis_config(scr_data=scr_data)

    def render_repaint(self):
        """

        :return:
        """
        self.qvtkWidget.Render()

    def draw(self, basic_simulation_data):
        """
        Main drawing function - calls ok_to_draw fcn from the GenericDrawer. All drawing happens there
        :param basic_simulation_data: {instance of BasicSimulationData}
        :return: None
        """

        self.qvtkWidget.draw(basic_simulation_data=basic_simulation_data)

    def set_status_bar(self, status_bar):
        """
        Sets status bar
        :param status_bar:
        :return:
        """
        self.status_bar = status_bar

    @pyqtSlot()
    def configs_changed(self):
        """

        :return:
        """
        # handling what happens after user presses stop - at this point pg is reset and no drawing should be allowed
        # for some reason on OSX we cannot fo
        # from cc3d import CompuCellSetup and  instead we access persistent globals via cc3d.CompuCellSetup
        pg = cc3d.CompuCellSetup.persistent_globals
        if pg.view_manager is None:
            return

        # here we are updating models based on the new set of configs
        self.gd.configsChanged()

    def Render(self):

        self.qvtkWidget.Render()

    def get_camera(self):
        """
        Conveenince function that fetches active camera obj
        :return: {vtkCamera}
        """
        return self.qvtkWidget.get_active_camera()

    def get_active_camera(self):

        return self.qvtkWidget.get_active_camera()

    def get_camera_3D(self):
        """
        Convenince function that fetches 3D camera obj
        :return: {vtkCamera}
        """
        return self.qvtkWidget.camera3D

    def get_camera_2D(self):
        return self.qvtkWidget.camera2D

    def setZoomItems(self, _zitems):
        """

        :param _zitems:
        :return:
        """

    def set_plane(self, plane, pos):
        self.qvtkWidget.set_plane(plane, pos)

    def get_plane(self):
        """
        Gets current plane tuple
        :return: {tuple} (plane label, plane position)
        """
        return self.qvtkWidget.get_plane()

    def init_cross_section_toolbar(self):
        """
        Initializes cross section toolbar
        :return: None
        """

        cstb = QtWidgets.QToolBar("CrossSection", self)
        cstb.setObjectName("CrossSection")
        cstb.setToolTip("Projection")

        cstb.addWidget(self.proj_combo_box)
        cstb.addWidget(self.proj_spin_box)

        cstb.addWidget(self.field_combo_box)
        cstb.addAction(self.screenshot_act)

        return cstb

    def init_cross_section_actions(self):
        """
        Initializes actions associated with the cross section toolbar
        :return: None
        """

        self.proj_combo_box_act = QtWidgets.QAction(self)
        self.proj_combo_box = QtWidgets.QComboBox()
        self.proj_combo_box.addAction(self.proj_combo_box_act)

        # NB: the order of these is important; rf. setInitialCrossSection where we set 'xy' to be default projection
        self.proj_combo_box.addItem("3D")
        self.proj_combo_box.addItem("xy")
        self.proj_combo_box.addItem("xz")
        self.proj_combo_box.addItem("yz")

        self.proj_spin_box = QtWidgets.QSpinBox()
        self.proj_spin_box.addAction(self.proj_sb_act)

        self.field_combo_box_act = QtWidgets.QAction(self)
        # Note that this is different than the fieldComboBox in the Prefs panel (rf. SimpleTabView.py)
        self.field_combo_box = QtWidgets.QComboBox()
        self.field_combo_box.addAction(self.field_combo_box_act)

        gip = DefaultData.getIconPath
        self.screenshot_act = QtWidgets.QAction(QtGui.QIcon(gip("screenshot.png")), "&Take Screenshot", self)

    def proj_combo_box_changed(self):
        """
        slot reacting to changes in the projection combo box
        :return: None
        """

        tvw = self.parentWidget()

        if tvw.completedFirstMCS:
            tvw.newDrawingUserRequest = True

        self.qvtkWidget.currentProjection = str(self.proj_combo_box.currentText())
        self.qvtkWidget.projection_position = int(self.proj_spin_box.value())

        if self.qvtkWidget.currentProjection == '3D':
            # disable spinbox
            self.proj_spin_box.setEnabled(False)
            self.set_drawing_style("3D")
            if tvw.completedFirstMCS:
                tvw.newDrawingUserRequest = True

        elif self.qvtkWidget.currentProjection == 'xy':
            self.update_projection_spin_box(spin_box_value=self.qvtkWidget.xyPlane)

        elif self.qvtkWidget.currentProjection == 'xz':
            self.update_projection_spin_box(spin_box_value=self.qvtkWidget.xzPlane)

        elif self.qvtkWidget.currentProjection == 'yz':
            self.update_projection_spin_box(spin_box_value=self.qvtkWidget.yzPlane)

        self.qvtkWidget.current_screenshot_data = self.qvtkWidget.compute_current_screenshot_data()

        # SimpleTabView.py
        tvw._drawField()

    def update_projection_spin_box(self, spin_box_value: int):
        """
        updates projection spin box without causing redundant redraws
        :param spin_box_value:
        :return:
        """

        self.proj_spin_box.valueChanged.disconnect(self.proj_spin_box_changed)
        self.proj_spin_box.setEnabled(True)
        self.proj_spin_box.setValue(spin_box_value)
        self.proj_spin_box_changed(spin_box_value, ok_to_draw=False)
        self.proj_spin_box.valueChanged.connect(self.proj_spin_box_changed)

    def proj_spin_box_changed(self, val, ok_to_draw=True):
        """
        Slot that reacts to position spin boox changes
        :param val: {int} number corresponding to the position of the crosse section
        :param ok_to_draw: flag indicating if it is OK to call draw function
        :return: None
        """

        tvw = self.parentWidget()

        if tvw.completedFirstMCS:
            tvw.newDrawingUserRequest = True

        self.set_drawing_style("2D")

        self.qvtkWidget.currentProjection = str(self.proj_combo_box.currentText())
        self.qvtkWidget.projection_position = int(self.proj_spin_box.value())

        if self.qvtkWidget.currentProjection == 'xy':
            if val > self.qvtkWidget.xyMaxPlane:
                val = self.qvtkWidget.xyMaxPlane
            self.proj_spin_box.setValue(val)
            # for some bizarre(!) reason, val=0 for xyPlane
            self.qvtkWidget.set_plane(self.qvtkWidget.currentProjection, val)
            self.qvtkWidget.xyPlane = val

        elif self.qvtkWidget.currentProjection == 'xz':
            if val > self.qvtkWidget.xzMaxPlane:
                val = self.qvtkWidget.xzMaxPlane
            self.proj_spin_box.setValue(val)
            self.qvtkWidget.set_plane(self.qvtkWidget.currentProjection, val)
            self.qvtkWidget.xzPlane = val

        elif self.qvtkWidget.currentProjection == 'yz':
            if val > self.qvtkWidget.yzMaxPlane:
                val = self.qvtkWidget.yzMaxPlane
            self.proj_spin_box.setValue(val)
            self.qvtkWidget.set_plane(self.qvtkWidget.currentProjection, val)
            self.qvtkWidget.yzPlane = val

        self.qvtkWidget.current_screenshot_data = self.qvtkWidget.compute_current_screenshot_data()
        # SimpleTabView.py
        if ok_to_draw:
            tvw._drawField()

    def field_type_changed(self):
        """
        Slot reacting to the field type combo box changes
        :return: None
        """

        tvw = self.parentWidget()
        if tvw.completedFirstMCS:
            tvw.newDrawingUserRequest = True
        field_name, field_type = self.get_current_field_name_and_type()

        tvw.setFieldType((field_name, field_type))
        self.qvtkWidget.field_name = self.field_combo_box.currentText()
        self.qvtkWidget.current_screenshot_data = self.qvtkWidget.compute_current_screenshot_data()

        tvw._drawField()

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization
        :param _style:{str} '2D' or '3D'
        :return: None
        """

        self.qvtkWidget.set_drawing_style(_style)

    def apply_3D_graphics_window_data(self, gwd):
        """
        Applies graphics window configuration data (stored on a disk) to graphics window (3D version)
        :param gwd: {instrance of GraphicsWindowData}
        :return: None
        """

        for p in range(self.proj_combo_box.count()):

            if str(self.proj_combo_box.itemText(p)) == '3D':
                self.proj_combo_box.setCurrentIndex(p)
                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D
                self.qvtkWidget.set_camera_from_graphics_window_data(camera=self.camera3D, gwd=gwd)
                break

    def apply_2D_graphics_window_data(self, gwd):
        """
        Applies graphics window configuration data (stored on a disk) to graphics window (2D version)
        :param gwd: {instrance of GraphicsWindowData}
        :return: None
        """

        for p in range(self.proj_combo_box.count()):

            if str(self.proj_combo_box.itemText(p)).lower() == str(gwd.planeName).lower():
                self.proj_combo_box.setCurrentIndex(p)
                # automatically invokes the callback (--Changed)
                self.proj_spin_box.setValue(gwd.planePosition)

                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 2D
                self.qvtkWidget.set_camera_from_graphics_window_data(camera=self.camera2D, gwd=gwd)

    def apply_graphics_window_data(self, gwd):
        """
        Applies graphics window configuration data (stored on a disk) to graphics window (2D version)
        :param gwd: {instrance of GraphicsWindowData}
        :return: None
        """

        for i in range(self.field_combo_box.count()):

            if str(self.field_combo_box.itemText(i)) == gwd.sceneName:

                self.field_combo_box.setCurrentIndex(i)
                # setting 2D projection or 3D
                if gwd.is3D:
                    self.apply_3D_graphics_window_data(gwd)
                else:
                    self.apply_2D_graphics_window_data(gwd)

                break

    def get_graphics_window_data(self) -> GraphicsWindowData:
        """
        returns instance of GraphicsWindowData for current widget
        :return:
        """

        return self.qvtkWidget.get_graphics_window_data()

    def add_screenshot_conf(self):
        """
        Adds screenshot configuration data for a current scene
        :return: None
        """
        tvw = self.parentWidget()
        print(MODULENAME, '  _takeShot():  self.renWin.GetSize()=', self.qvtkWidget.renWin.GetSize())

        if tvw.screenshotManager is not None:
            self.qvtkWidget.field_name = str(self.field_combo_box.currentText())
            self.qvtkWidget.add_screenshot_config(screenshot_manager=tvw.screenshotManager)

    def set_connects(self, _workspace):
        """
        connects signals and slots
        :param _workspace:
        :return:
        """
        # rf. Plugins/ViewManagerPlugins/SimpleTabView.py

        self.proj_combo_box.currentIndexChanged.connect(self.proj_combo_box_changed)
        self.proj_spin_box.valueChanged.connect(self.proj_spin_box_changed)

        self.field_combo_box.currentIndexChanged.connect(self.field_type_changed)

        self.screenshot_act.triggered.connect(self.add_screenshot_conf)

    def set_initial_cross_section(self, basic_simulation_data):

        field_dim = basic_simulation_data.fieldDim

        self.qvtkWidget.set_initial_cross_section(basic_simulation_data)

        # set to be 'xy' projection by default, regardless of 2D or 3D sim?
        self.proj_combo_box.setCurrentIndex(1)

        self.proj_spin_box.setMinimum(0)

        self.proj_spin_box.setMaximum(10000)

        # If you want to set the value from configuration
        self.proj_spin_box.setValue(field_dim.z / 2)

    def update_cross_section(self, basic_simulation_data):
        """
        :param basic_simulation_data:
        :return:
        """

        self.qvtkWidget.set_initial_cross_section(basic_simulation_data)

    def update_field_types_combo_box(self, field_types: dict) -> None:
        """
        Updates combo boxes
        :param field_types:{str}
        :return:
        """

        # assign field types to be the same as field types in the workspace
        self.qvtkWidget.fieldTypes = field_types

        cb = self.field_combo_box
        current_text = None
        if cb.count():
            current_text = self.field_combo_box.currentText()

        cb.clear()
        cb.addItem("Cell_Field")
        for key in list(self.qvtkWidget.fieldTypes.keys()):
            if key != "Cell_Field":
                cb.addItem(key)

        # setting value of the Combo box to be cellField - default action
        if current_text is None:
            cb.setCurrentIndex(0)
        else:
            current_idx = cb.findText(current_text)
            cb.setCurrentIndex(current_idx)

    def set_field_types_combo_box(self, field_types):

        self.update_field_types_combo_box(field_types=field_types)

        # essential initialization of the default cameras
        # last call triggers fisrt call to ok_to_draw function so we here reset camera so that
        # all the actors are initially visible
        self.qvtkWidget.reset_camera()
        self.qvtkWidget.copy_camera(src=self.qvtkWidget.camera2D, dst=self.qvtkWidget.camera3D)

    def reset_all_cameras(self, bsd):
        self.qvtkWidget.reset_all_cameras(bsd)

    def zoom_in(self):
        """
        Zooms in view
        :return:
        """
        self.qvtkWidget.zoom_in()

    def zoom_out(self):
        """
        Zooms out view
        :return:
        """

        self.qvtkWidget.zoom_out()

    def reset_camera(self):
        """
        Resets camera to default settings
        :return:
        """
        self.qvtkWidget.vtkWidget.resetCamera()

    # note that if you close widget using X button this slot is not called
    # we need to reimplement closeEvent
    def closeEvent(self, ev):
        """

        :param ev:
        :return:
        """
        print('CHANGE and update closeEvent')
        # cleaning up to release memory - notice that if we do not do this cleanup this widget
        # will not be destroyed and will take sizeable portion of the memory
        # not a big deal for a single simulation but repeated runs can easily exhaust all system memory

        self.qvtkWidget.close()
        self.qvtkWidget = None
