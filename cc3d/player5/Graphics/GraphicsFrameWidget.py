# todo - check simulation replay
# todo - resolve the issue of imports in the CompuCellSetup

from gc import collect
from weakref import ref
import cc3d.player5.DefaultData as DefaultData
import cc3d.player5.Configuration as Configuration
import cc3d
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from cc3d.core.enums import *
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from .GraphicsWindowData import GraphicsWindowData
from cc3d.core.GraphicsUtils.ScreenshotData import ScreenshotData
import cc3d.CompuCellSetup
from cc3d.player5.Utilities import qcolor_to_rgba, cs_string_to_typed_list
import sys
import typing
from collections import OrderedDict
from PyQt5.QtWidgets import QApplication

platform = sys.platform
if platform == 'darwin':
    from cc3d.player5.Utilities.QVTKRenderWindowInteractor_mac import QVTKRenderWindowInteractor
else:
    from cc3d.player5.Utilities.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

MODULENAME = '---- GraphicsFrameWidget.py: '


class GraphicsFrameWidget(QtWidgets.QFrame):
    def __init__(self, parent=None, originatingWidget=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.proj_sb_act = QtWidgets.QAction(self)
        self.is_screenshot_widget = False
        self.qvtkWidget = QVTKRenderWindowInteractor(self)  # a QWidget

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.vertical_resolution = QApplication.desktop().screenGeometry().height()

        # MDIFIX
        self.parentWidget = ref(originatingWidget)

        self.plane = None
        self.planePos = None

        self.status_bar = None

        self.proj_combo_box_act = None
        self.proj_combo_box = None
        self.proj_spin_box = None
        self.field_combo_box_act = None
        self.field_combo_box = None
        self.screenshot_act = None

        self.currentProjection = None
        self.xyPlane = None
        self.xzPlane = None
        self.yzPlane = None
        self.xyMaxPlane = None
        self.xzMaxPlane = None
        self.yzMaxPlane = None

        self.draw3DFlag = False

        self.fieldTypes = None

        self.lineEdit = QtWidgets.QLineEdit()

        self.init_cross_section_actions()
        self.cstb = self.init_cross_section_toolbar()

        layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.TopToBottom)
        layout.addWidget(self.cstb)
        layout.addWidget(self.qvtkWidget)
        self.setLayout(layout)
        self.setMinimumSize(100, 100)  # needs to be defined to resize smaller than 400x400
        self.resize(600, 600)

        self.qvtkWidget.Initialize()
        self.qvtkWidget.Start()

        # todo 5 - adding generic drawer

        sim = cc3d.CompuCellSetup.persistent_globals.simulator
        boundary_strategy = None
        if sim:
            boundary_strategy = sim.getBoundaryStrategy()

        self.gd = GenericDrawer(boundary_strategy=boundary_strategy)
        self.gd.set_interactive_camera_flag(True)
        self.gd.set_pixelized_cartesian_scene(Configuration.getSetting("PixelizedCartesianFields"))
        self.gd.set_vertical_resolution(vertical_resoultion=self.vertical_resolution)

        # placeholder for current screenshot data
        self.current_screenshot_data = None

        # placeholder for currently used basic simulation data
        self.current_bsd = None

        self.camera2D = self.gd.get_active_camera()
        self.camera3D = self.gd.get_renderer().MakeCamera()

        self.renWin = self.qvtkWidget.GetRenderWindow()
        self.renWin.AddRenderer(self.gd.get_renderer())

    @property
    def metadata_fetcher_dict(self) -> typing.Dict[str, typing.Callable[[str, str], dict]]:
        """
        Map from field type to field metadata getter

        :return: map
        :rtype: typing.Dict[str, typing.Callable[[str, str], dict]]
        """
        return {
            'CellField': self.get_cell_field_metadata,
            'ConField': self.get_con_field_metadata,
            'ScalarField': self.get_con_field_metadata,
            'ScalarFieldCellLevel': self.get_con_field_metadata,
            'VectorField': self.get_vector_field_metadata,
            'VectorFieldCellLevel': self.get_vector_field_metadata,
        }

    def copy_camera(self, src, dst):
        """
        Copies camera settings

        :param src: 
        :param dst: 
        :return: None
        """
        dst.SetClippingRange(src.GetClippingRange())
        dst.SetFocalPoint(src.GetFocalPoint())
        dst.SetPosition(src.GetPosition())
        dst.SetViewUp(src.GetViewUp())

    def get_metadata(self, field_name, field_type):
        """
        Fetches a dictionary that summarizes graphics/configs settings for the current scene

        :param field_name: {str} field name
        :param field_type: {str} field type
        :return: {dict}
        """
        try:
            metadata_fetcher_fcn = self.metadata_fetcher_dict[field_type]
        except KeyError:
            return {}

        metadata = metadata_fetcher_fcn(field_name=field_name, field_type=field_type)

        return metadata

    def get_cell_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to cell field

        :param field_name:{str} field_name
        :param field_type: {str} field type
        :return: {dict}
        """

        metadata_dict = self.get_config_metadata(field_name=field_name, field_type=field_type)
        return metadata_dict

    def get_config_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render borders

        :param field_name:{str} field_name
        :param field_type: {str} field type
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
        metadata_dict['FPPLinksColor'] = qcolor_to_rgba(Configuration.getSetting('FPPLinksColor'))

        metadata_dict['ShowHorizontalAxesLabels'] = Configuration.getSetting('ShowHorizontalAxesLabels')
        metadata_dict['ShowVerticalAxesLabels'] = Configuration.getSetting('ShowVerticalAxesLabels')

        # type-color map
        type_color_map_dict = OrderedDict()
        config_type_color_map = Configuration.getSetting("TypeColorMap")
        for type_id, qt_color in list(config_type_color_map.items()):
            type_color_map_dict[type_id] = qcolor_to_rgba(qt_color)

        metadata_dict['TypeColorMap'] = type_color_map_dict

        return metadata_dict

    def get_con_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene

        :param field_name:{str} field_name
        :param field_type: {str} field type
        :return: {dict}
        """

        # metadata_dict = {}
        metadata_dict = self.get_config_metadata(field_name=field_name, field_type=field_type)
        con_field_name = field_name
        metadata_dict['MinRangeFixed'] = Configuration.getSetting("MinRangeFixed", con_field_name)
        metadata_dict['MaxRangeFixed'] = Configuration.getSetting("MaxRangeFixed", con_field_name)
        metadata_dict['MinRange'] = Configuration.getSetting("MinRange", con_field_name)
        metadata_dict['MaxRange'] = Configuration.getSetting("MaxRange", con_field_name)
        metadata_dict['ContoursOn'] = Configuration.getSetting("ContoursOn", con_field_name)
        metadata_dict['NumberOfContourLines'] = Configuration.getSetting("NumberOfContourLines", field_name)
        metadata_dict['ScalarIsoValues'] = cs_string_to_typed_list(
            Configuration.getSetting("ScalarIsoValues", field_name))
        metadata_dict['LegendEnable'] = Configuration.getSetting("LegendEnable", field_name)
        metadata_dict['DisplayMinMaxInfo'] = Configuration.getSetting("DisplayMinMaxInfo")

        return metadata_dict

    def get_vector_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene

        :param field_name:{str} field_name
        :param field_type: {str} field type
        :return: {dict}
        """

        metadata_dict = self.get_con_field_metadata(field_name=field_name, field_type=field_type)
        metadata_dict['ArrowLength'] = Configuration.getSetting('ArrowLength', field_name)
        metadata_dict['FixedArrowColorOn'] = Configuration.getSetting('FixedArrowColorOn', field_name)
        metadata_dict['ArrowColor'] = qcolor_to_rgba(Configuration.getSetting('ArrowColor', field_name))
        metadata_dict['ScaleArrowsOn'] = Configuration.getSetting('ScaleArrowsOn', field_name)

        return metadata_dict

    def initialize_scene(self):
        """
        initialization function that sets up field extractor for the generic Drawer

        :return:
        """
        tvw = self.parentWidget()
        self.current_screenshot_data = self.compute_current_screenshot_data()

        self.gd.set_field_extractor(field_extractor=tvw.fieldExtractor)

    def get_current_field_name_and_type(self):
        """
        Returns current field name and type

        :return: {tuple}
        """
        tvw = self.parentWidget()
        field_name = str(self.field_combo_box.currentText())

        try:
            field_type = tvw.fieldTypes[field_name]
        except KeyError:
            return field_name, ''
            # raise KeyError('Could not figure out the type of the field {} you requested'.format(field_name))

        return field_name, field_type

    def compute_current_screenshot_data(self):
        """
        Computes/populates Screenshot Description data based ont he current GUI configuration
        for the current window

        :return: {screenshotData}
        """

        scr_data = ScreenshotData()
        self.store_gui_vis_config(scr_data=scr_data)

        projection_name = str(self.proj_combo_box.currentText())
        projection_position = int(self.proj_spin_box.value())

        field_name, field_type = self.get_current_field_name_and_type()
        scr_data.plotData = (field_name, field_type)

        metadata = self.get_metadata(field_name=scr_data.plotData[0], field_type=scr_data.plotData[1])

        if projection_name == '3D':
            scr_data.spaceDimension = '3D'
        else:
            scr_data.spaceDimension = '2D'
            scr_data.projection = projection_name
            scr_data.projectionPosition = projection_position

        scr_data.metadata = metadata

        return scr_data

    def store_gui_vis_config(self, scr_data):
        """
        Stores visualization settings such as cell borders, on/or cell on/off etc...

        :param scr_data: {instance of ScreenshotDescriptionData}
        :return: None
        """
        tvw = self.parentWidget()

        scr_data.cell_borders_on = tvw.border_act.isChecked()
        scr_data.cells_on = tvw.cells_act.isChecked()
        scr_data.cluster_borders_on = tvw.cluster_border_act.isChecked()
        scr_data.cell_glyphs_on = tvw.cell_glyphs_act.isChecked()
        scr_data.fpp_links_on = tvw.fpp_links_act.isChecked()
        scr_data.lattice_axes_on = Configuration.getSetting('ShowHorizontalAxesLabels') or Configuration.getSetting(
            'ShowVerticalAxesLabels')
        scr_data.lattice_axes_labels_on = Configuration.getSetting("ShowAxes")
        scr_data.bounding_box_on = Configuration.getSetting("BoundingBoxOn")

        invisible_types = Configuration.getSetting("Types3DInvisible")
        invisible_types = invisible_types.strip()

        if invisible_types:
            scr_data.invisible_types = list([int(x) for x in invisible_types.split(',')])
            if 0 not in scr_data.invisible_types:
                scr_data.invisible_types = [0] + scr_data.invisible_types
        else:
            scr_data.invisible_types = [0]

    def render_repaint(self):
        """

        :return:
        """
        self.Render()

    def draw(self, basic_simulation_data):
        """
        Main drawing function - calls ok_to_draw fcn from the GenericDrawer. All drawing happens there

        :param basic_simulation_data: {instance of BasicSimulationData}
        :return: None
        """
        self.current_bsd = basic_simulation_data

        if self.current_screenshot_data is None:
            self.initialize_scene()

        self.gd.clear_display()

        self.current_screenshot_data = self.compute_current_screenshot_data()

        self.gd.draw(screenshot_data=self.current_screenshot_data, bsd=basic_simulation_data, screenshot_name='')

        # this call seems to be needed to refresh qvtk widget
        self.gd.get_renderer().ResetCameraClippingRange()
        # essential call to refresh screen . otherwise need to move/resize graphics window
        self.Render()

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
        color = Configuration.getSetting("WindowColor")
        self.gd.get_renderer().SetBackground(float(color.red()) / 255, float(color.green()) / 255,
                                             float(color.blue()) / 255)
        self.qvtkWidget.Render()

    def get_camera(self):
        """
        Conveenince function that fetches active camera obj

        :return: {vtkCamera}
        """
        return self.get_active_camera()

    def get_active_camera(self):

        return self.gd.get_active_camera()

    def get_camera_3D(self):
        """
        Convenince function that fetches 3D camera obj

        :return: {vtkCamera}
        """
        return self.camera3D

    def get_camera_2D(self):
        return self.camera2D

    def setZoomItems(self, _zitems):
        """

        :param _zitems:
        :return:
        """

    def set_plane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)

    def get_plane(self):
        """
        Gets current plane tuple

        :return: {tuple} (plane label, plane position)
        """
        return self.plane, self.planePos

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
        name = str(self.proj_combo_box.currentText())
        self.currentProjection = name

        if self.currentProjection == '3D':
            # disable spinbox
            self.proj_spin_box.setEnabled(False)
            self.set_drawing_style("3D")
            if tvw.completedFirstMCS:
                tvw.newDrawingUserRequest = True

        elif self.currentProjection == 'xy':
            self.update_projection_spin_box(spin_box_value=self.xyPlane)

        elif self.currentProjection == 'xz':
            self.update_projection_spin_box(spin_box_value=self.xzPlane)

        elif self.currentProjection == 'yz':
            self.update_projection_spin_box(spin_box_value=self.yzPlane)

        self.current_screenshot_data = self.compute_current_screenshot_data()

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

        if self.currentProjection == 'xy':
            if val > self.xyMaxPlane:
                val = self.xyMaxPlane
            self.proj_spin_box.setValue(val)
            # for some bizarre(!) reason, val=0 for xyPlane
            self.set_plane(self.currentProjection, val)
            self.xyPlane = val

        elif self.currentProjection == 'xz':
            if val > self.xzMaxPlane: val = self.xzMaxPlane
            self.proj_spin_box.setValue(val)
            self.set_plane(self.currentProjection, val)
            self.xzPlane = val

        elif self.currentProjection == 'yz':
            if val > self.yzMaxPlane: val = self.yzMaxPlane
            self.proj_spin_box.setValue(val)
            self.set_plane(self.currentProjection, val)
            self.yzPlane = val

        self.current_screenshot_data = self.compute_current_screenshot_data()
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
        self.current_screenshot_data = self.compute_current_screenshot_data()

        tvw._drawField()

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        style = _style.upper()
        if style == "2D":
            self.draw3DFlag = False
            self.gd.get_renderer().SetActiveCamera(self.camera2D)
            self.qvtkWidget.setMouseInteractionSchemeTo2D()
        elif style == "3D":
            self.draw3DFlag = True
            self.gd.get_renderer().SetActiveCamera(self.camera3D)
            self.qvtkWidget.setMouseInteractionSchemeTo3D()

    @staticmethod
    def set_camera_from_graphics_window_data(camera, gwd):
        """
        Sets camera from graphics window data

        :param camera: {vtkCamera} camera obj
        :param gwd: {instance og GrtaphicsWindowData}
        :return: None
        """
        camera.SetClippingRange(gwd.cameraClippingRange)
        camera.SetFocalPoint(gwd.cameraFocalPoint)
        camera.SetPosition(gwd.cameraPosition)
        camera.SetViewUp(gwd.cameraViewUp)

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
                self.set_camera_from_graphics_window_data(camera=self.camera3D, gwd=gwd)
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
                self.set_camera_from_graphics_window_data(camera=self.camera2D, gwd=gwd)

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

    def get_graphics_window_data(self)->GraphicsWindowData:
        """
        returns instance of GraphicsWindowData for current widget

        :return:
        """
        tvw = self.parentWidget()
        print('UPDATE getGraphicsWindowData')

        gwd = GraphicsWindowData()
        active_camera = self.get_active_camera()
        gwd.sceneName = str(self.field_combo_box.currentText())
        gwd.sceneType = tvw.fieldTypes[gwd.sceneName]

        gwd.winType = GRAPHICS_WINDOW_LABEL

        if self.current_screenshot_data.spaceDimension == '3D':
            gwd.is3D = True
        else:

            gwd.planeName = self.current_screenshot_data.projection
            gwd.planePosition = self.current_screenshot_data.projectionPosition

        gwd.cameraClippingRange = active_camera.GetClippingRange()
        gwd.cameraFocalPoint = active_camera.GetFocalPoint()
        gwd.cameraPosition = active_camera.GetPosition()
        gwd.cameraViewUp = active_camera.GetViewUp()

        return gwd

    def add_screenshot_conf(self):
        """
        Adds screenshot configuration data for a current scene

        :return: None
        """
        tvw = self.parentWidget()
        print(MODULENAME, '  _takeShot():  self.renWin.GetSize()=', self.renWin.GetSize())
        camera = self.get_active_camera()

        if tvw.screenshotManager is not None:
            field_name = str(self.field_combo_box.currentText())

            field_type = tvw.fieldTypes[field_name]
            field_name_type_tuple = (field_name, field_type)
            print(MODULENAME, '  _takeShot():  fieldType=', field_name_type_tuple)

            if self.draw3DFlag:
                metadata = self.get_metadata(field_name=field_name, field_type=field_type)
                tvw.screenshotManager.add_3d_screenshot(field_name, field_type, camera, metadata)
            else:
                plane, position = self.get_plane()
                metadata = self.get_metadata(field_name=field_name, field_type=field_type)
                tvw.screenshotManager.add_2d_screenshot(field_name, field_type, plane,
                                                        position, camera, metadata)

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

        self.update_cross_section(basic_simulation_data)

        self.currentProjection = 'xy'
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

        field_dim = basic_simulation_data.fieldDim
        self.xyMaxPlane = field_dim.z - 1
        self.xyPlane = field_dim.z / 2

        self.xzMaxPlane = field_dim.y - 1
        self.xzPlane = field_dim.y / 2

        self.yzMaxPlane = field_dim.x - 1
        self.yzPlane = field_dim.x / 2

    def update_field_types_combo_box(self, field_types: dict) -> None:
        """
        Updates combo boxes

        :param field_types:{str}
        :return:
        """

        # assign field types to be the same as field types in the workspace
        self.fieldTypes = field_types

        cb = self.field_combo_box
        current_text = None
        if cb.count():
            current_text = self.field_combo_box.currentText()

        cb.clear()
        cb.addItem("Cell_Field")
        for key in list(self.fieldTypes.keys()):
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
        self.reset_camera()
        self.copy_camera(src=self.camera2D, dst=self.camera3D)

    def largest_dim(self, dim):
        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]

        return ldim

    def reset_all_cameras(self, bsd):
        field_dim = [bsd.fieldDim.x, bsd.fieldDim.y, bsd.fieldDim.z]

        distance = self.largest_dim(field_dim) * 2

        self.camera2D.SetPosition(field_dim[0] / 2, field_dim[1] / 2, distance)
        self.camera2D.SetFocalPoint(field_dim[0] / 2, field_dim[1] / 2, 0)
        self.camera2D.SetClippingRange(distance - 1, distance + 1)
        self.gd.get_renderer().ResetCameraClippingRange()

        self.copy_camera(src=self.camera2D, dst=self.camera3D)

    def zoom_in(self):
        """
        Zooms in view

        :return:
        """
        self.qvtkWidget.zoomIn()

    def zoom_out(self):
        """
        Zooms out view

        :return:
        """

        self.qvtkWidget.zoomOut()

    def reset_camera(self):
        """
        Resets camera to default settings

        :return:
        """
        self.qvtkWidget.resetCamera()

    def closeEvent(self, ev):
        """

        :param ev:
        :return:
        """
        print('CHANGE and update closeEvent')
        # cleaning up to release memory - notice that if we do not do this cleanup this widget
        # will not be destroyed and will take sizeable portion of the memory
        # not a big deal for a single simulation but repeated runs can easily exhaust all system memory

        if self.qvtkWidget:
            self.qvtkWidget.close()
        self.qvtkWidget = None
        collect()
        super(GraphicsFrameWidget, self).closeEvent(ev)
