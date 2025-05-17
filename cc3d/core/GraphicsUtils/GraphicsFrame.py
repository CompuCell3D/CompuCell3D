from typing import Callable, Dict, List, Optional, Sequence
import cc3d
from cc3d.core.enums import *
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.core.GraphicsUtils.ScreenshotManagerCore import ScreenshotManagerCore
from .GraphicsWindowData import GraphicsWindowData
from cc3d.core.GraphicsUtils.ScreenshotData import ScreenshotData
import cc3d.CompuCellSetup
from cc3d.core.GraphicsUtils.utils import color_to_rgba, cs_string_to_typed_list
from collections import OrderedDict

from vtkmodules.vtkRenderingCore import vtkTextActor


def default_field_label() -> vtkTextActor:
    """Generate a default field label actor"""

    field_label_actor = vtkTextActor()
    field_label_actor.GetTextProperty().SetFontFamilyToArial()
    field_label_actor.GetTextProperty().SetFontSize(14)
    field_label_actor.GetTextProperty().SetColor(1, 1, 1)
    field_label_actor.GetTextProperty().SetVerticalJustificationToTop()
    field_label_actor.GetTextProperty().SetJustificationToRight()
    field_label_actor.GetTextProperty().FrameOn()
    field_label_actor.GetTextProperty().SetFrameWidth(3)
    field_label_actor.GetTextProperty().SetBackgroundOpacity(1.0)
    field_label_actor.SetTextScaleModeToViewport()
    field_label_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    field_label_actor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
    field_label_actor.SetPosition(0.98, 0.98)
    return field_label_actor


class GraphicsFrame:
    """
    Base class for bringing together the components to offer interactive real-time simulation data visualization
    """

    def __init__(self,
                 generic_drawer: GenericDrawer,
                 current_bsd=None,
                 config_hook=None):

        self.plane = None
        self.planePos = None

        self.currentProjection = 'xy'
        self.projection_position = 0
        self.xyPlane = None
        self.xzPlane = None
        self.yzPlane = None
        self.xyMaxPlane = None
        self.xzMaxPlane = None
        self.yzMaxPlane = None

        self.draw3DFlag = False

        self.field_name = "Cell_Field"
        """Current field name"""
        self.fieldTypes: Optional[Dict[str, str]] = {}
        """Map of current field names to field type"""

        self.config_hook = config_hook

        self.renWin, self.vtkWidget = self.get_vtk_window()

        self.gd = generic_drawer
        self.gd.set_interactive_camera_flag(True)

        # placeholder for current screenshot data
        self.current_screenshot_data = None

        # placeholder for currently used basic simulation data
        self.current_bsd = current_bsd

        self.camera2D = self.gd.get_active_camera()
        self.camera3D = self.gd.get_renderer().MakeCamera()

        self.renWin.AddRenderer(self.gd.get_renderer())

        self.metadata_fetcher_dict: Dict[str, Callable[[str, str], dict]] = {
            'CellField': self.get_cell_field_metadata,
            'ConField': self.get_con_field_metadata,
            'ScalarField': self.get_con_field_metadata,
            'ScalarFieldCellLevel': self.get_con_field_metadata,
            'VectorField': self.get_vector_field_metadata,
            'VectorFieldCellLevel': self.get_vector_field_metadata,
        }

        if self.current_bsd is not None:
            self.set_initial_cross_section(self.current_bsd)

            self.init_field_types()

    def get_vtk_window(self):
        """Get an initialized vtk window and window interactor"""

        raise NotImplementedError

    def store_gui_vis_config(self, scr_data):
        """
        Stores visualization settings such as cell borders, on/or cell on/off etc...

        :param scr_data: data to populate
        :type scr_data: ScreenshotData
        :return: None
        """

        raise NotImplementedError

    def zoom_in(self):
        """
        Zooms in view

        :return: None
        """

        pass

    def zoom_out(self):
        """
        Zooms out view

        :return: None
        """

        pass

    @property
    def config(self):
        """
        Returns the object used for fetching plot configuration details.

        Ordinarily, this is whatever is set on persistent globals. However, it can be replaced with any
        object that implements ``getSetting`` in the same way as :meth:`Configuration.getSetting`.
        The object can be used by setting the attribute :attr:`config_hook` to it. Subclasses should use
        this hook when avoiding accessing the configuration (*e.g.*, in spawned processes).

        :return: object used for fetching plot configuration details
        """
        if self.config_hook is not None:
            return self.config_hook
        return cc3d.CompuCellSetup.persistent_globals.configuration

    def get_concentration_field_names(self) -> List[str]:
        """
        Get the current list of concentration field names.

        Subclasses should override this when avoiding persistent globals.

        :return: list of concentration field names
        :rtype: List[str]
        """

        return cc3d.CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector()


    def get_vector_field_names_engine_owned(self) -> List[str]:
        """
        Get the current list of vector field names that were created inside C+ engine

        Subclasses should override this when avoiding persistent globals.

        :return: list of concentration field names
        :rtype: List[str]
        """

        return cc3d.CompuCellSetup.persistent_globals.simulator.getVectorFieldNameVectorEngineOwned()


    def get_fields_to_create(self) -> Dict[str, str]:
        """
        Get the current names and types of fields to create.

        Subclasses should override this when avoiding persistent globals.

        :return: mapping of field name to field type
        :rtype: Dict[str, str]
        """

        field_dict = cc3d.CompuCellSetup.persistent_globals.field_registry.get_fields_to_create_dict()
        return {field_name: field_adapter.field_type for field_name, field_adapter in field_dict.items()}

    def init_field_types(self):
        """
        Initialize field types

        :return: None
        """
        # most likely this function can be eliminated altogether - field type initialization takes place in
        # setFieldTypes in SimpleTabView.py

        if cc3d.CompuCellSetup.persistent_globals.simulator is not None:

            self.fieldTypes["Cell_Field"] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[CELL_FIELD]

            # get concentration fields from simulator
            for fieldName in self.get_concentration_field_names():
                self.fieldTypes[fieldName] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[CON_FIELD]

            # for fieldName in self.get_vector_field_names_engine_owned():
            #     self.fieldTypes[fieldName] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[SHARED_VECTOR_NUMPY_FIELD]

            # inserting extra scalar fields managed from Python script
            for field_name, field_type in self.get_fields_to_create().items():
                self.fieldTypes[field_name] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[field_type]

    def copy_camera(self, src, dst):
        """
        Copies camera settings

        :param src: source camera
        :type src: vtkCamera
        :param dst: destination camera
        :type dst: vtkCamera
        :return: None
        """

        dst.SetClippingRange(src.GetClippingRange())
        dst.SetFocalPoint(src.GetFocalPoint())
        dst.SetPosition(src.GetPosition())
        dst.SetViewUp(src.GetViewUp())

    def get_metadata(self, field_name, field_type):
        """
        Fetches a dictionary that summarizes graphics/configs settings for the current scene

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """
        # field_type = field_type if isinstance(field_type, str) else field_type.field_type
        field_type = get_field_type(field_type_obj=field_type)
        field_precision_type = get_field_precision_type(field_type_obj=field_type)
        try:
            metadata_fetcher_fcn = self.metadata_fetcher_dict[field_type]
        except KeyError:
            return {}

        metadata = metadata_fetcher_fcn(field_name, field_type)
        # adding field precision type
        metadata['field_precision'] = field_precision_type


        return metadata

    def get_cell_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene for a cell field

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """

        metadata_dict = self.get_config_metadata(field_name=field_name, field_type=field_type)
        return metadata_dict

    def get_config_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render borders

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """

        metadata_dict = {'BorderColor': color_to_rgba(self.config.getSetting('BorderColor')),
                         'ClusterBorderColor': color_to_rgba(self.config.getSetting('ClusterBorderColor')),
                         'BoundingBoxColor': color_to_rgba(self.config.getSetting('BoundingBoxColor')),
                         'AxesColor': color_to_rgba(self.config.getSetting('AxesColor')),
                         'ContourColor': color_to_rgba(self.config.getSetting('ContourColor')),
                         'WindowColor': color_to_rgba(self.config.getSetting('WindowColor')),
                         'FPPLinksColor': color_to_rgba(self.config.getSetting('FPPLinksColor')),
                         'ShowHorizontalAxesLabels': self.config.getSetting('ShowHorizontalAxesLabels'),
                         'ShowVerticalAxesLabels': self.config.getSetting('ShowVerticalAxesLabels')}

        # type-color map
        type_color_map_dict = OrderedDict()
        config_type_color_map = self.config.getSetting("TypeColorMap")
        for type_id, qt_color in list(config_type_color_map.items()):
            type_color_map_dict[type_id] = color_to_rgba(qt_color)

        metadata_dict['TypeColorMap'] = type_color_map_dict

        return metadata_dict

    def get_con_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene for a concentration field

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """

        # metadata_dict = {}
        metadata_dict = self.get_config_metadata(field_name=field_name, field_type=field_type)
        metadata_dict['MinRangeFixed'] = self.config.getSetting("MinRangeFixed", field_name)
        metadata_dict['MaxRangeFixed'] = self.config.getSetting("MaxRangeFixed", field_name)
        metadata_dict['MinRange'] = self.config.getSetting("MinRange", field_name)
        metadata_dict['MaxRange'] = self.config.getSetting("MaxRange", field_name)
        metadata_dict['ContoursOn'] = self.config.getSetting("ContoursOn", field_name)
        metadata_dict['NumberOfContourLines'] = self.config.getSetting("NumberOfContourLines", field_name)
        metadata_dict['ScalarIsoValues'] = cs_string_to_typed_list(
            self.config.getSetting("ScalarIsoValues", field_name))
        metadata_dict['LegendEnable'] = self.config.getSetting("LegendEnable", field_name)
        metadata_dict['DisplayMinMaxInfo'] = self.config.getSetting("DisplayMinMaxInfo")

        return metadata_dict

    def get_vector_field_metadata(self, field_name, field_type):
        """
        Returns dictionary of auxiliary information needed to render a give scene for a vector field

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """

        metadata_dict = self.get_con_field_metadata(field_name=field_name, field_type=field_type)
        metadata_dict['ArrowLength'] = self.config.getSetting('ArrowLength', field_name)
        metadata_dict['FixedArrowColorOn'] = self.config.getSetting('FixedArrowColorOn', field_name)
        metadata_dict['ArrowColor'] = color_to_rgba(self.config.getSetting('ArrowColor', field_name))
        metadata_dict['ScaleArrowsOn'] = self.config.getSetting('ScaleArrowsOn', field_name)

        return metadata_dict

    def initialize_extractor(self, field_extractor):
        """
        initialization function that sets up field extractor for the generic Drawer

        :return: None
        """

        self.gd.set_field_extractor(field_extractor=field_extractor)

    def get_current_field_name_and_type(self):
        """
        Returns current field name and type. If the type of the current field name is unknown, then the type is
        returned as empty

        :return: field name and type
        :rtype: (str, str)
        """

        try:
            return self.field_name, self.fieldTypes[self.field_name]
        except KeyError:
            return self.field_name, ''

    def compute_current_screenshot_data(self):
        """
        Computes/populates Screenshot Description data based ont he current GUI configuration
        for the current window

        :return: computed screenshot data
        :rtype: ScreenshotData
        """

        scr_data = ScreenshotData()
        self.store_gui_vis_config(scr_data=scr_data)

        field_name, field_type = self.get_current_field_name_and_type()
        scr_data.plotData = (field_name, field_type)

        metadata = self.get_metadata(field_name=scr_data.plotData[0], field_type=scr_data.plotData[1])

        if self.currentProjection == '3D':
            scr_data.spaceDimension = '3D'
        else:
            scr_data.spaceDimension = '2D'
            scr_data.projection = self.currentProjection
            scr_data.projectionPosition = self.projection_position

        scr_data.metadata = metadata

        return scr_data

    def draw(self, basic_simulation_data=None):
        """
        Main drawing function - calls ok_to_draw fcn from the GenericDrawer. All drawing happens here

        :param basic_simulation_data: simulation data
        :type basic_simulation_data: BasicSimulationData
        :return: None
        """

        if basic_simulation_data is None:
            basic_simulation_data = self.current_bsd

        self.current_bsd = basic_simulation_data

        self.gd.clear_display()

        pg = cc3d.CompuCellSetup.persistent_globals

        self.current_screenshot_data = self.compute_current_screenshot_data()
        try:
            self.current_screenshot_data.cell_shell_optimization = pg.configuration.getSetting('CellShellOptimization')
        except KeyError:
            print("Could not extract CellShellOptimization setting using default")

        self.gd.draw(screenshot_data=self.current_screenshot_data, bsd=basic_simulation_data, screenshot_name='')

        # this call seems to be needed to refresh vtk widget
        self.gd.get_renderer().ResetCameraClippingRange()
        # essential call to refresh screen . otherwise need to move/resize graphics window
        self.Render()

    def Render(self):
        """
        Do rendering

        :return: None
        """

        # color = cc3d.CompuCellSetup.persistent_globals.configuration.getSetting("WindowColor")
        # self.gd.get_renderer().SetBackground(float(color.red()) / 255, float(color.green()) / 255,
        #                                      float(color.blue()) / 255)
        self.vtkWidget.Render()

    def get_active_camera(self):
        """
        Get the active camera

        :return: camera object
        :rtype: vtkCamera
        """

        return self.gd.get_active_camera()

    def set_plane(self, plane, pos):
        """Set the plane and position"""

        (self.plane, self.planePos) = (str(plane).upper(), pos)

    def get_plane(self):
        """
        Gets current plane tuple

        :return: {tuple} (plane label, plane position)
        """

        return self.plane, self.planePos

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
        elif style == "3D":
            self.draw3DFlag = True
            self.gd.get_renderer().SetActiveCamera(self.camera3D)

    @staticmethod
    def set_camera_from_graphics_window_data(camera, gwd):
        """
        Sets camera from graphics window data

        :param camera: camera obj
        :type camera: vtkCamera
        :param gwd: data to use
        :type gwd: GrtaphicsWindowData
        :return: None
        """

        camera.SetClippingRange(gwd.cameraClippingRange)
        camera.SetFocalPoint(gwd.cameraFocalPoint)
        camera.SetPosition(gwd.cameraPosition)
        camera.SetViewUp(gwd.cameraViewUp)

    def apply_3D_graphics_window_data(self, gwd):
        """
        Applies graphics window configuration data (stored on a disk) to graphics window (3D version)

        :param gwd: data to use
        :type gwd: GraphicsWindowData
        :return: None
        """

        self.currentProjection = '3D'

        # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D
        self.set_camera_from_graphics_window_data(camera=self.camera3D, gwd=gwd)

    def apply_2D_graphics_window_data(self, gwd):
        """
        Applies graphics window configuration data (stored on a disk) to graphics window (2D version)

        :param gwd: data to use
        :type gwd: GraphicsWindowData
        :return: None
        """

        self.currentProjection = gwd.planeName
        self.projection_position = gwd.planePosition

        # notice: there are two cameras one for 2D and one for 3D  here we set camera for 2D
        self.set_camera_from_graphics_window_data(camera=self.camera2D, gwd=gwd)

    def apply_graphics_window_data(self, gwd=None):
        """
        Applies graphics window configuration data to graphics window

        :param gwd: data to use
        :type gwd: GraphicsWindowData or None
        :return: None
        """

        if gwd is None:
            gwd = self.get_graphics_window_data()

        for field_name in self.fieldTypes.keys():
            if field_name == gwd.sceneName:
                self.field_name = field_name

                # setting 2D projection or 3D
                if gwd.is3D:
                    self.apply_3D_graphics_window_data(gwd)
                else:
                    self.apply_2D_graphics_window_data(gwd)

                break

    def get_graphics_window_data(self) -> GraphicsWindowData:
        """Returns instance of GraphicsWindowData for current widget"""

        gwd = GraphicsWindowData()
        gwd.sceneName, gwd.sceneType = self.get_current_field_name_and_type()

        gwd.winType = GRAPHICS_WINDOW_LABEL

        if self.current_screenshot_data.spaceDimension == '3D':
            gwd.is3D = True
        else:
            gwd.planeName = self.current_screenshot_data.projection
            gwd.planePosition = self.current_screenshot_data.projectionPosition

        active_camera = self.get_active_camera()
        gwd.cameraClippingRange = active_camera.GetClippingRange()
        gwd.cameraFocalPoint = active_camera.GetFocalPoint()
        gwd.cameraPosition = active_camera.GetPosition()
        gwd.cameraViewUp = active_camera.GetViewUp()

        return gwd

    def add_screenshot_config(self, screenshot_manager):
        """
        Adds screenshot configuration data for a current scene

        :param screenshot_manager: screenshot manager
        :type screenshot_manager: ScreenshotManagerCore
        :return: None
        """

        camera = self.get_active_camera()

        field_type = self.fieldTypes[self.field_name]

        if self.draw3DFlag:
            metadata = self.get_metadata(field_name=self.field_name, field_type=field_type)
            screenshot_manager.add_3d_screenshot(self.field_name, field_type, camera, metadata)
        else:
            plane, position = self.get_plane()
            metadata = self.get_metadata(field_name=self.field_name, field_type=field_type)
            screenshot_manager.add_2d_screenshot(self.field_name, field_type, plane,
                                                 position, camera, metadata)

    def set_initial_cross_section(self, basic_simulation_data):
        """
        Set the initial cross-section using simulation data

        :param basic_simulation_data: simulation data to use for update
        :type basic_simulation_data: BasicSimulationData
        :return: None
        """

        self.update_cross_section(basic_simulation_data)
        self.currentProjection = 'xy'
        self.projection_position = int(basic_simulation_data.fieldDim.z / 2)

    def update_cross_section(self, basic_simulation_data):
        """
        Update the cross-section using simulation data

        :param basic_simulation_data: simulation data to use for update
        :type basic_simulation_data: BasicSimulationData
        :return:
        """

        field_dim = basic_simulation_data.fieldDim
        self.xyMaxPlane = field_dim.z - 1
        self.xyPlane = field_dim.z / 2

        self.xzMaxPlane = field_dim.y - 1
        self.xzPlane = field_dim.y / 2

        self.yzMaxPlane = field_dim.x - 1
        self.yzPlane = field_dim.x / 2

    @staticmethod
    def largest_dim(dim: Sequence):
        """Convenience function to get the largest element of a sequence"""

        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]

        return ldim

    def reset_all_cameras(self, bsd):
        """
        Reset all cameras

        :param bsd: basic simulation data
        :type bsd: BasicSimulationData
        :return: None
        """

        field_dim = [bsd.fieldDim.x, bsd.fieldDim.y, bsd.fieldDim.z]

        distance = self.largest_dim(field_dim) * 2

        self.camera2D.SetPosition(field_dim[0] / 2, field_dim[1] / 2, distance)
        self.camera2D.SetFocalPoint(field_dim[0] / 2, field_dim[1] / 2, 0)
        self.camera2D.SetClippingRange(distance - 1, distance + 1)
        self.gd.get_renderer().ResetCameraClippingRange()

        self.copy_camera(src=self.camera2D, dst=self.camera3D)

    def apply_camera_standard_2d_view(self) -> bool:
        """
        Apply a standard two-dimensional view to the camera based on current data

        :return: True if applied
        :rtype: bool
        """

        if not self.draw3DFlag or self.current_bsd is None:
            return False
        elif self.plane not in ['xy', 'yz', 'xz']:
            return False

        if self.plane == 'xy':
            field_dim = [self.current_bsd.fieldDim.x, self.current_bsd.fieldDim.y]
        elif self.plane == 'yz':
            field_dim = [self.current_bsd.fieldDim.y, self.current_bsd.fieldDim.z]
        else:
            field_dim = [self.current_bsd.fieldDim.x, self.current_bsd.fieldDim.z]

        distance = self.largest_dim(field_dim) * 2

        self.camera2D.SetPosition(field_dim[0] / 2, field_dim[1] / 2, distance)
        self.camera2D.SetFocalPoint(field_dim[0] / 2, field_dim[1] / 2, 0)
        self.camera2D.SetClippingRange(distance - 1, distance + 1)
        self.gd.get_renderer().ResetCameraClippingRange()

        return True

    def reset_camera(self):
        """
        Resets camera to default settings

        :return: None
        """

        self.gd.get_renderer().ResetCamera()

    def close(self):
        """
        Clean up to release memory - notice that if we do not do this cleanup this widget
        will not be destroyed and will take sizeable portion of the memory

        :return: None
        """

        self.vtkWidget = None
