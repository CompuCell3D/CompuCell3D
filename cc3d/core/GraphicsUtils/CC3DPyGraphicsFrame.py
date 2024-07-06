"""
Defines features for interactive visualization for use with CC3D simservice applications.
"""

# todo: add CC3D logo to window title icon
# todo: disable built-in key commands for closing render windows
# todo: add support for additional plots (e.g., tracking fields)
# todo: improve synchronization to allow asynchronous execution and visualization while handling race conditions
# todo: implement setting a frame by provided screenshot data

import json
import multiprocessing
import numpy
import os
import threading
from typing import Any, Callable, Dict, List
import warnings
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
try:
    # vtk 8
    from vtkmodules.vtkIOExportPython import vtkGL2PSExporter
except ModuleNotFoundError:
    # vtk 9
    from vtkmodules.vtkIOExportGL2PS import vtkGL2PSExporter
from vtkmodules.vtkIOImage import vtkJPEGWriter, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor, vtkRenderWindow, vtkWindowToImageFilter, vtkTextActor
from vtkmodules.util import numpy_support

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import simulation_utils
from cc3d.core.BasicSimulationData import BasicSimulationData
from cc3d.core import Configuration
from cc3d.core.enums import *
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.cpp.PlayerPython import FieldExtractorCML, FieldStreamer, FieldStreamerData, FieldWriterCML
from cc3d.core.GraphicsUtils.GraphicsFrame import GraphicsFrame, default_field_label
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrameIO import *
from cc3d.cpp.CompuCell import Dim3D


# Get default data and do some minor cleanup
CONFIG_DEFAULT_SETTINGS = Configuration.default_settings_dict_xml()
if 'TypeColorMap' in CONFIG_DEFAULT_SETTINGS.keys():
    for k, v in list(CONFIG_DEFAULT_SETTINGS['TypeColorMap'].items()):
        CONFIG_DEFAULT_SETTINGS['TypeColorMap'][int(k)] = CONFIG_DEFAULT_SETTINGS['TypeColorMap'].pop(k)


class FieldStreamerDataPy:
    """Simple container for serializing :class:`FieldStreamerData` instances"""

    def __init__(self):
        super().__init__()

        self.cell_field_names: List[str] = []
        self.conc_field_names: List[str] = []
        self.scalar_field_names: List[str] = []
        self.scalar_field_cell_level_names: List[str] = []
        self.vector_field_names: List[str] = []
        self.vector_field_cell_level_names: List[str] = []
        self.field_dim = Dim3D()
        self.data = ''

    @staticmethod
    def from_base(fsd):
        """
        Return a python wrap instance from a base type instance

        :param fsd: base type instance
        :type fsd: FieldStreamerData
        :return: python wrap instance
        :rtype: FieldStreamerDataPy
        """

        result = FieldStreamerDataPy()
        v2l = lambda v: [v[i] for i in range(v.size())]
        result.cell_field_names = v2l(fsd.cellFieldNames)
        result.conc_field_names = v2l(fsd.concFieldNames)
        result.scalar_field_names = v2l(fsd.scalarFieldNames)
        result.scalar_field_cell_level_names = v2l(fsd.scalarFieldCellLevelNames)
        result.vector_field_names = v2l(fsd.vectorFieldNames)
        result.vector_field_cell_level_names = v2l(fsd.vectorFieldCellLevelNames)
        result.field_dim = fsd.fieldDim
        result.data = fsd.data
        return result

    @staticmethod
    def to_base(fsd):
        """
        Return a base type instance from a python wrap instance

        :param fsd: python wrap instance
        :type fsd: FieldStreamerDataPy
        :return: base type instance
        :rtype: FieldStreamerData
        """

        result = FieldStreamerData()
        l2v = lambda l, v: [v.push_back(s) for s in l]
        l2v(fsd.cell_field_names, result.cellFieldNames)
        l2v(fsd.conc_field_names, result.concFieldNames)
        l2v(fsd.scalar_field_names, result.scalarFieldNames)
        l2v(fsd.scalar_field_cell_level_names, result.scalarFieldCellLevelNames)
        l2v(fsd.vector_field_names, result.vectorFieldNames)
        l2v(fsd.vector_field_cell_level_names, result.vectorFieldCellLevelNames)
        result.fieldDim = fsd.field_dim
        result.data = fsd.data
        return result


def get_image_filter(ren_win: vtkRenderWindow,
                     scale: Union[int, Tuple[int, int]] = None,
                     transparent_background: bool = False) -> vtkWindowToImageFilter:
    """Create image filter"""

    image_filter = vtkWindowToImageFilter()
    image_filter.SetInput(ren_win)
    if scale is not None:
        image_filter.SetScale(scale) if isinstance(scale, int) else image_filter.SetScale(*scale)
    if transparent_background:
        image_filter.SetInputBufferTypeToRGBA()
    image_filter.ReadFrontBufferOff()
    image_filter.Update()
    return image_filter


def np_img_data(ren_win: vtkRenderWindow,
                scale: Union[int, Tuple[int, int]] = None,
                transparent_background: bool = False):
    """Get image data as numpy data"""

    image_filter = get_image_filter(ren_win=ren_win, scale=scale, transparent_background=transparent_background)

    image_output = image_filter.GetOutput()
    x_dim, y_dim, _ = image_output.GetDimensions()
    np_data = numpy_support.vtk_to_numpy(image_output)[:, [0, 1, 2]]
    np_data.reshape([x_dim, y_dim, -1])
    np_data = numpy.flip(np_data, axis=0)
    return np_data


def save_img(ren_win: vtkRenderWindow,
             file_path: str,
             scale: Union[int, Tuple[int, int]] = None,
             transparent_background: bool = False):
    """Save a window to file"""

    supported_exts = ['.eps', '.jpg', '.jpeg', '.pdf', '.png', '.svg']

    file_basename, file_ext = os.path.splitext(file_path)
    file_ext_lower = file_ext.lower()

    if file_ext_lower not in supported_exts:
        raise ValueError(f'Supported image types are {",".join(supported_exts)}')

    # Use exporter if appropriate

    if file_ext_lower in ['.eps', '.pdf', '.svg']:
        image_writer = vtkGL2PSExporter()
        image_writer.SetFilePrefix(file_basename)
        image_writer.SetRenderWindow(ren_win)
        image_writer.SetSortToBSP()
        image_writer.SilentOn()
        image_writer.Write3DPropsAsRasterImageOff()
        if file_ext_lower == '.eps':
            image_writer.SetFileFormatToEPS()
        elif file_ext_lower == '.pdf':
            image_writer.SetFileFormatToPDF()
        else:
            image_writer.SetFileFormatToSVG()
        image_writer.Write()
        return

    image_filter = get_image_filter(ren_win=ren_win, scale=scale, transparent_background=transparent_background)

    if file_ext_lower in ['.jpg', '.jpeg']:
        file_writer = vtkJPEGWriter()
    else:
        file_writer = vtkPNGWriter()
    file_writer.SetFileName(file_path)
    file_writer.SetInputData(image_filter.GetOutput())
    file_writer.Write()


class CC3DPyGraphicsFrameConnectionWatcher(threading.Thread):
    """A thread for watching a frame connection between processes"""

    def __init__(self, conn: Connection, msg_cb: Callable[[], None]):
        super().__init__(daemon=True)
        self.conn = conn
        self.msg_cb = msg_cb

    def run(self) -> None:
        """Run the thread. Executes its callback when an object is found in its pipe"""

        while True:
            if self.conn.poll():
                self.msg_cb()


class CC3DPyGraphicsFrameInterface:
    """Interface for passing requests and servicing them between processes"""

    def __init__(self, conn: Connection):

        self.conn = conn
        self.conn_watcher = CC3DPyGraphicsFrameConnectionWatcher(conn=self.conn, msg_cb=self._process_message)

    def _process_message(self):

        if not self.conn.poll():
            return

        msg: CC3DPyGraphicsFrameConnectionMsg = self.conn.recv()
        msg.service(self, self.conn)


class CC3DPyGraphicsConfig:
    """Configuration hook to request settings data between processes"""

    def __init__(self, conn: Connection):

        self._conn = conn

    def getSetting(self, *args, **kwargs):
        """Get setting value from remote source"""
        return MsgGetSetting.request(self._conn, True, *args, **kwargs)


class CC3DPyGraphicsDrawer(GenericDrawer):
    """
    Generic drawer for graphics frame.

    Functions the same as :class:`GenericDrawer`, but for doing rendering in a different process than that of the
    simulator.
    """

    def __init__(self, conn: Connection, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._conn = conn

        # Initialize field extractor; points need stored to prevent garbage collection
        self._streamed_points = None
        self.set_field_extractor(field_extractor=FieldExtractorCML())
        self._request_update_field_extractor()

        # set lattice_type and lattice_type_str on member models to prevent xml accessing
        lattice_type_str = MsgGetLatticeTypeStr.request(self._conn, True)
        lattice_type = Configuration.LATTICE_TYPES[lattice_type_str]
        self.draw_model_2D.lattice_type = lattice_type
        self.draw_model_3D.lattice_type = lattice_type
        self.draw_model_2D.lattice_type_str = lattice_type_str
        self.draw_model_3D.lattice_type_str = lattice_type_str

    def draw(self, *args, **kwargs):
        """Draw scene from remote data"""

        if not self._request_update_field_extractor():
            warnings.warn('Field extractor update failed', RuntimeWarning)
            return
        super().draw(*args, **kwargs)

    def _request_update_field_extractor(self) -> bool:

        fsd: FieldStreamerData = FieldStreamerDataPy.to_base(MsgGetStreamedData.request(self._conn, True))
        if fsd is None:
            return False
        self._streamed_points = FieldStreamer(fsd)
        self.field_extractor.setFieldDim(self._streamed_points.getFieldDim())
        self.field_extractor.setSimulationData(self._streamed_points.getPointsAddr())
        return True


class CC3DPyInteractorStyle(vtkInteractorStyleTrackballCamera):
    """
    Graphics frame style with user input callbacks and rotation control.

    To prevent rotation (e.g., for 2D plots), set :attr:`can_rotate` to ``False``.
    """

    def __init__(self):
        super().__init__()
        self.AddObserver('LeftButtonPressEvent', self._on_left_button_press_event)

        self.can_rotate = True

    def _on_left_button_press_event(self, obj, event):

        iren: vtkRenderWindowInteractor = self.GetInteractor()
        if not self.can_rotate:
            iren.SetShiftKey(1)

        self.OnLeftButtonDown()
        return 0


class CC3DPyGraphicsFrame(GraphicsFrame, CC3DPyGraphicsFrameInterface):
    """Graphics frame implementation for pure Python CC3D deployment"""

    WINDOWNAME_PREFIX = 'CC3D Visualization'

    def __init__(self,
                 interface_conn: Connection,
                 fps: int = 60,
                 window_name: str = None,
                 *args, **kwargs):

        self.metadata_data_dict: Optional[dict] = None
        self.style = None
        self._field_label_actor: Optional[vtkTextActor] = None

        CC3DPyGraphicsFrameInterface.__init__(self, conn=interface_conn)
        GraphicsFrame.__init__(self,
                               generic_drawer=CC3DPyGraphicsDrawer(conn=interface_conn),
                               config_hook=CC3DPyGraphicsConfig(conn=interface_conn),
                               *args, **kwargs)

        self.bounding_box_on = self.config.getSetting('BoundingBoxOn')
        self.cell_borders_on = self.config.getSetting('CellBordersOn')
        self.cell_glyphs_on = self.config.getSetting('CellGlyphsOn')
        self.cells_on = self.config.getSetting('CellsOn')
        self.cluster_borders_on = self.config.getSetting('ClusterBordersOn')
        self.fpp_links_on = self.config.getSetting('FPPLinksOn')
        self.lattice_axes_labels_on = self.config.getSetting('ShowAxes')
        self.lattice_axes_on = self.config.getSetting('ShowHorizontalAxesLabels') or self.config.getSetting(
            'ShowVerticalAxesLabels')

        self.metadata_fetcher_dict_copy = self.metadata_fetcher_dict.copy()
        self.metadata_fetcher_dict = {k: self.get_metadata_prefetched for k in self.metadata_fetcher_dict.keys()}

        renderer = self.gd.get_renderer()
        # noinspection PyUnresolvedReferences
        self.style.SetCurrentRenderer(renderer)
        renderer.AddActor(self.field_label_actor)

        bsd: BasicSimulationData = MsgGetBasicSimData.request(interface_conn, True)
        self.current_bsd = bsd

        self.draw()
        self.reset_camera()
        self.init_field_types()
        self.Render()

        renwin_name = str(self.WINDOWNAME_PREFIX)
        if window_name is not None:
            renwin_name += ': ' + window_name
        self.renWin.SetWindowName(renwin_name)

        self.timer_event_id = 0
        self.timer_event_type = 0
        self.event_loop_status = True
        self.vtkWidget: vtkRenderWindowInteractor
        self.vtkWidget.CreateRepeatingTimer(int(1000 / fps))
        self.vtkWidget.AddObserver('TimerEvent', self._on_step_event)

    def start(self, auto_shutdown: bool = True):
        """Start the event loop"""

        self.draw()
        self.reset_camera()

        self.vtkWidget.Start()

        if auto_shutdown:
            self.shutdown()

    def shutdown(self):
        """Stop the event loop and signal as such to interface"""

        self.event_loop_status = False
        MsgShutdownFrameInterface.request(self.conn, False)

    def _on_step_event(self, obj, event):

        event_id = obj.GetTimerEventId()
        event_type = obj.GetTimerEventType()
        if event_id != self.timer_event_id:
            return
        elif event_type == 1 or not self.event_loop_status:
            obj.TerminateApp()

        self.renWin.Render()

    @property
    def field_label_actor(self):
        """Field label actor of the frame. Text is synchronized with field name"""

        if self._field_label_actor is None:
            self._field_label_actor = default_field_label()
            self._field_label_actor.SetInput(self.field_name)
        return self._field_label_actor

    def get_vtk_window(self):
        """
        Get an initialized vtk window and window interactor.

        Implementation of :class:`GraphicsFrame` interface.
        """

        # noinspection PyArgumentList
        interactor = vtkRenderWindowInteractor()

        renWin = vtkRenderWindow()
        renWin.SetSize(600, 600)

        interactor.SetRenderWindow(renWin)
        renWin.SetInteractor(interactor)

        self.style = CC3DPyInteractorStyle()
        interactor.SetInteractorStyle(self.style)

        interactor.Initialize()

        return renWin, interactor

    def store_gui_vis_config(self, scr_data):
        """
        Save current internal data.

        Implementation of :class:`GraphicsFrame` interface.
        """

        scr_data.bounding_box_on = self.bounding_box_on
        scr_data.cell_borders_on = self.cell_borders_on
        scr_data.cell_glyphs_on = self.cell_glyphs_on
        scr_data.cells_on = self.cells_on
        scr_data.cluster_borders_on = self.cluster_borders_on
        scr_data.fpp_links_on = self.fpp_links_on
        scr_data.lattice_axes_labels_on = self.lattice_axes_labels_on
        scr_data.lattice_axes_on = self.lattice_axes_on

    def init_field_types(self):
        """Initialize field types using current internal data when available"""

        super().init_field_types()

        if not self.fieldTypes:
            self.fieldTypes = MsgInitFieldTypesFrameInterface.request(self.conn, True)

        self.metadata_data_dict = {k: {} for k in self.metadata_fetcher_dict.keys()}

        try:
            self.metadata_fetcher_dict = self.metadata_fetcher_dict_copy.copy()
            has_prefetched = True
        except (AttributeError, KeyError):
            has_prefetched = False

        [self.pull_metadata(field_name=field_name) for field_name in self.fieldTypes.keys()]

        if has_prefetched:
            self.metadata_fetcher_dict = {k: self.get_metadata_prefetched for k in self.metadata_fetcher_dict.keys()}

    def pull_metadata(self, field_name: str):

        field_type = self.fieldTypes[field_name]

        try:
            func = self.metadata_fetcher_dict_copy[field_type]
        except (AttributeError, KeyError):
            func = self.get_metadata

        self.metadata_data_dict[field_type][field_name] = func(field_name=field_name,
                                                               field_type=field_type)

    def get_metadata_prefetched(self, field_name, field_type):
        """
        Returns dictionary pre-fetched auxiliary information needed to render a give scene

        :param field_name: field_name
        :type field_name: str
        :param field_type: field type
        :type field_type: str
        :return: auxiliary data
        :rtype: dict
        """

        return self.metadata_data_dict[field_type][field_name]

    def get_screenshot_data(self):
        """
        Computes/populates Screenshot Description data based on the current GUI configuration
        for the current window

        :return: computed screenshot data
        :rtype: ScreenshotData
        """

        ss_data = self.compute_current_screenshot_data()
        ss_data.extractCameraInfo(self.get_active_camera())
        return ss_data

    def get_concentration_field_names(self) -> List[str]:
        """Get concentration field names from remote source"""
        return MsgGetConcFieldNames.request(self.conn, True)

    def get_fields_to_create(self) -> Dict[str, str]:
        """Get names of fields to create from remote source"""
        return MsgGetFieldsToCreate.request(self.conn, True)

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        field_names = list(self.fieldTypes.keys())

        if not field_names:
            warnings.warn('Field names not available', RuntimeWarning)
            return

        if _field_name not in field_names:
            warnings.warn(f'Field name not known: {_field_name}. Available field names are' + ','.join(field_names),
                          RuntimeWarning)
            return

        self.field_name = _field_name
        self.field_label_actor.SetInput(self.field_name)

        self.current_screenshot_data = self.compute_current_screenshot_data()

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        super().set_drawing_style(_style)

        self.current_screenshot_data = self.compute_current_screenshot_data()

        self.apply_graphics_window_data()

        self.style.can_rotate = _style == '3D'

    def set_plane(self, plane, pos):
        """Set the plane and position"""

        # todo: get set_plane working consistently

        plane = plane.lower()

        super().set_plane(plane, pos)

        if plane == 'xy':
            self.xyPlane = pos
        elif plane == 'yz':
            self.yzPlane = pos
        elif plane == 'xz':
            self.xzPlane = pos
        self.planePos = pos

        self.set_drawing_style('2D')

    def np_img_data(self,
                    scale: Union[int, Tuple[int, int]] = None,
                    transparent_background: bool = False):
        """Get image data as numpy data"""

        return np_img_data(ren_win=self.renWin, scale=scale, transparent_background=transparent_background)

    def save_img(self,
                 file_path: str,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):
        """Save a window to file"""

        save_img(ren_win=self.renWin, file_path=file_path, scale=scale, transparent_background=transparent_background)

    @property
    def window_size(self) -> Tuple[int, int]:
        """Get the window size"""

        return self.renWin.GetSize()

    @window_size.setter
    def window_size(self, _size: Tuple[int, int]):
        """Set the window size"""

        self.renWin.SetSize(_size)


class CC3DPyGraphicsFrameProcess(multiprocessing.Process):
    """Process running the graphics frame."""

    def __init__(self,
                 frame_conn: Connection,
                 fps: int,
                 window_name: str = None):

        multiprocessing.Process.__init__(self, daemon=True)

        self.frame_conn = frame_conn
        """Connection for communication between frame and client"""

        self.fps = fps
        """Frame refresh rate"""

        self.window_name = window_name
        """Name to assign to frame window"""

        self.frame: Optional[CC3DPyGraphicsFrame] = None
        """Graphics frame of the process"""

        self.queue_input = multiprocessing.Queue()
        """Input queue for communication with a controller"""

        self.queue_output = multiprocessing.Queue()
        """Output queue for communication with a controller"""

        self.confirm_conn: Optional[Connection] = None
        """Optional connection to report successful launch"""

    def confirmational_start(self, confirm_conn: Connection) -> None:
        """Start the process with a confirmational ping through a pipe on launch"""
        self.confirm_conn = confirm_conn
        super().start()

    def run(self) -> None:
        """Run the frame process"""

        self.frame = CC3DPyGraphicsFrame(interface_conn=self.frame_conn, fps=self.fps, window_name=self.window_name)
        self.frame.vtkWidget.AddObserver('TimerEvent', self._process_messages)

        if self.confirm_conn is not None:
            self.confirm_conn.send(True)

        self.frame.start()

    def _process_messages(self, obj, event):

        # scan through all messages first; process only the last request to draw
        # draw message can decide whether multiple draw requests can be issued
        messages = []
        draw_message = None
        while not self.queue_input.empty():
            msg: FrameControlMessage = self.queue_input.get()
            if isinstance(msg, ControlMessageDraw):
                draw_message = msg
            else:
                messages.append(msg)

        for msg in messages:
            msg.process(self)
        if draw_message is not None:
            draw_message.process(self)

    def close(self) -> None:
        if self.frame is not None:
            self.frame.close()
            self.frame = None
        super().close()


class CC3DPyGraphicsFrameControlInterface:
    """
    Control interface for a graphics frame process. Useful for sending control logic to a graphics frame process.
    """

    def __init__(self, proc: CC3DPyGraphicsFrameProcess):

        self.frame_proc = proc
        """Frame process"""

        self._mgs_counter = 0
        """Counter for assigning unique ids to messages"""

    def _process_message(self, msg):
        if self.frame_proc is None:
            raise AttributeError
        elif not isinstance(msg, FrameControlMessage):
            raise TypeError

        msg: FrameControlMessage
        msg.id = self._mgs_counter
        self._mgs_counter += 1

        self.frame_proc.queue_input.put(msg)
        return msg.control(self)

    def draw(self, blocking: bool = False):
        """Update visualization data in rendering process"""
        self._process_message(ControlMessageDraw(blocking=blocking))

    def close(self):
        """Close the graphics frame process"""
        self._process_message(ControlMessageShutdown())

    def get_screenshot_data(self):
        """Get screenshot data using the current frame configuration"""
        return self._process_message(ControlMessageGetScreenshotData())

    def pull_metadata(self, _field_name: str):
        """Update field metadata storage on a frame for a field"""
        self._process_message(ControlMessagePullMetadata(_field_name))

    def get_field_name(self):
        """Get the name of the current field"""
        return self._process_message(ControlMessageGetAttr(attr_name='field_name'))

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""
        self._process_message(ControlMessageSetFieldName(_field_name))

    def set_plane(self, plane, pos):
        """Set the plane and position"""
        self._process_message(ControlMessageSetPlanePos(plane, pos))

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """
        self._process_message(ControlMessageSetDrawingStyle(_style))

    def np_img_data(self,
                    scale: Union[int, Tuple[int, int]] = None,
                    transparent_background: bool = False):
        """Get image data as numpy data"""
        return self._process_message(ControlMessageGetNPImageData(scale=scale,
                                                                  transparent_background=transparent_background))

    def save_img(self,
                 file_path: str,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):
        """Save a window to file"""
        self._process_message(ControlMessageSaveImage(file_path=file_path,
                                                      scale=scale,
                                                      transparent_background=transparent_background))

    def get_bounding_box_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='bounding_box_on'))

    def set_bounding_box_on(self, _bounding_box_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='bounding_box_on',
                                                    attr_val=_bounding_box_on))

    def get_cell_borders_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='cell_borders_on'))

    def set_cell_borders_on(self, _cell_borders_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='cell_borders_on',
                                                    attr_val=_cell_borders_on))

    def get_cell_glyphs_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='cell_glyphs_on'))

    def set_cell_glyphs_on(self, _cell_glyphs_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='cell_glyphs_on',
                                                    attr_val=_cell_glyphs_on))

    def get_cells_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='cells_on'))

    def set_cells_on(self, _cells_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='cells_on',
                                                    attr_val=_cells_on))

    def get_cluster_borders_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='cluster_borders_on'))

    def set_cluster_borders_on(self, _cluster_borders_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='cluster_borders_on',
                                                    attr_val=_cluster_borders_on))

    def get_fpp_links_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='fpp_links_on'))

    def set_fpp_links_on(self, _fpp_links_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='fpp_links_on',
                                                    attr_val=_fpp_links_on))

    def get_lattice_axes_labels_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='lattice_axes_labels_on'))

    def set_lattice_axes_labels_on(self, _lattice_axes_labels_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='lattice_axes_labels_on',
                                                    attr_val=_lattice_axes_labels_on))

    def get_lattice_axes_on(self) -> bool:
        return self._process_message(ControlMessageGetAttr(attr_name='lattice_axes_on'))

    def set_lattice_axes_on(self, _lattice_axes_on: bool):
        self._process_message(ControlMessageSetAttr(attr_name='lattice_axes_on',
                                                    attr_val=_lattice_axes_on))

    def get_window_size(self) -> Tuple[int, int]:
        return self._process_message(ControlMessageGetAttr(attr_name='window_size'))

    def set_window_size(self, _window_size: Tuple[int, int]):
        self._process_message(ControlMessageSetAttr(attr_name='window_size',
                                                    attr_val=_window_size))


class CC3DPyGraphicsFrameClientBase:
    """Base class for a CC3D Python graphics frame client"""

    CONFIG_ENTRIES: List[Union[str, Tuple[str, Any]]] = [
        'AxesColor',
        'BorderColor',
        'BoundingBoxColor',
        'BoundingBoxOn',
        'CellBordersOn',
        'CellGlyphsOn',
        'CellsOn',
        'ClusterBorderColor',
        'ClusterBordersOn',
        'ContourColor',
        'FPPLinksColor',
        'FPPLinksOn',
        'ShowAxes',
        'ShowHorizontalAxesLabels',
        'ShowVerticalAxesLabels',
        ('TypeColorMap', 0),
        ('TypeColorMap', 1),
        ('TypeColorMap', 2),
        ('TypeColorMap', 3),
        ('TypeColorMap', 4),
        ('TypeColorMap', 5),
        ('TypeColorMap', 6),
        ('TypeColorMap', 7),
        ('TypeColorMap', 8),
        ('TypeColorMap', 9),
        ('TypeColorMap', 10),
        'WindowColor'
    ]

    CONFIG_ENTRIES_FIELDS_BYNAME = [
        'MinRangeFixed',
        'MaxRangeFixed',
        'MinRange',
        'MaxRange',
        'ContoursOn',
        'NumberOfContourLines',
        'ScalarIsoValues',
        'LegendEnable'
    ]
    """Configuration keys with database values by field name"""

    CONFIG_ENTRIES_FIELDS_UNIFORM = [
        'DisplayMinMaxInfo'
    ]
    """Configuration keys with database values that are uniformly applied to all fields"""

    def __init__(self,
                 name: str = None,
                 config_fp: str = None):

        self.name = name
        """Convenience attribute for labeling a frame"""

        self._field_name = "Cell_Field"
        """Current target field name"""

        self._callbacks_draw: List[Callable[[], None]] = []
        """Registered callbacks for when draw is executed"""

        self._callbacks_close: List[Callable[[], None]] = []
        """Registered callbacks for when close is executed"""

        self.config_data: Optional[Dict[str, Union[Dict[str, Any], Any]]] = None
        """Pre-loaded configuration data"""

        self._preload_configs()

        if config_fp is not None:
            self.load_configs(fp=config_fp)

    def launch(self, timeout: float = None) -> Optional[Any]:
        """
        Launches the graphics frame process and blocks until startup completes

        :param timeout: permissible duration of launch attempt
        :type timeout: float
        :return: interface object on success, or None on failure
        :rtype: Any or None
        """

        raise NotImplementedError

    def close(self):
        """
        Close the frame

        :return: True on success
        :rtype: bool
        """

        pass

    def draw(self, blocking: bool = False):
        """
        Update visualization data

        :param blocking: flag to block until update is complete
        :type blocking: bool
        :return: True on success
        :rtype: bool
        """

        raise NotImplementedError

    def get_field_name(self) -> str:
        """Get the name of the current field to render"""

        return self._field_name

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        self._field_name = _field_name

    def set_plane(self, plane, pos):
        """Set the plane and position"""

        raise NotImplementedError

    @property
    def field_names(self) -> List[str]:
        """Current available field names if available, otherwise None"""

        raise NotImplementedError

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        raise NotImplementedError

    def np_img_data(self,
                    scale: Union[int, Tuple[int, int]] = None,
                    transparent_background: bool = False):
        """
        Get image data as numpy data.

        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: image array data
        :rtype: numpy.array
        """

        raise NotImplementedError

    def save_img(self,
                 file_path: str,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):
        """
        Save a window to file.

        Supported image types are .eps, .jpg, .jpeg, .pdf, .png, .svg.

        :param file_path: absolute path to save the image
        :type file_path: str
        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: None
        """

        raise NotImplementedError

    def add_callback_draw(self, _cb: Callable[[], None]):
        """
        Add a callback for when calls to draw are made. Callbacks are executed before the actual draw execution.

        :param _cb: callback
        :type _cb: () -> None
        :return: None
        """

        self._callbacks_draw.append(_cb)

    def rem_callback_draw(self, _cb: Callable[[], None]) -> bool:
        """
        Remove a callback for when calls to draw are made.

        :param _cb: callback
        :type _cb: () -> None
        :return: True on success
        :rtype: bool
        """

        try:
            self._callbacks_draw.remove(_cb)
            return True
        except ValueError:
            return False

    def add_callback_close(self, _cb: Callable[[], None]):
        """
        Add a callback for when a call to close is made. Callbacks are executed before the actual close execution.

        :param _cb: callback
        :type _cb: () -> None
        :return: None
        """

        self._callbacks_close.append(_cb)

    def rem_callback_close(self, _cb: Callable[[], None]):
        """
        Remove a callback for when calls to close are made.

        :param _cb: callback
        :type _cb: () -> None
        :return: True on success
        :rtype: bool
        """

        try:
            self._callbacks_close.remove(_cb)
            return True
        except ValueError:
            return False

    def save_configs(self, fp: str):
        """
        Save current state of configuration to file.

        :param fp: absolute path of file
        :type fp: str
        :return: True on success
        :rtype: bool
        """

        with open(fp, 'w') as f:
            json.dump(self.config_data, f, indent=4)

        return True

    def load_configs(self, fp: str):
        """
        Load current state of configuraiton from file.

        :param fp: absolute path of file
        :type fp: str
        :return: True on success
        :rtype: bool
        """

        if not os.path.isfile(fp):
            warnings.warn('Configuration file not found', ResourceWarning)
            return False

        with open(fp, 'r') as f:
            config_data = json.load(f)
        for k, v in config_data:
            if k in self.config_data:
                self.config_data[k] = v
        return True

    def set_config(self, key: str, val: Any, field_name: Any = None):
        """
        Set a configuration entry value.

        :param key: configuration key
        :type key: str
        :param val: configuration value
        :type val: Any or None
        :param field_name: configuration field specifier, optional
        :type field_name: str or None
        :return: True on success
        :rtype: bool
        """

        if key not in self.config_data.keys():
            warnings.warn('Configuration key not found: ' + key, RuntimeWarning)
            return False
        entry = self.config_data[key]

        if field_name is not None:
            if not isinstance(entry, dict):
                warnings.warn('Configuration key has no fields: ' + key, RuntimeWarning)
                return False

            try:
                entry[field_name] = type(entry[field_name])(val)
            except (KeyError, TypeError):
                warnings.warn(f'Could not configure: {key}, {str(field_name)}', RuntimeWarning)
                return False

        else:
            self.config_data[key] = type(entry)(val)

        return True

    def _preload_configs(self):
        """Pre-loads configuration values that are unsafe to access for any thread but the main thread"""

        self.config_data = {}

        # Pre-load basic configuration data

        for entry in self.CONFIG_ENTRIES:
            if isinstance(entry, str):
                self.config_data[entry] = CompuCellSetup.persistent_globals.configuration.getSetting(entry)
            else:
                key, field_name = entry
                val = CompuCellSetup.persistent_globals.configuration.getSetting(key, field_name)
                try:
                    self.config_data[key][field_name] = val[field_name]
                except KeyError:
                    self.config_data[key] = {field_name: val[field_name]}

        # Pre-load field configuration data

        config = CompuCellSetup.persistent_globals.configuration

        for field_name in self.field_names:
            for fk in CC3DPyGraphicsFrameClientBase.CONFIG_ENTRIES_FIELDS_BYNAME:
                val = config.getSetting(fk, field_name)
                try:
                    self.config_data[fk][field_name] = val
                except KeyError:
                    self.config_data[fk] = {field_name: val}

            for fk in CC3DPyGraphicsFrameClientBase.CONFIG_ENTRIES_FIELDS_UNIFORM:
                val = config.getSetting(fk)
                try:
                    self.config_data[fk][field_name] = val
                except KeyError:
                    self.config_data[fk] = {field_name: val}


class CC3DPyGraphicsFrameClient(CC3DPyGraphicsFrameInterface, CC3DPyGraphicsFrameClientBase):
    """
    Client for a graphics frame

    The actual user interface is provided by :class:`CC3DPyGraphicsFrameClientProxy`,
    to support serialization during service executions.

    A proxy is returned by :meth:`CC3DPyGraphicsFrameClientProxy.launch` that can be piped
    to other processes. However, the proxy is not necessary for client operations in the same process.
    """

    def __init__(self,
                 name: str = None,
                 fps: int = 60,
                 config_fp: str = None):

        self.frame_conn, frame_conn = multiprocessing.Pipe()
        """
        Connections for servicing requests from rendering process. 
        Requests of the rendering process should be made through the controller. 
        """

        CC3DPyGraphicsFrameClientBase.__init__(self, name=name, config_fp=config_fp)
        CC3DPyGraphicsFrameInterface.__init__(self, conn=self.frame_conn)

        self._frame_process = CC3DPyGraphicsFrameProcess(frame_conn=frame_conn, fps=fps, window_name=name)
        self._frame_controller = CC3DPyGraphicsFrameControlInterface(proc=self._frame_process)

        self._executor: Optional[CC3DPyGraphicsFrameClientExecutor] = None
        self._proxy: Optional[CC3DPyGraphicsFrameClientProxy] = None

    def launch(self, timeout: float = None):
        """
        Launches the graphics frame process and blocks until startup completes.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param timeout: permissible duration of launch attempt
        :type timeout: float
        :return: interface object on success, or None on failure
        :rtype: Any or None
        """

        try:
            self.conn_watcher.start()

            dest_conn, proc_conn = multiprocessing.Pipe()
            self._frame_process.confirmational_start(proc_conn)
            while not dest_conn.poll(timeout):
                pass

            self._proxy, self._executor = CC3DPyGraphicsFrameClientProxy.start(frame_client=self)
            return self._proxy

        except Exception as e:
            warnings.warn(f'Failed to launch: {str(e)}', RuntimeWarning)
            return None

    def draw(self, blocking: bool = False):
        """
        Update visualization data in rendering process.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param blocking: flag to block until update is complete
        :type blocking: bool
        :return: True on success
        :rtype: bool
        """

        try:
            [cb() for cb in self._callbacks_draw]
            self._frame_controller.draw(blocking)
            return True
        except Exception as e:
            warnings.warn(f'Failed to draw: {str(e)}', RuntimeWarning)
            return False

    def close(self):
        """
        Close the frame.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :return: True on success
        :rtype: bool
        """

        try:
            [cb() for cb in self._callbacks_close]
            self._callbacks_draw.clear()
            self._callbacks_close.clear()
            self._frame_controller.close()
            return True
        except Exception as e:
            warnings.warn(f'Failed to close: {str(e)}', RuntimeWarning)
            return False

    def get_screenshot_data(self):
        """Get screenshot data using the current frame configuration"""

        return self._frame_controller.get_screenshot_data()

    def pull_metadata(self, _field_name: str):
        """Update field metadata storage on a frame for a field"""

        self._frame_controller.pull_metadata(_field_name)

    def set_config(self, key: str, val: Any, field_name: Any = None, pull_metadata: bool = True):
        """
        Set a configuration entry value.

        :param key: configuration key
        :param val: configuration value
        :param field_name: configuration field specifier, optional
        :param pull_metadata: flag to tell underlying frame to pull field metadata when a field name is specified
        :return: True on success
        :rtype: bool
        """

        ret_val = super().set_config(key, val, field_name)
        if field_name is not None and pull_metadata and ret_val:
            ret_val = ret_val and self.pull_metadata(field_name)
        return ret_val

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        super().set_field_name(_field_name)
        self._frame_controller.set_field_name(_field_name)

    def set_plane(self, plane, pos):
        """Set the plane and position"""

        self._frame_controller.set_plane(plane, pos)

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        self._frame_controller.set_drawing_style(_style)

    def get_bounding_box_on(self) -> Optional[bool]:
        return self._frame_controller.get_bounding_box_on()

    def set_bounding_box_on(self, _val: bool):
        self._frame_controller.set_bounding_box_on(_val)

    def get_cell_borders_on(self) -> Optional[bool]:
        return self._frame_controller.get_cell_borders_on()

    def set_cell_borders_on(self, _val: bool):
        self._frame_controller.set_cell_borders_on(_val)

    def get_cell_glyphs_on(self) -> Optional[bool]:
        return self._frame_controller.get_cell_glyphs_on()

    def set_cell_glyphs_on(self, _val: bool):
        self._frame_controller.set_cell_glyphs_on(_val)

    def get_cells_on(self) -> Optional[bool]:
        return self._frame_controller.get_cells_on()

    def set_cells_on(self, _val: bool):
        self._frame_controller.set_cells_on(_val)

    def get_cluster_borders_on(self) -> Optional[bool]:
        return self._frame_controller.get_cluster_borders_on()

    def set_cluster_borders_on(self, _val: bool):
        self._frame_controller.set_cluster_borders_on(_val)

    def get_fpp_links_on(self) -> Optional[bool]:
        return self._frame_controller.get_fpp_links_on()

    def set_fpp_links_on(self, _val: bool):
        self._frame_controller.set_fpp_links_on(_val)

    def get_lattice_axes_labels_on(self) -> Optional[bool]:
        return self._frame_controller.get_lattice_axes_labels_on()

    def set_lattice_axes_labels_on(self, _val: bool):
        self._frame_controller.set_lattice_axes_labels_on(_val)

    def get_lattice_axes_on(self) -> Optional[bool]:
        return self._frame_controller.get_lattice_axes_on()

    def set_lattice_axes_on(self, _val: bool):
        self._frame_controller.set_lattice_axes_on(_val)

    def get_window_size(self) -> Tuple[int, int]:
        return self._frame_controller.get_window_size()

    def set_window_size(self, _window_size: Tuple[int, int]):
        self._frame_controller.set_window_size(_window_size)

    def np_img_data(self,
                    scale: Union[int, Tuple[int, int]] = None,
                    transparent_background: bool = False):
        """
        Get image data as numpy data.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: image array data
        :rtype: numpy.array
        """

        return self._frame_controller.np_img_data(scale=scale, transparent_background=transparent_background)

    def save_img(self,
                 file_path: str,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):
        """
        Save a window to file.

        Supported image types are .eps, .jpg, .jpeg, .pdf, .png, .svg.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param file_path: absolute path to save the image
        :type file_path: str
        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: None
        """

        self._frame_controller.save_img(file_path=file_path,
                                        scale=scale,
                                        transparent_background=transparent_background)

    def _get_field_names(self) -> List[str]:
        field_names = CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector()
        return list(field_names)

    @property
    def field_names(self) -> List[str]:
        """Current available field names"""

        return self._get_field_names()

    def _standard_bsd(self) -> BasicSimulationData:
        bsd = BasicSimulationData()
        bsd.fieldDim = CompuCellSetup.persistent_globals.simulator.getPotts().getCellFieldG().getDim()
        bsd.numberOfSteps = CompuCellSetup.persistent_globals.simulator.getNumSteps()
        return bsd

    # Request services for rendering process

    def _service_get_config(self, _key, field_name=None):

        # If config data hasn't been loaded, then try to load it and hope we're in the main thread
        if self.config_data is None:
            self._preload_configs()

        try:
            if field_name is None:
                try:
                    return self.config_data[_key]
                except KeyError:
                    self.config_data[_key] = CONFIG_DEFAULT_SETTINGS[_key]
                    return self.config_data[_key]
            else:
                if _key not in self.config_data.keys():
                    self.config_data[_key] = {field_name: CONFIG_DEFAULT_SETTINGS[_key]}
                try:
                    return self.config_data[_key][field_name]
                except KeyError:
                    self.config_data[_key][field_name] = CONFIG_DEFAULT_SETTINGS[_key]
                    return self.config_data[_key][field_name]
        except KeyError:
            raise RuntimeError('Failed fetching configuration:', _key, field_name)

    def _service_get_concentration_field_names(self) -> List[str]:
        return self.field_names

    def _service_get_fields_to_create(self) -> Dict[str, str]:
        field_dict = CompuCellSetup.persistent_globals.field_registry.get_fields_to_create_dict()
        return {field_name: field_adapter.field_type for field_name, field_adapter in field_dict.items()}

    def _service_get_streamed_data(self) -> FieldStreamerDataPy:
        field_storage = CompuCellSetup.persistent_globals.persistent_holder['field_storage']
        field_writer = FieldWriterCML()
        field_writer.init(CompuCellSetup.persistent_globals.simulator)
        field_writer.setFieldStorage(field_storage)

        field_writer.addCellFieldForOutput()

        if self._field_name != 'Cell_Field':
            if not field_writer.addFieldForOutput(self._field_name):
                warnings.warn(f'Failed to add field for output: {self._field_name}', RuntimeWarning)

        return FieldStreamerDataPy.from_base(FieldStreamer.dump(field_writer))

    def _service_get_bsd(self) -> BasicSimulationData:
        return self._standard_bsd()

    def _service_get_lattice_type_str(self) -> str:
        lattice_type_str = simulation_utils.extract_lattice_type()
        if lattice_type_str not in list(Configuration.LATTICE_TYPES.keys()):
            lattice_type_str = 'Square'
        return lattice_type_str

    def _service_init_field_types(self):

        field_types = {}
        if CompuCellSetup.persistent_globals.simulator is not None:

            field_types["Cell_Field"] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[CELL_FIELD]

            # get concentration fields from simulator
            for fieldName in CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector():
                field_types[fieldName] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[CON_FIELD]

            # inserting extra scalar fields managed from Python script
            field_dict = CompuCellSetup.persistent_globals.field_registry.get_fields_to_create_dict()
            names_types = {field_name: field_adapter.field_type for field_name, field_adapter in field_dict.items()}
            for field_name, field_type in names_types.items():
                field_types[field_name] = FIELD_NUMBER_TO_FIELD_TYPE_MAP[field_type]

        return field_types

    @property
    def field_name(self) -> str:
        return self.get_field_name()

    @field_name.setter
    def field_name(self, _field_name):
        self.set_field_name(_field_name)

    @property
    def min_range_fixed(self) -> bool:
        key = 'MinRangeFixed'

        try:
            return self._service_get_config(key, field_name=self.field_name)
        except KeyError:
            return False

    @min_range_fixed.setter
    def min_range_fixed(self, _val: bool) -> None:
        key = 'MinRangeFixed'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def min_range(self) -> Optional[float]:
        key = 'MinRange'

        try:
            return self._service_get_config(key, field_name=self.field_name)
        except KeyError:
            return None

    @min_range.setter
    def min_range(self, _val: float):
        key = 'MinRange'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def max_range_fixed(self) -> bool:
        key = 'MaxRangeFixed'

        try:
            return self._service_get_config(key, field_name=self.field_name)
        except KeyError:
            return False

    @max_range_fixed.setter
    def max_range_fixed(self, _val: bool):
        key = 'MaxRangeFixed'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def max_range(self) -> Optional[float]:
        key = 'MaxRange'

        try:
            return self._service_get_config(key, field_name=self.field_name)
        except KeyError:
            return None

    @max_range.setter
    def max_range(self, _val: float):
        key = 'MaxRange'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    bounding_box_on = property(fget=get_bounding_box_on, fset=set_bounding_box_on)
    cell_borders_on = property(fget=get_cell_borders_on, fset=set_cell_borders_on)
    cell_glyphs_on = property(fget=get_cell_glyphs_on, fset=set_cell_glyphs_on)
    cells_on = property(fget=get_cells_on, fset=set_cells_on)
    cluster_borders_on = property(fget=get_cluster_borders_on, fset=set_cluster_borders_on)
    fpp_links_on = property(fget=get_fpp_links_on, fset=set_fpp_links_on)
    lattice_axes_labels_on = property(fget=get_lattice_axes_labels_on, fset=set_lattice_axes_labels_on)
    lattice_axes_on = property(fget=get_lattice_axes_on, fset=set_lattice_axes_on)
    window_size = property(fget=get_window_size, fset=set_window_size)


class CC3DPyGraphicsFrameClientProxyMsg:

    def __init__(self, method: str, args, kwargs):

        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.returns = False


class CC3DPyGraphicsFrameClientExecutor(threading.Thread):
    """
    Executor for :class:`CC3DPyGraphicsFrameClientProxy`.

    Supports serialization for server-side interface during service execution.
    """

    def __init__(self,
                 frame_client: CC3DPyGraphicsFrameClient,
                 proxy_conn: Connection):

        super().__init__(daemon=True)

        self.frame_client = frame_client
        self._proxy_conn = proxy_conn

    def run(self) -> None:
        while not self._proxy_conn.closed:
            if self._proxy_conn.poll():
                msg = self._proxy_conn.recv()
                if msg is None:
                    self._proxy_conn.close()
                    return

                msg: CC3DPyGraphicsFrameClientProxyMsg
                return_val = getattr(self.frame_client, msg.method)(*msg.args, **msg.kwargs)
                if msg.returns:
                    self._proxy_conn.send(return_val)

        self._proxy_conn.close()


class CC3DPyGraphicsFrameClientProxy:
    """
    Proxy for :class:`CC3DPyGraphicsFrameClient`.

    Supports serialization for client-side interface during service execution.
    """

    def __init__(self, executor_conn: Connection):

        self._executor_conn = executor_conn

    @staticmethod
    def start(frame_client: CC3DPyGraphicsFrameClient):

        proxy_conn, executor_conn = multiprocessing.Pipe()
        proxy = CC3DPyGraphicsFrameClientProxy(executor_conn)
        executor = CC3DPyGraphicsFrameClientExecutor(frame_client=frame_client,
                                                     proxy_conn=proxy_conn)
        executor.start()
        return proxy, executor

    def _process_msg(self, msg: str, *args, **kwargs):
        if self._executor_conn.closed:
            warnings.warn('Frame proxy has been disconnected', RuntimeWarning)
            return

        self._executor_conn.send(CC3DPyGraphicsFrameClientProxyMsg(msg, args, kwargs))

    def _process_ret_msg(self, msg: str, *args, **kwargs):
        if self._executor_conn.closed:
            warnings.warn('Frame proxy has been disconnected', RuntimeWarning)
            return None

        msg = CC3DPyGraphicsFrameClientProxyMsg(msg, args, kwargs)
        msg.returns = True
        self._executor_conn.send(msg)
        while not self._executor_conn.poll():
            pass
        return self._executor_conn.recv()

    def launch(self, timeout: float = None):
        """
        Launches the graphics frame process and blocks until startup completes.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param timeout: permissible duration of launch attempt
        :type timeout: float
        :return: interface object on success, or None on failure
        :rtype: Any or None
        """

        self._process_msg('launch', timeout=timeout)
        return self

    def draw(self, blocking: bool = False):
        """
        Update visualization data in rendering process.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param blocking: flag to block until update is complete
        :type blocking: bool
        :return: True on success
        :rtype: bool
        """

        return self._process_ret_msg('draw', blocking=blocking)

    def close(self):
        """
        Close the frame.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :return: True on success
        :rtype: bool
        """

        rval = self._process_ret_msg('close')
        self._executor_conn.send(None)
        return rval

    def get_screenshot_data(self):
        """Get screenshot data using the current frame configuration"""

        return self._process_ret_msg('get_screenshot_data')

    def pull_metadata(self, _field_name: str):
        """Update field metadata storage on a frame for a field"""

        return self._process_msg('pull_metadata', _field_name)

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        return self._process_msg('set_field_name', _field_name)

    def set_plane(self, plane, pos):
        """Set the plane and position"""

        return self._process_msg('set_plane', plane=plane, pos=pos)

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        return self._process_msg('set_drawing_style', _style)

    def np_img_data(self,
                    scale: Union[int, Tuple[int, int]] = None,
                    transparent_background: bool = False):
        """
        Get image data as numpy data.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: image array data
        :rtype: numpy.array
        """

        return self._process_ret_msg('np_img_data', scale=scale, transparent_background=transparent_background)

    def save_img(self,
                 file_path: str,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):
        """
        Save a window to file.

        Supported image types are .eps, .jpg, .jpeg, .pdf, .png, .svg.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param file_path: absolute path to save the image
        :type file_path: str
        :param scale: image scale
        :type scale: int or (int, int) or None
        :param transparent_background: flag to generate with a transparent background
        :type transparent_background: bool
        :return: None
        """

        return self._process_msg('save_img',
                                 file_path=file_path,
                                 scale=scale,
                                 transparent_background=transparent_background)

    @property
    def field_names(self) -> List[str]:
        """Current available field names"""

        return self._process_ret_msg('_get_field_names')

    def save_configs(self, fp: str):
        """
        Save current state of configuration to file.

        :param fp: absolute path of file
        :type fp: str
        :return: True on success
        :rtype: bool
        """

        return self._process_ret_msg('save_configs', fp=fp)

    def load_configs(self, fp: str):
        """
        Load current state of configuraiton from file.

        :param fp: absolute path of file
        :type fp: str
        :return: True on success
        :rtype: bool
        """

        return self._process_ret_msg('load_configs', fp=fp)

    def set_config(self, key: str, val: Any, field_name: Any = None, pull_metadata: bool = True):
        """
        Set a configuration entry value.

        :param key: configuration key
        :param val: configuration value
        :param field_name: configuration field specifier, optional
        :param pull_metadata: flag to tell underlying frame to pull field metadata when a field name is specified
        :return: True on success
        :rtype: bool
        """

        return self._process_ret_msg('set_config', key=key, val=val, field_name=field_name, pull_metadata=pull_metadata)

    @property
    def field_name(self) -> str:
        return self._process_ret_msg('get_field_name')

    @field_name.setter
    def field_name(self, _field_name):
        self.set_field_name(_field_name)

    @property
    def min_range_fixed(self) -> bool:
        key = 'MinRangeFixed'

        try:
            return self._process_ret_msg('_service_get_config', key, field_name=self.field_name)
        except KeyError:
            return False

    @min_range_fixed.setter
    def min_range_fixed(self, _val: bool) -> None:
        key = 'MinRangeFixed'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def min_range(self) -> Optional[float]:
        key = 'MinRange'

        try:
            return self._process_ret_msg('_service_get_config', key, field_name=self.field_name)
        except KeyError:
            return None

    @min_range.setter
    def min_range(self, _val: float):
        key = 'MinRange'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def max_range_fixed(self) -> bool:
        key = 'MaxRangeFixed'

        try:
            return self._process_ret_msg('_service_get_config', key, field_name=self.field_name)
        except KeyError:
            return False

    @max_range_fixed.setter
    def max_range_fixed(self, _val: bool):
        key = 'MaxRangeFixed'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    @property
    def max_range(self) -> Optional[float]:
        key = 'MaxRange'

        try:
            return self._process_ret_msg('_service_get_config', key, field_name=self.field_name)
        except KeyError:
            return None

    @max_range.setter
    def max_range(self, _val: float):
        key = 'MaxRange'

        self.set_config(key=key, val=_val, field_name=self.field_name)

    def get_range_fixed(self) -> Tuple[bool, bool]:
        """Get whether both concentration limits are fixed"""

        return self.min_range_fixed, self.max_range_fixed

    def set_range_fixed(self, range_min: bool = None, range_max: bool = None):
        """Set whether one or both concentration limits are fixed"""

        if range_min is not None:
            self.min_range = range_min
        if range_max is not None:
            self.max_range = range_max

    def get_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the concentration limits, if any."""

        return self.min_range, self.max_range

    def set_range(self, range_min: float = None, range_max: float = None):
        """Set one or both concentration limits. When setting a range value, the range is automatically fixed."""

        if range_min is not None:
            self.min_range = range_min
            self.min_range_fixed = True
        if range_max is not None:
            self.max_range = range_max
            self.max_range_fixed = True

    range = property(fget=get_range, fset=set_range)

    @property
    def bounding_box_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_bounding_box_on')

    @bounding_box_on.setter
    def bounding_box_on(self, _val: bool):
        self._process_msg('set_bounding_box_on', _val)

    @property
    def cell_borders_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_cell_borders_on')

    @cell_borders_on.setter
    def cell_borders_on(self, _val: bool):
        self._process_msg('set_cell_borders_on', _val)

    @property
    def cell_glyphs_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_cell_glyphs_on')

    @cell_glyphs_on.setter
    def cell_glyphs_on(self, _val: bool):
        self._process_msg('set_cell_glyphs_on', _val)

    @property
    def cells_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_cells_on')

    @cells_on.setter
    def cells_on(self, _val: bool):
        self._process_msg('set_cells_on', _val)

    @property
    def cluster_borders_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_cluster_borders_on')

    @cluster_borders_on.setter
    def cluster_borders_on(self, _val: bool):
        self._process_msg('set_cluster_borders_on', _val)

    @property
    def fpp_links_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_fpp_links_on')

    @fpp_links_on.setter
    def fpp_links_on(self, _val: bool):
        self._process_msg('set_fpp_links_on', _val)

    @property
    def lattice_axes_labels_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_lattice_axes_labels_on')

    @lattice_axes_labels_on.setter
    def lattice_axes_labels_on(self, _val: bool):
        self._process_msg('set_lattice_axes_labels_on', _val)

    @property
    def lattice_axes_on(self) -> Optional[bool]:
        return self._process_ret_msg('get_lattice_axes_on')

    @lattice_axes_on.setter
    def lattice_axes_on(self, _val: bool):
        self._process_msg('set_lattice_axes_on', _val)

    @property
    def window_size(self) -> Optional[Tuple[int, int]]:
        return self._process_ret_msg('get_window_size')

    @window_size.setter
    def window_size(self, _val: Tuple[int, int]):
        self._process_ret_msg('set_window_size', _val)
