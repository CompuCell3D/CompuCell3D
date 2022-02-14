"""
Defines features for interactive visualization for use with CC3D simservice applications.
"""

# todo: add CC3D logo to window title icon
# todo: add target field name display
# todo: handle settings values for individual fields (i.e., create defaults when not in pre-fetched data)
# todo: disable built-in key commands for closing render windows
# todo: add support for additional plots (e.g., tracking fields)

import json
import multiprocessing
import numpy
import os
import threading
from typing import Any, Callable, Dict, List
import warnings
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOExportPython import vtkGL2PSExporter
from vtkmodules.vtkIOImage import vtkJPEGWriter, vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor, vtkRenderWindow, vtkWindowToImageFilter
from vtkmodules.util import numpy_support

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import simulation_utils
from cc3d.core.BasicSimulationData import BasicSimulationData
from cc3d.core import Configuration
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.core.GraphicsUtils.FieldStreamer import FieldStreamer
from cc3d.core.GraphicsUtils.GraphicsFrame import GraphicsFrame
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrameIO import *
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object
from cc3d.cpp.CompuCell import Dim3D
from cc3d.cpp.PlayerPython import FieldExtractorCML

from .prototypes.FieldWriterCML import FieldWriterCML


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

        streamer: FieldStreamer = MsgGetStreamedData.request(self._conn, True)
        if streamer is None:
            return False
        self._streamed_points = FieldStreamer.loadp(streamer.data)
        self.field_extractor.setFieldDim(Dim3D(*streamer.data.field_dim))
        self.field_extractor.setSimulationData(extract_address_int_from_vtk_object(self._streamed_points))
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

        # noinspection PyUnresolvedReferences
        self.style.SetCurrentRenderer(self.gd.get_renderer())

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

        self.metadata_data_dict = {k: {} for k in self.metadata_fetcher_dict.keys()}

        try:
            self.metadata_fetcher_dict = self.metadata_fetcher_dict_copy.copy()
            has_prefetched = True
        except (AttributeError, KeyError):
            has_prefetched = False

        for field_name, field_type in self.fieldTypes.items():

            self.metadata_data_dict[field_type][field_name] = self.get_metadata(field_name=field_name,
                                                                                field_type=field_type)

        if has_prefetched:
            self.metadata_fetcher_dict = {k: self.get_metadata_prefetched for k in self.metadata_fetcher_dict.keys()}

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

    def get_concentration_field_names(self) -> List[str]:
        """Get concentration field names from remote source"""
        return MsgGetConcFieldNames.request(self.conn, True)

    def get_fields_to_create(self) -> Dict[str, str]:
        """Get names of fields to create from remote source"""
        return MsgGetFieldsToCreate.request(self.conn, True)

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        if _field_name not in self.fieldTypes.keys():
            warnings.warn('Field name not known: ' + _field_name, RuntimeWarning)
            return
        self.field_name = _field_name

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
        self._process_message(ControlMessageSaveImame(file_path=file_path,
                                                      scale=scale,
                                                      transparent_background=transparent_background))


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

    # todo: implement smarter CC3D default configuration data

    CONFIG_DEFAULT_VALUES: List[Tuple[str, Union[Any, Dict[Any, Any]]]] = [
        ('AxesColor', [255, 255, 255]),
        ('BorderColor', [255, 255, 0]),
        ('BoundingBoxColor', [255, 255, 255]),
        ('BoundingBoxOn', True),
        ('CellBordersOn', True),
        ('CellGlyphsOn', False),
        ('CellsOn', True),
        ('ClusterBorderColor', [0, 0, 255]),
        ('ClusterBordersOn', False),
        ('ContourColor', [255, 255, 255]),
        ('FPPLinksColor', [255, 255, 255]),
        ('FPPLinksOn', False),
        ('ShowAxes', True),
        ('ShowHorizontalAxesLabels', True),
        ('ShowVerticalAxesLabels', True),
        ('TypeColorMap', {0: [0, 0, 0],
                          1: [0, 255, 0],
                          2: [0, 0, 255],
                          3: [255, 0, 0],
                          4: [128, 128, 0],
                          5: [192, 192, 192],
                          6: [255, 0, 255],
                          7: [0, 0, 128],
                          8: [0, 255, 255],
                          9: [0, 128, 0],
                          10: [255, 255, 255]}),
        ('WindowColor', [0, 0, 0])
    ]

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
    def field_names(self) -> Optional[List[str]]:
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
    """Client for a graphics frame"""

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

            return self

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
            self._frame_controller.close()
            return True
        except Exception as e:
            warnings.warn(f'Failed to close: {str(e)}', RuntimeWarning)
            return False

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

    @property
    def field_names(self) -> Optional[List[str]]:
        """Current available field names"""

        field_names = CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector()
        return list(field_names)

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
                return self.config_data[_key]
            else:
                return self.config_data[_key][field_name]
        except KeyError:
            raise RuntimeError('Failed fetching configuration:', _key, field_name)

    def _service_get_concentration_field_names(self) -> List[str]:
        return self.field_names

    def _service_get_fields_to_create(self) -> Dict[str, str]:
        field_dict = CompuCellSetup.persistent_globals.field_registry.get_fields_to_create_dict()
        return {field_name: field_adapter.field_type for field_name, field_adapter in field_dict.items()}

    def _service_get_streamed_data(self) -> FieldStreamer:
        field_storage = CompuCellSetup.persistent_globals.persistent_holder['field_storage']
        field_writer = FieldWriterCML()
        field_writer.init(CompuCellSetup.persistent_globals.simulator)
        field_writer.setFieldStorage(field_storage)

        field_writer.addCellFieldForOutput()

        if self._field_name != 'Cell_Field':
            field_writer.addFieldForOutput(self._field_name)

        return FieldStreamer(field_writer=field_writer)

    def _service_get_bsd(self) -> BasicSimulationData:
        return self._standard_bsd()

    def _service_get_lattice_type_str(self) -> str:
        lattice_type_str = simulation_utils.extract_lattice_type()
        if lattice_type_str not in list(Configuration.LATTICE_TYPES.keys()):
            lattice_type_str = 'Square'
        return lattice_type_str
