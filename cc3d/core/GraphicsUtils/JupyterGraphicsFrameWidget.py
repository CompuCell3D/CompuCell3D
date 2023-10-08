"""
Defines features for interactive visualization for use with CC3D simservice applications in a Jupyter notebook
"""

from typing import Optional, Union, Tuple, List, Any, Dict
import warnings
from weakref import ref

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkPolyLine
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkRenderWindowInteractor, vtkRenderWindow, vtkPolyDataMapper2D, \
    vtkCoordinate, vtkTextActor

import cc3d.CompuCellSetup
from cc3d.core.GraphicsOffScreen.primitives import Color
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.core.GraphicsUtils.GraphicsFrame import GraphicsFrame, default_field_label
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame import (
    CC3DPyGraphicsFrameClientBase, CC3DPyInteractorStyle, np_img_data, save_img
)


# Test for IPython
try:
    get_ipython
    __has_interactive__ = True
    from ipyvtklink.viewer import ViewInteractiveWidget
    from IPython.display import display
    from ipywidgets import HBox, VBox
    from cc3d.core.GraphicsUtils.JupyterControlPanel import JupyterControlPanel, JupyterSettingsPanel
except NameError:
    __has_interactive__ = False
    ViewInteractiveWidget = object
    JupyterSettingsPanel = object
    JupyterControlPanel = object


class CC3DJupyterGraphicsConfig:
    """Configuration hook to request settings data between processes"""

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

    def __init__(self, config_data=None):
        if config_data:
            self.config_data = config_data
        else:
            self.config_data = {}
            config = cc3d.CompuCellSetup.persistent_globals.configuration
            # Pre-load basic configuration data
            for entry in self.CONFIG_ENTRIES:
                if isinstance(entry, str):
                    self.config_data[entry] = config.getSetting(entry)
                else:
                    key, field_name = entry
                    val = config.getSetting(key, field_name)
                    try:
                        self.config_data[key][field_name] = val[field_name]
                    except KeyError:
                        self.config_data[key] = {field_name: val[field_name]}

            # Pre-load field configuration data
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

    @property
    def field_names(self) -> Optional[List[str]]:
        """Current available field names"""
        field_names = cc3d.CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector()
        return list(field_names)

    def getSetting(self, key, *args, **kwargs):
        """Get setting value from remote source"""
        if args and len(args) > 0:
            return self.config_data[key][args[0]]
        return self.config_data[key]

    def setSetting(self, key, value):
        """Set value"""
        self.config_data[key] = value


class JupyterGraphicsFrame(GraphicsFrame):
    """
    Interactive graphics in a Jupyter notebook
    """

    def __init__(self):

        if not __has_interactive__:
            raise RuntimeError('Interactive frame launched outside of an interactive environment.')

        # Validate current simulation state
        pg = cc3d.CompuCellSetup.persistent_globals
        if pg.simulator is None:
            raise RuntimeError('Simulator not set')
        elif pg.screenshot_manager is None:
            raise RuntimeError('Screenshot manager not set')
        try:
            field_extractor = pg.persistent_holder['field_extractor']
        except KeyError:
            raise RuntimeError('Field extractor not set')

        self.style: Optional[CC3DPyInteractorStyle] = None
        self._border_actor: Optional[vtkActor2D] = None
        self.border_color_unsel = [0.5, 0.5, 0.5]
        self.border_color_sel = [0.0, 0.6, 1.0]
        self.border_selected = False
        self._field_label_actor: Optional[vtkTextActor] = None

        generic_drawer = GenericDrawer(boundary_strategy=pg.simulator.getBoundaryStrategy())
        generic_drawer.set_pixelized_cartesian_scene(pg.configuration.getSetting("PixelizedCartesianFields"))
        generic_drawer.set_field_extractor(field_extractor=field_extractor)

        super().__init__(generic_drawer=generic_drawer,
                         current_bsd=pg.screenshot_manager.bsd,
                         config_hook=CC3DJupyterGraphicsConfig())

        # Initialize options
        self._set_vars_from_config()

        self.colormap = self.config.getSetting('TypeColorMap')

        # Initialize initial rendered state

        renderer = self.gd.get_renderer()
        # noinspection PyUnresolvedReferences
        self.style.SetCurrentRenderer(renderer)
        renderer.AddViewProp(self.border_actor)
        renderer.AddActor(self.field_label_actor)
        self.draw()
        self.reset_camera()
        self.init_field_types()
        self.Render()

    def Render(self):
        self._field_label_actor.SetInput(self.field_name)
        super().Render()

    @property
    def border_actor(self):
        """Actor implementing the drawn frame border"""

        if self._border_actor is not None:
            return self._border_actor

        border_points = vtkPoints()
        border_points.SetNumberOfPoints(4)
        border_points.InsertPoint(0, 0.0, 0.0, 0.0)
        border_points.InsertPoint(1, 1.0, 0.0, 0.0)
        border_points.InsertPoint(2, 1.0, 1.0, 0.0)
        border_points.InsertPoint(3, 0.0, 1.0, 0.0)

        border_cells = vtkCellArray()
        border_cells.Initialize()
        border_lines = vtkPolyLine()

        border_lines.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            border_lines.GetPointIds().SetId(i, i)
        border_cells.InsertNextCell(border_lines)

        polydata = vtkPolyData()
        polydata.Initialize()
        polydata.SetPoints(border_points)
        polydata.SetLines(border_cells)

        view_coords = vtkCoordinate()
        view_coords.SetCoordinateSystemToNormalizedViewport()

        polymapper = vtkPolyDataMapper2D()
        polymapper.SetInputData(polydata)
        polymapper.SetTransformCoordinate(view_coords)

        self._border_actor = vtkActor2D()
        self._border_actor.SetMapper(polymapper)
        self._border_actor.GetProperty().SetColor(self.border_color_unsel)
        self._border_actor.GetProperty().SetLineWidth(10.0)
        return self._border_actor

    @property
    def border_color(self) -> List[float]:
        """Color of drawn frame border"""

        return self.border_actor.GetProperty().GetColor()

    @border_color.setter
    def border_color(self, _color: List[float]):

        self.border_actor.GetProperty().SetColor(_color)

    @property
    def border_width(self) -> float:
        """Widget of drawn frame border"""

        return self.border_actor.GetProperty().GetLineWidth()

    @border_width.setter
    def border_width(self, _width: float):

        self.border_actor.GetProperty().SetLineWidth(_width)

    @property
    def border_selected(self) -> bool:
        """Flag signifying whether the drawn border shows a selected state"""

        return self._border_selected

    @border_selected.setter
    def border_selected(self, _border_selected: bool):

        self._border_selected = _border_selected
        self.border_color = self.border_color_sel if self._border_selected else self.border_color_unsel

    @property
    def field_label_actor(self):
        """Field label actor of the frame. Text is synchronized with field name"""

        if self._field_label_actor is None:
            self._field_label_actor = default_field_label()
        return self._field_label_actor

    def _set_vars_from_config(self):
        self.bounding_box_on = self.config.getSetting('BoundingBoxOn')
        self.cell_borders_on = self.config.getSetting('CellBordersOn')
        self.cell_glyphs_on = self.config.getSetting('CellGlyphsOn')
        self.cells_on = self.config.getSetting('CellsOn')
        self.cluster_borders_on = self.config.getSetting('ClusterBordersOn')
        self.fpp_links_on = self.config.getSetting('FPPLinksOn')
        self.lattice_axes_labels_on = self.config.getSetting('ShowAxes')
        self.lattice_axes_on = self.config.getSetting('ShowHorizontalAxesLabels') or self.config.getSetting(
            'ShowVerticalAxesLabels')

    def get_vtk_window(self):
        """
        Get an initialized vtk window and window interactor.

        Implementation of :class:`GraphicsFrame` interface.
        """

        # noinspection PyArgumentList
        interactor = vtkRenderWindowInteractor()

        renWin = vtkRenderWindow()
        renWin.SetOffScreenRendering(1)
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

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according tpo the dimension of the visualization.

        Override of :class:`GraphicsFrame` interface.

        :param _style:{str} '2D' or '3D'
        :return: None
        """
        super().set_drawing_style(_style)

        self.style.can_rotate = _style == '3D'

    def on_cell_type_color(self):
        """Handle when a cell type color changes"""

        scene_metadata = {'TypeColorMap': self.config.getSetting('TypeColorMap')}
        lut = self.gd.draw_model_2D.generate_cell_type_lookup_table(scene_metadata, True)
        self.gd.draw_model_2D.celltypeLUT = lut

        lut = self.gd.draw_model_3D.generate_cell_type_lookup_table(scene_metadata, True)
        self.gd.draw_model_3D.celltypeLUT = lut

    @property
    def min_range_fixed(self) -> bool:
        try:
            return self.config.getSetting('MinRangeFixed')[self.field_name]
        except KeyError:
            return False

    @min_range_fixed.setter
    def min_range_fixed(self, _val: bool) -> None:
        key = 'MinRangeFixed'

        setting = self.config.getSetting(key)
        if self.field_name in setting.keys():
            setting[self.field_name] = _val
            self.config.setSetting(key, setting)

    @property
    def min_range(self) -> Optional[float]:
        return self.config.getSetting('MinRange').get(self.field_name)

    @min_range.setter
    def min_range(self, _val: float):
        key = 'MinRange'

        setting = self.config.getSetting(key)
        if self.field_name in setting.keys():
            setting[self.field_name] = _val
            self.config.setSetting(key, setting)

    @property
    def max_range_fixed(self) -> bool:
        try:
            return self.config.getSetting('MaxRangeFixed')[self.field_name]
        except KeyError:
            return False

    @max_range_fixed.setter
    def max_range_fixed(self, _val: bool):
        key = 'MaxRangeFixed'

        setting = self.config.getSetting(key)
        if self.field_name in setting.keys():
            setting[self.field_name] = _val
            self.config.setSetting(key, setting)

    @property
    def max_range(self) -> Optional[float]:
        return self.config.getSetting('MaxRange').get(self.field_name)

    @max_range.setter
    def max_range(self, _val: float):
        key = 'MaxRange'

        setting = self.config.getSetting(key)
        if self.field_name in setting.keys():
            setting[self.field_name] = _val
            self.config.setSetting(key, setting)


class CC3DViewInteractiveWidget(ViewInteractiveWidget):
    """:class:`ViewInteractiveWidget` that shares interactions"""

    def __init__(self, frame: JupyterGraphicsFrame, *args, **kwargs):

        self._frame = ref(frame)
        self._camera = None
        self._forwarding = False
        self._partners: List[CC3DViewInteractiveWidget] = []

        super().__init__(*args, **kwargs)

    def update_canvas(self, force_render=True, quality=75):
        if self._forwarding:
            return
        self._forwarding = True

        for p in self._partners:
            p.update_canvas(force_render, quality)

        self._forwarding = False
        super().update_canvas(force_render, quality)

    def sync_cameras(self, interactor):
        """Synchronize all cameras"""

        interactor: CC3DViewInteractiveWidget

        if interactor in self._partners or interactor is self:
            return

        if self._forwarding:
            return
        self._forwarding = True

        frame = self._frame()
        frame_p = interactor._frame()
        if frame is None or frame_p is None:
            warnings.warn('Could not synchronize cameras')
            return

        frame: JupyterGraphicsFrame
        frame_p: JupyterGraphicsFrame

        if self._camera is None:
            self._camera = frame.gd.get_active_camera()
        frame.gd.get_renderer().SetActiveCamera(frame_p.gd.get_active_camera())

        for p in self._partners:
            p.sync_cameras(interactor)

        interactor._partners.append(self)
        self._partners.append(interactor)

        self._forwarding = False

    def unsync_camera(self):
        """Unsynchronize all cameras"""

        for p in self._partners:
            p._partners.remove(self)
        self._partners.clear()

        if self._camera is None:
            return

        frame = self._frame()
        if frame is None:
            warnings.warn('Could not unsychronize camera')
        frame: JupyterGraphicsFrame

        frame.gd.get_renderer().SetActiveCamera(self._camera)
        self._camera = None


class JupyterGraphicsFrameClient(CC3DPyGraphicsFrameClientBase):
    """Client for a Jupyter graphics frame"""

    def __init__(self,
                 name: str = None,
                 config_fp: str = None):

        self.frame: Optional[JupyterGraphicsFrame] = None
        self.widget: Optional[CC3DViewInteractiveWidget] = None

        super().__init__(name=name, config_fp=config_fp)

    def launch(self, timeout: float = None):
        """
        Launches the graphics frame process and blocks until startup completes.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param timeout: permissible duration of launch attempt
        :type timeout: float
        :return: interface object on success, or None on failure
        :rtype: Any or None
        """

        self.frame = JupyterGraphicsFrame()
        self.frame.gd.get_renderer().ResetCamera()

        self.widget = CC3DViewInteractiveWidget(frame=self.frame,
                                                render_window=self.frame.renWin)
        return self

    def show(self):
        display(self.widget)

    def draw(self, blocking: bool = False):
        """
        Update visualization data in rendering process.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :param blocking: flag to block until update is complete
        :type blocking: bool
        :return: True on success
        :rtype: bool
        """

        self.frame.draw()
        self.widget: CC3DViewInteractiveWidget
        self.widget.update_canvas()

    def close(self):
        """
        Close the frame. Does nothing.

        Implementation of :class:`CC3DPyGraphicsFrameClientBase` interface.

        :return: True on success
        :rtype: bool
        """

        return True

    def np_img_data(self, scale: Union[int, Tuple[int, int]] = None, transparent_background: bool = False):
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

        return np_img_data(ren_win=self.frame.renWin, scale=scale, transparent_background=transparent_background)

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

        return save_img(ren_win=self.frame.renWin,
                        file_path=file_path,
                        scale=scale,
                        transparent_background=transparent_background)

    def set_drawing_style(self, _style):
        """
        Function that wires-up the widget to behave according to the dimension of the visualization

        :param _style:{str} '2D' or '3D'
        :return: None
        """

        self.frame.set_drawing_style(_style)

        self._update()

    def set_plane(self, plane, pos=0):
        """Set the plane and position"""

        self.frame.currentProjection = plane
        self.frame.projection_position = pos

        if self.frame.currentProjection == 'xy':
            if pos > self.frame.xyMaxPlane:
                pos = self.frame.xyMaxPlane
            self.frame.xyPlane = pos

        elif self.frame.currentProjection == 'xz':
            if pos > self.frame.xzMaxPlane:
                pos = self.frame.xzMaxPlane
            self.frame.xzPlane = pos

        elif self.frame.currentProjection == 'yz':
            if pos > self.frame.yzMaxPlane:
                pos = self.frame.yzMaxPlane
            self.frame.yzPlane = pos

        self.frame.set_plane(self.frame.currentProjection, pos)
        self._update()

    def get_range_fixed(self) -> Tuple[bool, bool]:
        """Get whether both concentration limits are fixed"""

        return self.frame.min_range_fixed, self.frame.max_range_fixed

    def set_range_fixed(self, range_min: bool = None, range_max: bool = None):
        """Set whether one or both concentration limits are fixed"""

        if range_min is not None:
            self.frame.min_range = range_min
        if range_max is not None:
            self.frame.max_range = range_max
        self._update()

    @property
    def range_fixed(self) -> Tuple[bool, bool]:
        return self.get_range_fixed()

    @range_fixed.setter
    def range_fixed(self, _range_fixed: Tuple[bool, bool]):
        self.set_range_fixed(*_range_fixed)

    @property
    def min_range_fixed(self) -> bool:
        return self.range_fixed[0]

    @min_range_fixed.setter
    def min_range_fixed(self, _val: bool) -> None:
        self.set_range_fixed(range_min=_val)

    @property
    def max_range_fixed(self) -> bool:
        return self.range_fixed[1]

    @max_range_fixed.setter
    def max_range_fixed(self, _val: bool):
        self.set_range_fixed(range_max=_val)

    def get_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the concentration limits, if any."""

        return self.frame.min_range, self.frame.max_range

    def set_range(self, range_min: float = None, range_max: float = None):
        """Set one or both concentration limits. When setting a range value, the range is automatically fixed."""

        if range_min is not None:
            self.frame.min_range = range_min
            self.frame.min_range_fixed = True
        if range_max is not None:
            self.frame.max_range = range_max
            self.frame.max_range_fixed = True
        self._update()

    @property
    def range(self) -> Tuple[float, float]:
        return self.get_range()

    @range.setter
    def range(self, _range: Tuple[float, float]):
        self.set_range(*_range)

    @property
    def min_range(self) -> Optional[float]:
        return self.range[0]

    @min_range.setter
    def min_range(self, _val: float):
        self.set_range(range_min=_val)

    @property
    def max_range(self) -> Optional[float]:
        return self.range[1]

    @max_range.setter
    def max_range(self, _val: float):
        self.set_range(range_max=_val)

    @property
    def bounding_box_on(self):
        return self.frame.bounding_box_on

    @bounding_box_on.setter
    def bounding_box_on(self, _bounding_box_on):
        self.frame.bounding_box_on = _bounding_box_on
        self._update()

    @property
    def cell_borders_on(self):
        return self.frame.cell_borders_on

    @cell_borders_on.setter
    def cell_borders_on(self, _cell_borders_on):
        self.frame.cell_borders_on = _cell_borders_on
        self._update()

    @property
    def cell_glyphs_on(self):
        return self.frame.cell_glyphs_on

    @cell_glyphs_on.setter
    def cell_glyphs_on(self, _cell_glyphs_on):
        self.frame.cell_glyphs_on = _cell_glyphs_on
        self._update()

    @property
    def cells_on(self):
        return self.frame.cells_on

    @cells_on.setter
    def cells_on(self, _cells_on):
        self.frame.cells_on = _cells_on
        self._update()

    @property
    def cluster_borders_on(self):
        return self.frame.cluster_borders_on

    @cluster_borders_on.setter
    def cluster_borders_on(self, _cluster_borders_on):
        self.frame.cluster_borders_on = _cluster_borders_on
        self._update()

    @property
    def fpp_links_on(self):
        return self.frame.fpp_links_on

    @fpp_links_on.setter
    def fpp_links_on(self, _fpp_links_on):
        self.frame.fpp_links_on = _fpp_links_on
        self._update()

    @property
    def lattice_axes_labels_on(self):
        return self.frame.lattice_axes_labels_on

    @lattice_axes_labels_on.setter
    def lattice_axes_labels_on(self, _lattice_axes_labels_on):
        self.frame.lattice_axes_labels_on = _lattice_axes_labels_on
        self._update()

    @property
    def lattice_axes_on(self):
        return self.frame.lattice_axes_on

    @lattice_axes_on.setter
    def lattice_axes_on(self, _lattice_axes_on):
        self.frame.lattice_axes_on = _lattice_axes_on
        self._update()

    @property
    def window_width_percent(self) -> float:
        """Percent of cell width occupied by the window"""

        return float(self.widget.layout.width.replace('%', '')) / 100

    @window_width_percent.setter
    def window_width_percent(self, _window_width_percent: float):
        self.widget.layout.width = f'{int(_window_width_percent * 100)}%'

    def _update(self):
        self.frame.reset_camera()
        self.frame.current_screenshot_data = self.frame.compute_current_screenshot_data()
        self.frame.draw()
        self.widget.update_canvas()

    @property
    def field_names(self) -> List[str]:
        """Current available field names"""

        if self.frame is None or self.frame.fieldTypes is None:
            return []
        return list(self.frame.fieldTypes.keys())

    def set_field_name(self, _field_name: str):
        """Set the name of the field to render"""

        field_names = self.field_names

        if not field_names:
            warnings.warn('Field names not available', RuntimeWarning)
            return

        if _field_name not in field_names:
            warnings.warn(f'Field name not known: {_field_name}. Available field names are' + ','.join(field_names),
                          RuntimeWarning)
            return

        super().set_field_name(_field_name)
        self.frame.field_name = _field_name
        self._update()

    @property
    def field_name(self) -> str:
        return self.get_field_name()

    @field_name.setter
    def field_name(self, _field_name: str):
        self.set_field_name(_field_name)

    def sync_cameras(self, frame):
        """Synchronize all cameras"""

        return self.widget.sync_cameras(frame.widget)

    def unsync_camera(self):
        """Unsynchronize all cameras"""

        return self.widget.unsync_camera()

    def settings_panel(self, config_fp: str = None) -> JupyterSettingsPanel:
        """
        Inspect the configuration file from path inside of Jupyter.
        Displays a JupyterSettingsPanel.
        """
        if config_fp:
            # todo: get config from fp
            raise NotImplementedError

        def import_callback(data):
            for key, val in data.items():
                if "Color" in key:
                    if type(val) is dict:
                        for k, v in val.items():
                            val[k] = Color(v)
                    else:
                        data[key] = Color(val)
                self.frame.config.setSetting(key, val)
            self.frame._set_vars_from_config()

        panel = JupyterSettingsPanel(self.frame.config, import_callback)
        panel.show()
        return panel

    def control_panel(self) -> JupyterControlPanel:
        """Get a control panel for the client"""

        cp = JupyterControlPanel()
        cp.set_frame(self, 0, 0)
        cp._create_control_panel()
        return cp


class CC3DJupyterGraphicsFrameGrid:

    def __init__(self, rows: int = 1, cols: int = 1):
        if rows <= 0 or cols <= 0:
            raise ValueError('Must have positive rows and columns')

        self._items: List[List[Optional[JupyterGraphicsFrameClient]]] = []
        for r in range(rows):
            self._items.append([None for _ in range(cols)])

        self.grid_box: Optional[VBox] = None

    @property
    def rows(self) -> int:
        """Current number of rows"""

        return len(self._items)

    @property
    def cols(self) -> int:
        """Current number of columns"""

        return len(self._items[0])

    def _prep_grid(self, row: int, col: int):
        while row >= self.rows:
            self._items.append([None for _ in range(self.cols)])
        if col > self.cols:
            for c in range(self.cols, col):
                [r.append(None) for r in self._items]

    def set_frame(self, frame: JupyterGraphicsFrameClient, row: int, col: int):
        """Set the frame at a position"""

        if row < 0 or col < 0:
            raise ValueError('Indices must be non-negative')

        self._prep_grid(row, col)
        self._items[row][col] = frame

    def sync_cameras(self):
        """Synchronize all cameras"""

        for rows_i in range(self.rows):
            for cols_i in range(self.cols):
                item_i = self._items[rows_i][cols_i]
                if item_i is not None:
                    for rows_j in range(self.rows):
                        for cols_j in range(self.cols):
                            item_j = self._items[rows_j][cols_j]
                            if item_j is not None and item_i is not item_j:
                                item_i.sync_cameras(item_j)

    def unsync_cameras(self):
        """Unsynchronize all cameras"""

        for rows in range(self.rows):
            for cols in range(self.cols):
                item = self._items[rows][cols]
                if item is not None:
                    item.unsync_camera()

    def show(self):
        """Show the grid"""

        to_show = []

        for r in range(self.rows):
            to_show_r = [item for item in self._items[r] if item is not None]
            if to_show_r:
                to_show.append(to_show_r)

        hboxes = []
        for to_show_r in to_show:
            if to_show_r:
                hboxes.append(HBox([to_show_rc.widget for to_show_rc in to_show_r]))

        if hboxes:
            self.grid_box = VBox(hboxes)
            display(self.grid_box)

    def control_panel(self) -> JupyterControlPanel:
        """Get a control panel for the grid"""

        cp = JupyterControlPanel(rows=self.rows, cols=self.cols)
        for r, rw in enumerate(self._items):
            for c, w in enumerate(rw):
                if c is not None:
                    cp.set_frame(w, r, c)
        cp._create_control_panel()
        return cp
