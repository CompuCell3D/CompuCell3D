"""
Defines features for interactive visualization for use with CC3D simservice applications in a Jupyter notebook
"""

from typing import Optional, Union, Tuple

from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor, vtkRenderWindow

import cc3d.CompuCellSetup
from cc3d.core.GraphicsOffScreen.GenericDrawer import GenericDrawer
from cc3d.core.GraphicsUtils.GraphicsFrame import GraphicsFrame
from cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame import (
    CC3DPyGraphicsFrameClientBase, CC3DPyInteractorStyle, np_img_data, save_img
)


# Test for IPython
try:
    get_ipython
    __has_interactive__ = True
    from ipyvtklink.viewer import ViewInteractiveWidget
    from IPython.display import display
except NameError:
    __has_interactive__ = False
    ViewInteractiveWidget = object


class JupyterGraphicsFrame(GraphicsFrame):
    """
    Interactive graphics in a Jupyter notebook
    """

    def __init__(self, *args, **kwargs):

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

        generic_drawer = GenericDrawer(boundary_strategy=pg.simulator.getBoundaryStrategy())
        generic_drawer.set_pixelized_cartesian_scene(pg.configuration.getSetting("PixelizedCartesianFields"))
        generic_drawer.set_field_extractor(field_extractor=field_extractor)

        super().__init__(generic_drawer=generic_drawer, current_bsd=pg.screenshot_manager.bsd, *args, **kwargs)

        # Initialize options
        self.bounding_box_on = self.config.getSetting('BoundingBoxOn')
        self.cell_borders_on = self.config.getSetting('CellBordersOn')
        self.cell_glyphs_on = self.config.getSetting('CellGlyphsOn')
        self.cells_on = self.config.getSetting('CellsOn')
        self.cluster_borders_on = self.config.getSetting('ClusterBordersOn')
        self.fpp_links_on = self.config.getSetting('FPPLinksOn')
        self.lattice_axes_labels_on = self.config.getSetting('ShowAxes')
        self.lattice_axes_on = self.config.getSetting('ShowHorizontalAxesLabels') or self.config.getSetting(
            'ShowVerticalAxesLabels')

        # Initialize initial rendered state

        # noinspection PyUnresolvedReferences
        self.style.SetCurrentRenderer(self.gd.get_renderer())
        self.draw()
        self.reset_camera()
        self.init_field_types()
        self.Render()

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


class JupyterGraphicsFrameClient(CC3DPyGraphicsFrameClientBase):
    """Client for a Jupyter grpahics frame"""

    def __init__(self,
                 name: str = None,
                 config_fp: str = None):

        super().__init__(name=name, config_fp=config_fp)

        self.frame: Optional[JupyterGraphicsFrame] = None
        self._interactor_widget: Optional[ViewInteractiveWidget] = None

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

        self._interactor_widget = ViewInteractiveWidget(self.frame.renWin)
        display(self._interactor_widget)
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

        self.frame.draw()
        self._interactor_widget: ViewInteractiveWidget
        self._interactor_widget.update_canvas()

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
        Function that wires-up the widget to behave according tpo the dimension of the visualization

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

    def _update(self):
        self.frame.reset_camera()
        self.frame.current_screenshot_data = self.frame.compute_current_screenshot_data()
        self.frame.draw()
        self._interactor_widget.update_canvas()