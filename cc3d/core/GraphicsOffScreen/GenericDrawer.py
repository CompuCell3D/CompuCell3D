# todo list:
# todo - make once call per 2D/3D or even try global call if possible
# todo - get max cell typ from the simulation
# todo  - fix this so that it is called only once per drawing series
# todo cell_field_data_dict = self.extract_cell_field_data()


# todo - find better way to determine if fpp plugin is properly loaded
# todo - any references to simpletabview should be via weakref

# todo - add action to remove screenshot file
# todo - mcs and min max concentrations are not displayed - fix it
# todo - add fcn get_metadata(scene_metadata, label, field) - returns scene_metadata entry of value from config
# todo - add fpp link color in the configuration dialog - fix metadata in Graphics FrameWidget then
# todo -  add versioned read_screenshot_json fcn and same for write  - to make the choice pf parser seamless to the user
# todo - improve scr data error handling - metadata, version etc. test it for robustness
# todo - cleanup info bar for the player - remove min max from there

# workflow for adding new setting
# ===============================
# 1. Add setting to _settings.sqllite - use SqlBrowser to edit the database
# 2. Edit configuration dialog (edit form via designer, generate ui_xxx.py using pyuic5 and edit CondifugationDialog.py)
# 3. Edit ScreenshotManagerCore - readScreenshotData fcn and write screenshot fct data
# 4. Edit Graphics widget function that geenrate sceen metadata to have this new setting be reflected in sscene metadata

from cc3d.core import Configuration
import vtk
from os.path import splitext
from cc3d.core.enums import *
from copy import deepcopy
from .MVCDrawView2D import MVCDrawView2D
from .MVCDrawModel2D import MVCDrawModel2D
from .MVCDrawView3D import MVCDrawView3D
from .MVCDrawModel3D import MVCDrawModel3D
from cc3d.core.GraphicsUtils.ScreenshotData import ScreenshotData
from .Specs import ActorSpecs
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object
from .DrawingParameters import DrawingParameters
from cc3d.CompuCellSetup import persistent_globals

from typing import Optional

MODULENAME = '---- GraphicsFrameWidget.py: '


class GenericDrawer:
    def __init__(self, boundary_strategy=None):
        # we will do a lazy initialization of the self.ren_win - inside output_screenshot method
        self.ren_win = None
        self.plane = None
        self.planePos = None
        self.field_extractor = None
        self.boundary_strategy = boundary_strategy

        self.ren_2D = vtk.vtkRenderer()
        self.interactive_camera_flag = False
        self.vertical_resolution = None

        # MDIFIX
        self.draw_model_2D = MVCDrawModel2D(boundary_strategy=boundary_strategy, ren=self.ren_2D)
        self.draw_model_2D.set_generic_drawer(gd=self)
        self.draw_view_2D = MVCDrawView2D(self.draw_model_2D, ren=self.ren_2D)

        self.draw_model_3D = MVCDrawModel3D(boundary_strategy=boundary_strategy, ren=self.ren_2D)
        self.draw_model_3D.set_generic_drawer(gd=self)
        self.draw_view_3D = MVCDrawView3D(self.draw_model_3D, ren=self.ren_2D)

        #
        # dict {field_type: drawing fcn}
        self.drawing_fcn_dict = {
            'CellField': self.draw_cell_field,
            'ConField': self.draw_concentration_field,
            'ScalarField': self.draw_concentration_field,
            'ScalarFieldCellLevel': self.draw_concentration_field,
            'VectorField': self.draw_vector_field,
            'VectorFieldCellLevel': self.draw_vector_field,

        }
        self.screenshotWindowFlag = False
        self.lattice_type = Configuration.LATTICE_TYPES['Square']

        self.current_step = None
        self.cell_field_data_dict = None
        self.cell_shell_optimization = None

    def set_vertical_resolution(self, vertical_resoultion):
        """

        :param vertical_resoultion:
        :return:
        """
        self.vertical_resolution = vertical_resoultion

    def set_pixelized_cartesian_scene(self, flag: bool) -> None:
        """
        Enables pixelized cartesian scene

        :param flag:
        :return:
        """

        self.draw_model_2D.pixelized_cartesian_field = flag

    def set_boundary_strategy(self, boundary_strategy):
        """
        sets boundary strategy C++ obj reference

        :param boundary_strategy:
        :return:
        """

        self.boundary_strategy = boundary_strategy
        self.draw_model_2D.set_boundary_strategy(boundary_strategy=boundary_strategy)
        self.draw_model_3D.set_boundary_strategy(boundary_strategy=boundary_strategy)

    def configsChanged(self):
        """
        Function called by the GraphicsFrameWidget after configuration dialog has been approved
        We update information from configuration dialog here

        :return: None
        """
        self.draw_model_2D.populate_cell_type_lookup_table()
        self.draw_model_3D.populate_cell_type_lookup_table()

    def set_interactive_camera_flag(self, flag):
        """
        sets flag that allows resetting of camera parameters during each draw function. for
        Interactive model when GenericDrwer is called form the GUI this should be set to False
        but for screenshots this should be True

        :param flag:{bool}
        :return:
        """
        self.interactive_camera_flag = flag

    def get_renderer(self):
        """

        :return:
        """

        return self.ren_2D

    def get_active_camera(self):
        """
        returns active camera object

        :return: {vtkCamera}
        """

        return self.get_renderer().GetActiveCamera()

    def clear_display(self):
        self.draw_view_2D.remove_all_actors_from_renderer()
        self.draw_view_3D.remove_all_actors_from_renderer()

    def extract_cell_field_data(self, cell_shell_optimization: Optional[bool] = False):
        """
        Extracts basic information about cell field

        :return:
        """
        if cell_shell_optimization is None:
            cell_shell_optimization = False

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")
        cell_type_array_int_addr = extract_address_int_from_vtk_object(cell_type_array)
        # Also get the CellId
        cell_id = vtk.vtkLongArray()
        cell_id.SetName("cellid")
        cell_id_int_addr = extract_address_int_from_vtk_object(cell_id)

        used_cell_types_list = self.field_extractor.fillCellFieldData3D(cell_type_array_int_addr, cell_id_int_addr,
                                                                        cell_shell_optimization)

        ret_val = {
            'cell_type_array': cell_type_array,
            'cell_id_array': cell_id,
            'used_cell_types': used_cell_types_list
        }
        return ret_val

    def set_field_extractor(self, field_extractor):

        self.field_extractor = field_extractor
        self.draw_model_2D.field_extractor = field_extractor
        self.draw_model_3D.field_extractor = field_extractor

    def draw_vector_field(self, drawing_params):
        """
        Draws vector field

        :param drawing_params: {DrawingParameters}
        :return: None
        """

        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()

        actor_specs_final = view.prepare_vector_field_actors(actor_specs=actor_specs, drawing_params=drawing_params)

        model.init_vector_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

        view.show_vector_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

    def draw_concentration_field(self, drawing_params):
        """
        Draws concentration field

        :param drawing_params: {DrawingParameters}
        :return: None
        """

        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()

        actor_specs_final = view.prepare_concentration_field_actors(actor_specs=actor_specs,
                                                                    drawing_params=drawing_params)

        model.init_concentration_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

        view.show_concentration_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

    def draw_cell_field(self, drawing_params):
        """
        Draws cell field

        :param drawing_params:{DrawingParameters}
        :return: None
        """
        if not drawing_params.screenshot_data.cells_on:
            return

        model, view = self.get_model_view(drawing_params=drawing_params)
        try:
            max_cell_type_used = max(drawing_params.bsd.cell_types_used)
        except ValueError:
            max_cell_type_used = 0

        actor_specs = ActorSpecs()
        actor_specs.actor_label_list = ['cellsActor']

        # todo 5 - get max cell type here
        actor_specs.metadata = {
            'invisible_types': drawing_params.screenshot_data.invisible_types,
            'all_types': list(range(max_cell_type_used + 1))
        }

        actor_specs_final = view.prepare_cell_field_actors(actor_specs, drawing_params=drawing_params)
        model.init_cell_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        view.show_cell_actors(actor_specs=actor_specs_final)

    def draw_cell_borders(self, drawing_params):
        """
        Draws cell borders

        :param drawing_params:
        :return:
        """

        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs.actor_label_list = ['borderActor']
        actor_specs_final = view.prepare_border_actors(actor_specs=actor_specs)

        model.init_borders_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        view.show_cell_borders(actor_specs=actor_specs_final)

    def draw_cluster_borders(self, drawing_params):
        """
        Draws cluster borders

        :param drawing_params:
        :return: None
        """

        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs_final = view.prepare_cluster_border_actors(actor_specs=actor_specs)
        model.init_cluster_border_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        view.show_cluster_border_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

    def draw_fpp_links(self, drawing_params):
        """
        Draws FPP links

        :param drawing_params:
        :return: None
        """

        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs_final = view.prepare_fpp_links_actors(actor_specs=actor_specs)
        model.init_fpp_links_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        view.show_fpp_links_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

    def draw_bounding_box(self, drawing_params):
        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs_final = view.prepare_outline_actors(actor_specs=actor_specs)

        model.init_outline_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        show_flag = drawing_params.screenshot_data.bounding_box_on
        view.show_outline_actors(actor_specs=actor_specs_final, drawing_params=drawing_params, show_flag=show_flag)

    def draw_axes(self, drawing_params):
        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs_final = view.prepare_axes_actors(actor_specs=actor_specs)
        camera = view.getCamera()
        if actor_specs_final.metadata is None:
            actor_specs_final.metadata = {'camera': camera}
        else:
            actor_specs_final.metadata['camera'] = camera

        model.init_axes_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)
        show_flag = drawing_params.screenshot_data.lattice_axes_on
        view.show_axes_actors(actor_specs=actor_specs_final, drawing_params=drawing_params, show_flag=show_flag)

    def get_model_view(self, drawing_params):
        """
        returns pair of model view objects depending on the dimension label

        :param drawing_params: DrawingParameters instance for the model view
        :type drawing_params: DrawingParameters
        :return: mode, view object tuple
        :rtype: (cc3d.core.GraphicsOffScreen.MVCDrawModel2D.MVCDrawModel2D,
            cc3d.core.GraphicsOffScreen.MVCDrawView2D.MVCDrawView2D) or
            (cc3d.core.GraphicsOffScreen.MVCDrawModel3D.MVCDrawModel3D,
            cc3d.core.GraphicsOffScreen.MVCDrawView3D.MVCDrawView3D)
        """
        dimension_label = drawing_params.screenshot_data.spaceDimension

        if dimension_label == '2D':
            return self.draw_model_2D, self.draw_view_2D
        else:
            return self.draw_model_3D, self.draw_view_3D

    def draw(self, screenshot_data: ScreenshotData, bsd, screenshot_name=None):
        """

        :param screenshot_data:
        :param bsd:
        :param screenshot_name:
        :return:
        """

        drawing_params = DrawingParameters()
        drawing_params.screenshot_data = screenshot_data
        drawing_params.bsd = bsd
        drawing_params.plane = screenshot_data.projection
        drawing_params.planePosition = screenshot_data.projectionPosition
        drawing_params.planePos = screenshot_data.projectionPosition
        # e.g. plotData = ('Cell_Field','CellField')
        drawing_params.fieldName = screenshot_data.plotData[0]
        drawing_params.fieldType = screenshot_data.plotData[1]

        model, view = self.get_model_view(drawing_params=drawing_params)

        if self.current_step is None:
            self.current_step = bsd.current_step


        current_cell_shell_optimization = screenshot_data.cell_shell_optimization
        if self.cell_shell_optimization is None:
            self.cell_shell_optimization = current_cell_shell_optimization

        if (self.current_step != bsd.current_step
                or self.cell_field_data_dict is None
                or current_cell_shell_optimization != self.cell_shell_optimization):
            self.cell_shell_optimization = current_cell_shell_optimization

            self.cell_field_data_dict = self.extract_cell_field_data(
                cell_shell_optimization=self.cell_shell_optimization)

            self.current_step = bsd.current_step

            bsd.cell_types_used = deepcopy(self.cell_field_data_dict['used_cell_types'])

            drawing_params.bsd = bsd

            # passes information about cell lattice
        model.set_cell_field_data(cell_field_data_dict=self.cell_field_data_dict)

        model.setDrawingParametersObject(drawing_params)

        try:
            key = get_field_type(drawing_params.fieldType)
            draw_fcn = self.drawing_fcn_dict[key]
        except KeyError:
            # if str(key).strip() != '':
            #     print(f"Could not find function for {key}")
            draw_fcn = None

        if draw_fcn is not None:
            # removing all current actors
            view.clear_scene()

            draw_fcn(drawing_params=drawing_params)

            # decorations
            if drawing_params.screenshot_data.cell_borders_on:

                try:
                    self.draw_cell_borders(drawing_params=drawing_params)
                except NotImplementedError:
                    pass

            # decorations
            if drawing_params.screenshot_data.cluster_borders_on:

                try:
                    self.draw_cluster_borders(drawing_params=drawing_params)
                except NotImplementedError:
                    pass

            # decorations
            if drawing_params.screenshot_data.fpp_links_on:

                try:
                    self.draw_fpp_links(drawing_params=drawing_params)
                except NotImplementedError:
                    pass

            if drawing_params.screenshot_data.bounding_box_on:
                try:
                    self.draw_bounding_box(drawing_params=drawing_params)
                except NotImplementedError:
                    pass

            if drawing_params.screenshot_data.lattice_axes_on:
                try:
                    self.draw_axes(drawing_params=drawing_params)
                except NotImplementedError:
                    pass

            # we allow resetting of camera properties only in the non-interactive mode
            # in the interactive mode camera is managed by the GUI
            if not self.interactive_camera_flag:
                if screenshot_data.clippingRange is not None:
                    view.set_custom_camera(camera_settings=screenshot_data)

    def output_screenshot(self, screenshot_fname, file_format='png', screenshot_data=None):
        """
        Saves scene rendered in the renderer to the image

        :param screenshot_fname: screenshot filename
        :type screenshot_fname: str
        :param file_format: output suffix
        :type file_format: str
        :param screenshot_data: screenshot data
        :type screenshot_data: ScreenshotData
        :return: None
        """

        ren = self.get_renderer()
        if self.ren_win is None:
            self.ren_win = vtk.vtkRenderWindow()
            self.ren_win.SetOffScreenRendering(1)
            self.ren_win.AddRenderer(ren)

        ren_win = self.ren_win

        if screenshot_data is not None:
            ren_win.SetSize(screenshot_data.win_width, screenshot_data.win_height)

        ren_win.Render()

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(ren_win)
        window_to_image_filter.Update()

        if file_format.lower() == 'png':
            writer = vtk.vtkPNGWriter()
        elif file_format.lower() == 'tiff':
            writer = vtk.vtkTIFFWriter()
            screenshot_fname = splitext(screenshot_fname)[0] + '.tif'
        else:
            writer = vtk.vtkPNGWriter()

        writer.SetFileName(screenshot_fname)

        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def resetCamera(self):
        """
        Resets camera to default settings

        :return: None
        """
        pass


class GenericDrawerCC3DPy(GenericDrawer):
    """
    Subclass with necessary hooks for Python API
    """

    def __init__(self, parent=None, originating_widget=None):
        super().__init__()

    def get_model_view(self, drawing_params):
        model, view = GenericDrawer.get_model_view(self, drawing_params)

        lattice_type = self.lattice_type
        lattice_type_str = [k for k, v in Configuration.LATTICE_TYPES.items() if v == lattice_type][0]

        def init_lattice_type():
            model.lattice_type = lattice_type
            model.lattice_type_str = lattice_type_str

        model.init_lattice_type = init_lattice_type
        model.init_lattice_type()

        return model, view
