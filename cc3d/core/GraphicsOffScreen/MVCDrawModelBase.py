import vtk
from cc3d.core import Configuration
from cc3d import CompuCellSetup
from cc3d.core.GraphicsUtils.utils import to_vtk_rgb
import numpy as np
from cc3d.cpp import CompuCell
import weakref
from math import log10, fabs

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()


class MVCDrawModelBase:
    def __init__(self, boundary_strategy, ren=None):

        (self.minCon, self.maxCon) = (0, 0)

        self.currentDrawingFunction = None
        self.fieldTypes = None
        self.currentDrawingParameters = None
        self.field_extractor = None
        self.boundary_strategy = boundary_strategy

        self.cell_type_array = None
        self.cell_id_array = None
        self.used_cell_types_list = None
        self.lattice_type = None
        self.lattice_type_str = None

        self.celltypeLUT = None

        self.gd_ref = None
        self.ren = ren

    @property
    def boundary_strategy(self):
        try:
            o = self._boundary_strategy()
        except TypeError:
            o = self._boundary_strategy
        return o

    @boundary_strategy.setter
    def boundary_strategy(self, _i):
        try:
            self._boundary_strategy = weakref.ref(_i)
        except TypeError:
            self._boundary_strategy = _i

    @property
    def ren(self):
        try:
            o = self._ren()
        except TypeError:
            o = self._ren
        return o

    @ren.setter
    def ren(self, _i):
        try:
            self._ren = weakref.ref(_i)
        except TypeError:
            self._ren = _i

    def set_generic_drawer(self, gd):
        """

        :param gd:
        :return:
        """
        self.gd_ref = weakref.ref(gd)

    def set_boundary_strategy(self, boundary_strategy):
        """
        sets boundary strategy C++ obj reference
        :param boundary_strategy:
        :return:
        """

        self.boundary_strategy = boundary_strategy

    # should also set "periodic" boundary condition flag(s) (e.g. for drawing FPP links that wraparound)

    def setParams(self):

        # for FPP links (and offset also for cell glyphs)
        self.eps = 1.e-4  # not sure how small this should be (checking to see if cell volume -> 0)
        self.stubSize = 3.0  # dangling line stub size for lines that wraparound periodic BCs
        #        self.offset = 1.0    # account for fact that COM of cell is offset from visualized lattice
        #        self.offset = 0.0    # account for fact that COM of cell is offset from visualized lattice

        # scaling factors to map square lattice to hex lattice (rf. CC3D Manual)
        self.xScaleHex = 1.0
        self.yScaleHex = 0.866
        self.zScaleHex = 0.816

        self.lutBlueRed = vtk.vtkLookupTable()
        self.lutBlueRed.SetHueRange(0.667, 0.0)
        self.lutBlueRed.Build()

    def get_type_lookup_table(self, scene_metadata=None):
        try:
            actual_screenshot = scene_metadata['actual_screenshot']
        except (KeyError, TypeError):
            actual_screenshot = False

        # todo - optimize it to avoid generating it each time we take screenshot
        if actual_screenshot:
            cell_type_color_lookup_table = self.generate_cell_type_lookup_table(scene_metadata=scene_metadata,
                                                                                actual_screenshot=actual_screenshot)
            return cell_type_color_lookup_table

        if self.celltypeLUT is not None:
            return self.celltypeLUT

        self.populate_cell_type_lookup_table(scene_metadata=scene_metadata)

        return self.celltypeLUT

    @staticmethod
    def generate_cell_type_lookup_table(scene_metadata=None, actual_screenshot=False):
        """
        generates cell type color lookup table. Depending whether we got metadata for
        actual screenshot or not we will use cell type color lookup table based on
        settings or we will use colors defined int he screenshot description file
        :param scene_metadata: scene metadata dict
        :param actual_screenshot: flag that tells if we got metadata for actual screenshot
        :return:
        """
        configuration = CompuCellSetup.persistent_globals.configuration
        if actual_screenshot:
            if scene_metadata is None:
                color_map = configuration.getSetting("TypeColorMap")
            else:
                color_map = scene_metadata["TypeColorMap"]
        else:
            color_map = configuration.getSetting("TypeColorMap")

        cell_type_color_lookup_table = vtk.vtkLookupTable()
        # You need to explicitly call Build() when constructing the LUT by hand
        cell_type_color_lookup_table.Build()
        cell_type_color_lookup_table.SetNumberOfTableValues(len(color_map))
        cell_type_color_lookup_table.SetNumberOfColors(len(color_map))

        for type_id, color_obj in list(color_map.items()):
            type_id = int(type_id)
            rgba = to_vtk_rgb(color_obj=color_obj)

            rgba.append(1.0)
            cell_type_color_lookup_table.SetTableValue(type_id, *rgba)

        return cell_type_color_lookup_table

    def populate_cell_type_lookup_table(self, scene_metadata=None):
        """
        Populates "global" cell type color lookup table - based on stored settings
        :param scene_metadata:
        :return:
        """

        self.celltypeLUT = self.generate_cell_type_lookup_table(scene_metadata=scene_metadata)

    def init_lattice_type(self):
        """
        Initializes lattice type and lattice type enum
        :return: None
        """
        self.lattice_type_str = CompuCellSetup.simulation_utils.extract_lattice_type()

        if self.lattice_type_str in list(Configuration.LATTICE_TYPES.keys()):
            self.lattice_type = Configuration.LATTICE_TYPES[self.lattice_type_str]
        else:
            # default choice
            self.lattice_type = Configuration.LATTICE_TYPES["Square"]

    def get_lattice_type(self):
        """
        Returns lattice type as str
        :return: {str} lattice type str
        """
        if self.lattice_type is None:
            self.init_lattice_type()
        return self.lattice_type

    def get_lattice_type_str(self):
        """
        Returns lattice type as integer
        :return: {int} enum corresponding to lattice type
        """
        if self.lattice_type_str is None:
            self.init_lattice_type()
        return self.lattice_type_str

    def set_cell_field_data(self, cell_field_data_dict):
        """
        Stores information about cell field as class variables
        :param cell_field_data_dict: {dict}
        :return:
        """

        self.cell_type_array = cell_field_data_dict['cell_type_array']
        self.cell_id_array = cell_field_data_dict['cell_id_array']
        self.used_cell_types_list = cell_field_data_dict['used_cell_types']

    def setDrawingParametersObject(self, _drawingParams):
        self.currentDrawingParameters = _drawingParams

    def setDrawingParameters(self, _bsd, _plane, _planePos, _fieldType):
        self.bsd = _bsd
        self.plane = _plane
        self.planePos = _planePos
        self.fieldtype = _fieldType

    def setDrawingFunctionName(self, _fcnName):

        if self.drawingFcnName != _fcnName:
            self.drawingFcnHasChanged = True
        else:
            self.drawingFcnHasChanged = False
        self.drawingFcnName = _fcnName

    def clearDisplay(self):
        for actor in self.currentActors:
            self.graphicsFrameWidget.ren.RemoveActor(self.currentActors[actor])

        self.currentActors.clear()

    def is_lattice_hex(self, drawing_params):
        """
        returns if flag that states if the lattice is hex or not. Notice
        In 2D we may use cartesian coordinates for certain projections
        :param drawing_params: {instance of DrawingParameters}
        :return: {bool}
        """
        raise NotImplementedError()

    def get_cell_actors_metadata(self):
        pass

    def float_formatting(self, val, max_exp=6):
        """
        Formats float for display
        :param val:
        :param max_exp:
        :return:
        """

        if val == 0.0:
            return '0.0'

        try:
            val_log = fabs(log10(fabs(val)))
            if val_log <= max_exp:
                val_str = f'{val:f}'
            else:
                val_str = f'{val:e}'
        except:
            val_str = 'NaN'

        return val_str

    def init_min_max_actor(self, min_max_actor, range_array, scene_metadata=None):
        """

        :param min_max_actor:
        :param range_array:
        :param scene_metadata:
        :return:
        """
        min_str = self.float_formatting(range_array[0])
        max_str = self.float_formatting(range_array[1])
        min_max_actor.SetInput("Min: {} Max: {}".format(min_str, max_str))

        font_size = 11
        if self.gd_ref is not None:
            generic_drawer = self.gd_ref()
            vertical_resolution = generic_drawer.vertical_resolution
            if vertical_resolution is not None:
                font_size = int(generic_drawer.vertical_resolution / 100 * 1.1)

        txtprop = min_max_actor.GetTextProperty()
        txtprop.SetFontFamilyToArial()
        txtprop.SetFontSize(font_size)
        txtprop.SetColor(1, 1, 1)
        min_max_actor.SetPosition(20, 20)

    def get_min_max_metadata(self, scene_metadata, field_name):
        """
        Returns dictionary with the following entries:
        1. MinRangeFixed
        2. MaxRangeFixed
        3. MinRange
        3. MaxRange

        :param scene_metadata:{dict} metadata dictionary
        :param field_name: {str} field name
        :return: {dict}
        """
        out_dict = {}
        if {'MinRangeFixed', "MaxRangeFixed", 'MinRange', 'MaxRange'}.issubset(set(scene_metadata.keys())):

            min_range_fixed = scene_metadata['MinRangeFixed']
            max_range_fixed = scene_metadata['MaxRangeFixed']
            min_range = scene_metadata['MinRange']
            max_range = scene_metadata['MaxRange']
        else:
            configuration = CompuCellSetup.persistent_globals.configuration

            min_range_fixed = configuration.getSetting("MinRangeFixed", field_name)
            max_range_fixed = configuration.getSetting("MaxRangeFixed", field_name)
            min_range = configuration.getSetting("MinRange", field_name)
            max_range = configuration.getSetting("MaxRange", field_name)

        out_dict['MinRangeFixed'] = min_range_fixed
        out_dict['MaxRangeFixed'] = max_range_fixed
        out_dict['MinRange'] = min_range
        out_dict['MaxRange'] = max_range

        return out_dict

    def init_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes vector field actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        initializes cluster border actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        initializes fpp links actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        raise NotImplementedError()

    def init_legend_actors(self, actor_specs, drawing_params=None):
        """
        initializes legend (for concentration fields) actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """
        try:
            mapper = actor_specs.metadata['mapper']
        except KeyError:
            print('Could not find mapper object to draw legend')
            return

        actors_dict = actor_specs.actors_dict

        legend_actor = actors_dict['legend_actor']

        legend_actor.SetLookupTable(mapper.GetLookupTable())
        legend_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legend_actor.GetPositionCoordinate().SetValue(0.01, 0.1)
        legend_actor.SetOrientationToHorizontal()

        legend_actor.SetOrientationToVertical()
        # self.legendActor.SetWidth(0.8)
        # self.legendActor.SetHeight(0.10)

        legend_actor.SetWidth(0.1)
        legend_actor.SetHeight(0.9)

        if VTK_MAJOR_VERSION >= 6:
            legend_actor.SetTitle('')

        # You don't actually need to make contrast for the text as
        # it has shadow!
        text_property = legend_actor.GetLabelTextProperty()
        text_property.SetFontSize(12)  # For some reason it doesn't make effect
        # text.BoldOff()
        text_property.SetColor(1.0, 1.0, 1.0)

        legend_actor.SetLabelTextProperty(text_property)

    def init_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        raise NotImplementedError()

    def init_outline_actors(self, actor_specs, drawing_params=None):
        """
        Initializes outline actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        raise NotImplementedError()

    def init_axes_actors(self, actor_specs, drawing_params=None):
        """
        Initializes axes actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        raise NotImplementedError()

    def prepareOutlineActors(self, _actors):
        pass

    def getCamera(self):
        return self.ren.GetActiveCamera()

    def configsChanged(self):
        pass

    def largestDim(self, dim):
        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]

        return ldim

    def prepareAxesActors(self, _mappers, _actors):
        pass

    def prepareLegendActors(self, _mappers, _actors):
        legendActor = _actors[0]
        mapper = _mappers[0]

        legendActor.SetLookupTable(mapper.GetLookupTable())
        legendActor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legendActor.GetPositionCoordinate().SetValue(0.01, 0.1)
        legendActor.SetOrientationToHorizontal()

        legendActor.SetOrientationToVertical()
        # self.legendActor.SetWidth(0.8)
        # self.legendActor.SetHeight(0.10)

        legendActor.SetWidth(0.1)
        legendActor.SetHeight(0.9)

        if VTK_MAJOR_VERSION >= 6:
            legendActor.SetTitle('')

        # You don't actually need to make contrast for the text as
        # it has shadow!
        text_property = legendActor.GetLabelTextProperty()
        text_property.SetFontSize(12)  # For some reason it doesn't make effect
        # text.BoldOff()
        text_property.SetColor(1.0, 1.0, 1.0)

        legendActor.SetLabelTextProperty(text_property)

    def setLatticeType(self, latticeType):
        self.latticeType = latticeType

    def configs_changed(self):

        self.populate_cell_type_lookup_table()

    # @staticmethod
    # def unconditional_invariant_distance_vector(p1, p2, dim):
    #
    #     dist_vec = CompuCell.distanceVectorCoordinatesInvariant(p2, p1, dim)
    #     return np.array([dist_vec.x, dist_vec.y, dist_vec.z])

    def invariant_distance(self, p1, p2, dim):
        """
        Computes invariant distance
        :param p1: 3-element array like obj representing point
        :param p2: 3-element array like obj representing point
        :param dim: field dimension
        :return: invariant distance
        """

        inv_dist = CompuCell.distInvariantCM(p2[0], p2[1], p2[2], p1[0], p1[1], p1[2], dim, self.boundary_strategy)
        return inv_dist

    def invariant_distance_vector(self, p1, p2, dim):
        """
        Computes invariant distance
        :param p1: 3-element array like obj representing point
        :param p2: 3-element array like obj representing point
        :param dim: field dimension
        :return: invariant distance
        """

        dist_vec = CompuCell.distanceVectorCoordinatesInvariant(p2, p1, dim, self.boundary_strategy)
        return np.array([dist_vec.x, dist_vec.y, dist_vec.z])
