from .MVCDrawModelBase import MVCDrawModelBase
import vtk
import math
from math import sqrt
import numpy as np
import string
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object, to_vtk_rgb
from cc3d.core.GraphicsOffScreen.MetadataHandler import MetadataHandler
from cc3d.core.iterators import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
from cc3d.cpp import CompuCell
from cc3d.cpp import PlayerPython
import sys
from cc3d.core.enums import *

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()

epsilon = sys.float_info.epsilon

MODULENAME = "----- MVCDrawModel2D.py:  "


class MVCDrawModel2D(MVCDrawModelBase):
    def __init__(self, boundary_strategy=None, ren=None):
        MVCDrawModelBase.__init__(self, boundary_strategy=boundary_strategy, ren=ren)

        self.cellsMapper = None
        self.hex_cells_mapper = None
        self.cartesianCellsMapper = None
        self.borderMapper = None
        self.borderMapperHex = None
        self.clusterBorderMapper = None
        self.clusterBorderMapperHex = None
        self.cellGlyphsMapper = None
        self.FPPLinksMapper = None

        self.outlineDim = None

        # Set up the mappers (2D) for concentration field.
        self.con_mapper = None
        self.contour_mapper = None
        self.glyphs_mapper = None

        # # Concentration lookup table
        self.numberOfTableColors = None
        self.clut = None

        self.lowTableValue = None
        self.highTableValue = None

        # Contour lookup table
        self.ctlut = None

        self.initArea()
        self.setParams()

        self.pixelized_cartesian_field = True

    # Sets up the VTK simulation area
    def initArea(self):

        # Set up the mappers (2D) for cell vis.
        self.cellsMapper = vtk.vtkPolyDataMapper()
        self.hex_cells_mapper = vtk.vtkPolyDataMapper()
        self.cartesianCellsMapper = vtk.vtkPolyDataMapper()
        self.borderMapper = vtk.vtkPolyDataMapper()
        self.borderMapperHex = vtk.vtkPolyDataMapper()
        self.clusterBorderMapper = vtk.vtkPolyDataMapper()
        self.clusterBorderMapperHex = vtk.vtkPolyDataMapper()
        self.cellGlyphsMapper = vtk.vtkPolyDataMapper()
        self.FPPLinksMapper = vtk.vtkPolyDataMapper()

        self.outlineDim = [0, 0, 0]

        # Set up the mappers (2D) for concentration field.
        self.con_mapper = vtk.vtkPolyDataMapper()
        self.contour_mapper = vtk.vtkPolyDataMapper()
        self.glyphs_mapper = vtk.vtkPolyDataMapper()

        # # Concentration lookup table
        self.numberOfTableColors = 1024
        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0, 1.0)
        self.clut.SetValueRange(1.0, 1.0)
        self.clut.SetAlphaRange(1.0, 1.0)
        self.clut.SetNumberOfColors(self.numberOfTableColors)
        self.clut.Build()

        self.lowTableValue = self.clut.GetTableValue(0)
        self.highTableValue = self.clut.GetTableValue(self.numberOfTableColors - 1)

        # Contour lookup table
        # Do I need lookup table? May be just one color?
        self.ctlut = vtk.vtkLookupTable()
        self.ctlut.SetHueRange(0.6, 0.6)
        self.ctlut.SetSaturationRange(0, 1.0)
        self.ctlut.SetValueRange(1.0, 1.0)
        self.ctlut.SetAlphaRange(1.0, 1.0)
        self.ctlut.SetNumberOfColors(self.numberOfTableColors)
        self.ctlut.Build()

    def is_lattice_hex(self, drawing_params):
        """
        returns if flag that states if the lattice is hex or not. Notice
        In 2D we may use cartesian coordinates for certain projections
        :return: {bool}
        """
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == "hexagonal" and drawing_params.plane.lower() == "xy":
            return True
        else:
            return False

    def init_outline_actors(self, actor_specs, drawing_params=None):
        """
        Initializes outline actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        # field_dim = self.currentDrawingParameters.bsd.fieldDim
        # dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata

        outlineData = vtk.vtkImageData()

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
        dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))

        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
        if self.is_lattice_hex(drawing_params=drawing_params):

            outlineData.SetDimensions(dim[0] + 1, int(dim[1] * math.sqrt(3.0) / 2.0) + 2, 1)
            # print "dim[0]+1,int(dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(dim[0]+1,int(dim[1]*math.sqrt(3.0)/2.0)+2,1)
        else:
            outlineData.SetDimensions(dim[0] + 1, dim[1] + 1, 1)

        outline = vtk.vtkOutlineFilter()

        if VTK_MAJOR_VERSION >= 6:
            outline.SetInputData(outlineData)
        else:
            outline.SetInput(outlineData)

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        outline_actor = actors_dict["outline_actor"]
        outline_actor.SetMapper(outlineMapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)

        outline_color = to_vtk_rgb(scene_metadata["BoundingBoxColor"])

        outline_actor.GetProperty().SetColor(*outline_color)

    def init_axes_actors(self, actor_specs, drawing_params=None):
        """
        Initializes outline actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata

        axes_actor = actors_dict["axes_actor"]

        # lattice_type_str = self.get_lattice_type_str()

        dim_array = [field_dim.x, field_dim.y, field_dim.z]
        # if lattice_type_str.lower() == 'hexagonal':
        if self.is_lattice_hex(drawing_params=drawing_params):
            dim_array = [field_dim.x, field_dim.y * math.sqrt(3.0) / 2.0, field_dim.z * math.sqrt(6.0) / 3.0]

        axes_labels = ["X", "Y", "Z"]

        horizontal_length = dim_array[dim_order[0]]  # x-axis - equivalent
        vertical_length = dim_array[dim_order[1]]  # y-axis - equivalent
        horizontal_label = axes_labels[dim_order[0]]  # x-axis - equivalent
        vertical_label = axes_labels[dim_order[1]]  # y-axis - equivalent

        # eventually do this smarter (only get/update when it changes)
        # color = Configuration.getSetting("AxesColor")

        # color = (float(color.red()) / 255, float(color.green()) / 255, float(color.blue()) / 255)

        axes_color = to_vtk_rgb(scene_metadata["AxesColor"])
        axes_actor.GetProperty().SetColor(axes_color)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(axes_color)
        # tprop.ShadowOn()

        # axesActor.SetNumberOfLabels(4) # number of labels
        axes_actor.SetUse2DMode(1)
        # axesActor.SetScreenSize(50.0) # for labels and axes titles
        # axesActor.SetLabelScaling(True,0,0,0)

        if scene_metadata["ShowHorizontalAxesLabels"]:
            axes_actor.SetXAxisLabelVisibility(1)
        else:
            axes_actor.SetXAxisLabelVisibility(0)

        if scene_metadata["ShowVerticalAxesLabels"]:
            axes_actor.SetYAxisLabelVisibility(1)
        else:
            axes_actor.SetYAxisLabelVisibility(0)

        axes_actor.SetBounds(0, horizontal_length, 0, vertical_length, 0, 0)

        axes_actor.SetXTitle(horizontal_label)
        axes_actor.SetYTitle(vertical_label)
        # title_prop_x = axes_actor.GetTitleTextProperty(0)

        axes_actor.XAxisMinorTickVisibilityOff()
        axes_actor.YAxisMinorTickVisibilityOff()

        axes_actor.SetTickLocationToOutside()

        axes_actor.GetTitleTextProperty(0).SetColor(axes_color)
        axes_actor.GetLabelTextProperty(0).SetColor(axes_color)

        axes_actor.GetXAxesLinesProperty().SetColor(axes_color)
        axes_actor.GetYAxesLinesProperty().SetColor(axes_color)

        axes_actor.GetTitleTextProperty(1).SetColor(axes_color)
        axes_actor.GetLabelTextProperty(1).SetColor(axes_color)


    def init_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes vector field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict

        field_name = drawing_params.fieldName
        field_type = self.get_field_type(drawing_params=drawing_params)


        scene_metadata = drawing_params.screenshot_data.metadata

        mdata = MetadataHandler(mdata=scene_metadata)

        vector_grid = vtk.vtkUnstructuredGrid()

        points = vtk.vtkPoints()
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        points_int_addr = extract_address_int_from_vtk_object(vtkObj=points)
        vectors_int_addr = extract_address_int_from_vtk_object(vtkObj=vectors)

        fill_successful = False
        if self.is_lattice_hex(drawing_params=drawing_params):
            if field_type == FIELD_NUMBER_TO_FIELD_TYPE_MAP[VECTOR_FIELD].lower():
                fill_successful = self.field_extractor.fillVectorFieldData2DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos,
                )
            elif field_type == FIELD_NUMBER_TO_FIELD_TYPE_MAP[VECTOR_FIELD_CELL_LEVEL].lower():
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData2DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos,
                )
        else:
            if field_type == FIELD_NUMBER_TO_FIELD_TYPE_MAP[VECTOR_FIELD].lower():
                fill_successful = self.field_extractor.fillVectorFieldData2D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos,
                )
            elif field_type == FIELD_NUMBER_TO_FIELD_TYPE_MAP[VECTOR_FIELD_CELL_LEVEL].lower():
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData2D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos,
                )

        if not fill_successful:
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_ERROR,
                                           f'fill unsuccessful for field type {field_type} ({field_name})')
            return

        vector_grid.SetPoints(points)
        vector_grid.GetPointData().SetVectors(vectors)

        cone = vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        # cone.SetRadius(4)

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_magnitude_fixed = min_max_dict["MinRangeFixed"]
        max_magnitude_fixed = min_max_dict["MaxRangeFixed"]
        min_magnitude_read = min_max_dict["MinRange"]
        max_magnitude_read = min_max_dict["MaxRange"]

        range_array = vectors.GetRange(-1)

        min_magnitude = range_array[0]
        max_magnitude = range_array[1]

        if min_magnitude_fixed:
            min_magnitude = min_magnitude_read

        if max_magnitude_fixed:
            max_magnitude = max_magnitude_read

        glyphs = vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION >= 6:
            glyphs.SetInputData(vector_grid)
        else:
            glyphs.SetInput(vector_grid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        # glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # rwh: should use of this factor depend on the state of the "Scale arrow length" checkbox?

        # scaling factor for an arrow (ArrowLength indicates scaling factor not actual length)
        arrow_scaling_factor = scene_metadata["ArrowLength"]

        vector_field_actor = actors_dict["vector_field_actor"]
        if mdata.get("FixedArrowColorOn", data_type="bool"):
            glyphs.SetScaleModeToScaleByVector()
            # rangeSpan = maxMagnitude - minMagnitude
            data_scaling_factor = max(abs(min_magnitude), abs(max_magnitude))

            if data_scaling_factor == 0.0:
                # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                data_scaling_factor = 1.0
            glyphs.SetScaleFactor(arrow_scaling_factor / data_scaling_factor)
            # coloring arrows
            # arrow_color = to_vtk_rgb(scene_metadata['ArrowColor'])

            arrow_color = to_vtk_rgb(mdata.get("ArrowColor", data_type="color"))
            vector_field_actor.GetProperty().SetColor(arrow_color)
        else:

            if mdata.get("ScaleArrowsOn", data_type="bool"):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                # rangeSpan = max_magnitude - min_magnitude
                data_scaling_factor = max(abs(min_magnitude), abs(max_magnitude))

                if data_scaling_factor == 0.0:
                    # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                    data_scaling_factor = 1.0
                glyphs.SetScaleFactor(arrow_scaling_factor / data_scaling_factor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrow_scaling_factor)

        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)

        self.glyphs_mapper.SetScalarRange([min_magnitude, max_magnitude])

        vector_field_actor.SetMapper(self.glyphs_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict["min_max_text_actor"], range_array=range_array)

    def init_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        lattice_type_str = self.get_lattice_type_str()
        scr_data = drawing_params.screenshot_data
        if scr_data.cell_glyphs_on:
            self.init_concentration_field_glyphs_actors(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            if lattice_type_str.lower() == "hexagonal" and drawing_params.plane.lower() == "xy":
                self.init_concentration_field_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
            else:
                if self.pixelized_cartesian_field:
                    self.init_concentration_field_actors_cartesian_pixelized(
                        actor_specs=actor_specs, drawing_params=drawing_params
                    )
                else:
                    self.init_concentration_field_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)

        # Handle if something previously prevented initialization of metadata,
        # of which we notify in subsequent calls
        if actor_specs.metadata is None:
            actor_specs.metadata = {}

        if mdata.get("LegendEnable", default=True):
            self.init_legend_actors(actor_specs=actor_specs, drawing_params=drawing_params)

    def init_concentration_field_actors_hex(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors for hex lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)

        # [fieldDim.x, fieldDim.y, fieldDim.z]
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(vtkObj=con_array)
        hex_points_con = vtk.vtkPoints()
        hex_points_con_int_addr = extract_address_int_from_vtk_object(vtkObj=hex_points_con)

        hex_cells_con = vtk.vtkCellArray()
        hex_cells_con_int_addr = extract_address_int_from_vtk_object(vtkObj=hex_cells_con)
        hex_cells_con_poly_data = vtk.vtkPolyData()

        field_type = self.get_field_type(drawing_params=drawing_params)

        if field_type == "confield":
            fill_successful = self.field_extractor.fillConFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfield":
            fill_successful = self.field_extractor.fillScalarFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfieldcelllevel":
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        else:
            print(("unsuported field type {}".format(field_type)))
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_WARNING,
                                           f'unsupported field type {field_type} ({field_name})')
            return

        if not fill_successful:
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_ERROR,
                                           f'fill unsuccessful for field type {field_type} ({field_name})')
            return

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict["MinRangeFixed"]
        max_range_fixed = min_max_dict["MaxRangeFixed"]
        min_range = min_max_dict["MinRange"]
        max_range = min_max_dict["MaxRange"]

        range_array = con_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        if mdata.get("ContoursOn", default=False):
            contour_actor = actors_dict["contour_actor"]
            num_contour_lines = mdata.get("NumberOfContourLines", default=3)
            self.initialize_contours_pixelized(
                [dim[0], dim[1]],
                con_array,
                [min_con, max_con],
                contour_actor,
                num_contour_lines=num_contour_lines,
                hex_flag=True,
            )

        hex_cells_con_poly_data.GetCellData().SetScalars(con_array)
        hex_cells_con_poly_data.SetPoints(hex_points_con)
        hex_cells_con_poly_data.SetPolys(hex_cells_con)

        if VTK_MAJOR_VERSION >= 6:
            self.con_mapper.SetInputData(hex_cells_con_poly_data)
        else:
            self.con_mapper.SetInput(hex_cells_con_poly_data)

        self.con_mapper.ScalarVisibilityOn()
        self.con_mapper.SetLookupTable(self.clut)
        self.con_mapper.SetScalarRange(min_con, max_con)

        concentration_actor = actors_dict["concentration_actor"]

        concentration_actor.SetMapper(self.con_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict["min_max_text_actor"], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {"mapper": self.con_mapper}
        else:
            actor_specs.metadata["mapper"] = self.con_mapper

    def initialize_contours_pixelized(
        self, dim, con_array, min_max, contour_actor, num_contour_lines=2, hex_flag=False
    ):
        """
        INitializes contour actor
        :param dim: {tuple}
        :param con_array: {vtkDoubleArray}
        :param min_max: {tuple (float, float)} concentration min, max
        :param contour_actor: {vtkActor} conrour actor
        :param num_contour_lines: {int} number of contour lines
        :param hex_flag: {bool} indicates if we are on the hex lattice
        :return: None
        """

        data = vtk.vtkImageData()
        data.SetDimensions(dim[0], dim[1], 1)

        if VTK_MAJOR_VERSION >= 6:
            data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        else:
            data.SetScalarTypeToUnsignedChar()

        data.GetPointData().SetScalars(con_array)
        field_image_data = vtk.vtkImageDataGeometryFilter()

        if VTK_MAJOR_VERSION >= 6:
            field_image_data.SetInputData(data)
        else:
            field_image_data.SetInput(data)

        transform = vtk.vtkTransform()
        if hex_flag:
            transform.Scale(1, math.sqrt(3.0) / 2.0, 1)
            if self.currentDrawingParameters.planePos % 3 == 0:
                transform.Translate(0.5, 0, 0)  # z%3==0
            elif self.currentDrawingParameters.planePos % 3 == 1:
                transform.Translate(0, math.sqrt(3.0) / 4.0, 0)  # z%3==1
            else:
                transform.Translate(0.0, -math.sqrt(3.0) / 4.0, 0)  # z%3==2
        else:
            transform.Scale(1, 1, 1)
            transform.Translate(0.5, 0.5, 0)

        iso_contour = vtk.vtkContourFilter()

        iso_contour.SetInputConnection(field_image_data.GetOutputPort())

        iso_contour.GenerateValues(num_contour_lines + 2, min_max)

        tpd1 = vtk.vtkTransformPolyDataFilter()
        tpd1.SetInputConnection(iso_contour.GetOutputPort())
        tpd1.SetTransform(transform)

        # self.contourMapper.SetInputConnection(contour.GetOutputPort())
        self.contour_mapper.SetInputConnection(tpd1.GetOutputPort())
        self.contour_mapper.SetLookupTable(self.ctlut)
        self.contour_mapper.SetScalarRange(min_max)
        self.contour_mapper.ScalarVisibilityOff()
        contour_actor.SetMapper(self.contour_mapper)

    def init_concentration_field_actors_cartesian_pixelized(self, actor_specs, drawing_params=None):

        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        dim = self.planeMapper(
            dim_order, (field_dim.x, field_dim.y, field_dim.z)
        )  # [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(vtkObj=con_array)
        points_con = vtk.vtkPoints()
        points_con_int_addr = extract_address_int_from_vtk_object(vtkObj=points_con)

        cells_con = vtk.vtkCellArray()
        cells_con_int_addr = extract_address_int_from_vtk_object(vtkObj=cells_con)
        cells_con_poly_data = vtk.vtkPolyData()

        field_type = self.get_field_type(drawing_params=drawing_params)

        if field_type == "confield":
            fill_successful = self.field_extractor.fillConFieldData2DCartesian(
                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        elif field_type == "scalarfield":
            fill_successful = self.field_extractor.fillScalarFieldData2DCartesian(
                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfieldcelllevel":
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2DCartesian(
                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        else:
            print(("unsuported field type {}".format(field_type)))
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_WARNING,
                                           f'unsupported field type {field_type} ({field_name})')
            return
        # print("ATTEMPTING PIXELIZED")
        if not fill_successful:
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_ERROR,
                                           f'fill unsuccessful for field type {field_type} ({field_name})')
            return

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict["MinRangeFixed"]
        max_range_fixed = min_max_dict["MaxRangeFixed"]
        min_range = min_max_dict["MinRange"]
        max_range = min_max_dict["MaxRange"]

        range_array = con_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        cells_con_poly_data.GetCellData().SetScalars(con_array)
        cells_con_poly_data.SetPoints(points_con)
        cells_con_poly_data.SetPolys(cells_con)

        if VTK_MAJOR_VERSION >= 6:
            self.con_mapper.SetInputData(cells_con_poly_data)
        else:
            self.con_mapper.SetInput(cells_con_poly_data)

        if mdata.get("ContoursOn", default=False):
            contour_actor = actors_dict["contour_actor"]
            num_contour_lines = mdata.get("NumberOfContourLines", default=3)
            self.initialize_contours_pixelized(
                [dim[0], dim[1]],
                con_array,
                [min_con, max_con],
                contour_actor,
                num_contour_lines=num_contour_lines,
                hex_flag=False,
            )

        self.con_mapper.ScalarVisibilityOn()
        self.con_mapper.SetLookupTable(self.clut)
        self.con_mapper.SetScalarRange(min_con, max_con)

        concentration_actor = actors_dict["concentration_actor"]

        concentration_actor.SetMapper(self.con_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict["min_max_text_actor"], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {"mapper": self.con_mapper}
        else:
            actor_specs.metadata["mapper"] = self.con_mapper

    def init_concentration_field_actors_cartesian(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        dim = self.planeMapper(
            dim_order, (field_dim.x, field_dim.y, field_dim.z)
        )  # [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(vtkObj=con_array)


        field_type = self.get_field_type(drawing_params=drawing_params)

        if field_type == "confield":
            fill_successful = self.field_extractor.fillConFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfield":
            fill_successful = self.field_extractor.fillScalarFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfieldcelllevel":
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        else:
            print(("unsuported field type {}".format(field_type)))
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_WARNING,
                                           f'unsupported field type {field_type} ({field_name})')
            return

        if not fill_successful:
            CompuCell.CC3DLogger.get().log(CompuCell.LOG_ERROR,
                                           f'fill unsuccessful for field type {field_type} ({field_name})')
            return

        # # todo 5 - revisit later
        # numIsos = Configuration.getSetting("NumberOfContourLines", field_name)
        # #        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict["MinRangeFixed"]
        max_range_fixed = min_max_dict["MaxRangeFixed"]
        min_range = min_max_dict["MinRange"]
        max_range = min_max_dict["MaxRange"]

        range_array = con_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        dim_0 = dim[0] + 1
        dim_1 = dim[1] + 1

        data = vtk.vtkImageData()
        data.SetDimensions(dim_0, dim_1, 1)

        if VTK_MAJOR_VERSION >= 6:
            data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        else:
            data.SetScalarTypeToUnsignedChar()

        data.GetPointData().SetScalars(con_array)

        field_image_data = vtk.vtkImageDataGeometryFilter()

        if VTK_MAJOR_VERSION >= 6:
            field_image_data.SetInputData(data)
        else:
            field_image_data.SetInput(data)

        field_image_data.SetExtent(0, dim_0, 0, dim_1, 0, 0)

        if mdata.get("ContoursOn", default=False):
            contour_actor = actors_dict["contour_actor"]
            num_contour_lines = mdata.get("NumberOfContourLines", default=3)
            self.initialize_contours_cartesian(
                field_image_data,
                [min_con, max_con],
                contour_actor,
                num_contour_lines=num_contour_lines,
                scene_metadata=scene_metadata,
            )

        self.clut.SetTableRange([min_con, max_con])

        self.con_mapper.SetInputConnection(field_image_data.GetOutputPort())  # port index = 0

        self.con_mapper.ScalarVisibilityOn()

        self.con_mapper.SetLookupTable(self.clut)
        # 0, self.clut.GetNumberOfColors()) # may manually set range so that
        # type reassignment will not be scalled dynamically when one type is missing
        self.con_mapper.SetScalarRange(min_con, max_con)

        self.con_mapper.SetScalarModeToUsePointData()

        concentration_actor = actors_dict["concentration_actor"]
        concentration_actor.SetMapper(self.con_mapper)  # concentration actor

        self.init_min_max_actor(min_max_actor=actors_dict["min_max_text_actor"], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {"mapper": self.con_mapper}
        else:
            actor_specs.metadata["mapper"] = self.con_mapper

    def initialize_contours_cartesian(
        self, field_image_data, min_max, contour_actor, num_contour_lines=2, scene_metadata=None
    ):

        min_con, max_con = min_max[0], min_max[1]
        iso_contour = vtk.vtkContourFilter()
        iso_contour.SetInputConnection(field_image_data.GetOutputPort())
        mdata = MetadataHandler(mdata=scene_metadata)

        # TODO - FIX IT
        # isoValList = self.getIsoValues(field_name)
        isoValList = []

        # todo 5 - fix handling of specific iso contours
        # num_contour_lines = 0
        # for isoVal in isoValList:
        #     try:
        #         if printIsoValues:  print MODULENAME, '  initScalarFieldActors(): setting (specific) isoval= ', isoVal
        #         isoContour.SetValue(num_contour_lines, isoVal)
        #         num_contour_lines += 1
        #     except:
        #         print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ', self.isovalStr[idx]
        # if num_contour_lines > 0:  num_contour_lines += 1
        del_iso = (max_con - min_con) / (num_contour_lines + 1)  # exclude the min,max for isovalues

        iso_val = min_con + del_iso
        for idx in range(num_contour_lines):
            iso_contour.SetValue(num_contour_lines, iso_val)
            num_contour_lines += 1
            iso_val += del_iso

        iso_contour.SetInputConnection(field_image_data.GetOutputPort())
        # isoContour.GenerateValues(Configuration.getSetting("NumberOfContourLines",
        #                                                    self.currentDrawingParameters.fieldName)+2,
        #                           [self.minCon, self.maxCon])

        self.contour_mapper.SetInputConnection(iso_contour.GetOutputPort())
        self.contour_mapper.SetLookupTable(self.ctlut)
        self.contour_mapper.SetScalarRange(min_con, max_con)
        # this is required to do a SetColor on the actor's property
        self.contour_mapper.ScalarVisibilityOff()

        contour_actor.SetMapper(self.contour_mapper)

        color = to_vtk_rgb(mdata.get("ContourColor", data_type="color"))
        contour_actor.GetProperty().SetColor(color)

    def init_cell_field_glyphs_actors(self, actor_specs, drawing_params=None):

        centroids = vtk.vtkPoints()
        volume_scaling_factors = vtk.vtkFloatArray()
        volume_scaling_factors.SetName("volume_scaling_factors")

        cell_types = vtk.vtkIntArray()
        cell_types.SetName("cell_types")

        mapper = vtk.vtkPolyDataMapper()

        centroids_addr = extract_address_int_from_vtk_object(vtkObj=centroids)
        cell_types_addr = extract_address_int_from_vtk_object(vtkObj=cell_types)
        volume_scaling_factors_addr = extract_address_int_from_vtk_object(vtkObj=volume_scaling_factors)

        types_invisible = PlayerPython.vectorint()
        for type_label in drawing_params.screenshot_data.invisible_types:
            types_invisible.append(int(type_label))

        used_types = self.field_extractor.fillCellFieldGlyphs2D(
            centroids_addr,
            volume_scaling_factors_addr,
            cell_types_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos,
        )
        # polydata should be initialized/created AFTER all arrays have been filled in
        centroid_polydata = vtk.vtkPolyData()
        centroid_polydata.SetPoints(centroids)
        centroid_polydata.GetPointData().AddArray(volume_scaling_factors)
        centroid_polydata.GetPointData().AddArray(cell_types)

        circle = self.get_vtk_circle_source(radius=1.0)

        try:
            actor = list(actor_specs.actors_dict.values())[0]
        except IndexError:
            print("Could not find any actor for listed cell types")
            return

        glyphs = vtk.vtkGlyph3D()

        glyphs.SetInputData(centroid_polydata)
        glyphs.SetSourceConnection(0, circle.GetOutputPort())

        glyphs.ScalingOn()
        glyphs.SetScaleModeToScaleByScalar()
        glyphs.SetColorModeToColorByScalar()

        glyphs.SetScaleFactor(1)  # Overall scaling factor

        # Tell it to index into the glyph table according to scalars
        glyphs.SetIndexModeToScalar()

        # Tell glyph which attribute arrays to use for what
        # see also https://stackoverflow.com/questions/29768049/
        # python-vtk-glyphs-with-independent-position-orientation-color-and-height
        glyphs.SetInputArrayToProcess(0, 0, 0, 0, "volume_scaling_factors")  # 0 - scalars for scaling
        glyphs.SetInputArrayToProcess(3, 0, 0, 0, "cell_types")  # 3 - color

        cell_type_lut = self.get_type_lookup_table()
        mapper.SetInputConnection(glyphs.GetOutputPort())
        mapper.SetLookupTable(cell_type_lut)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(0, cell_type_lut.GetNumberOfTableValues() - 1)
        mapper.SetColorModeToMapScalars()
        # TODO add hex actor scaling - only for the xy plane for other projections we use cartesian
        # display so no scaling
        actor.SetMapper(mapper)
        if self.is_lattice_hex(drawing_params=drawing_params):
            actor.SetScale(self.xScaleHex, self.yScaleHex, 1.0)

    def init_concentration_field_glyphs_actors(self, actor_specs, drawing_params=None):

        centroids = vtk.vtkPoints()
        volume_scaling_factors = vtk.vtkFloatArray()
        volume_scaling_factors.SetName("volume_scaling_factors")

        scalar_value_at_com_array = vtk.vtkFloatArray()
        scalar_value_at_com_array.SetName("scalar_value_at_com")

        centroids_addr = extract_address_int_from_vtk_object(vtkObj=centroids)
        scalar_value_at_com_addr = extract_address_int_from_vtk_object(vtkObj=scalar_value_at_com_array)
        volume_scaling_factors_addr = extract_address_int_from_vtk_object(vtkObj=volume_scaling_factors)

        mapper = vtk.vtkPolyDataMapper()

        # polydata should be initialized/created AFTER all arrays have been filled in
        centroid_polydata = vtk.vtkPolyData()
        centroid_polydata.SetPoints(centroids)
        centroid_polydata.GetPointData().AddArray(volume_scaling_factors)
        # centroid_polydata.GetPointData().AddArray(cell_types)
        centroid_polydata.GetPointData().AddArray(scalar_value_at_com_array)

        field_type = drawing_params.fieldType.lower()
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)
        actors_dict = actor_specs.actors_dict

        if field_type == "confield":
            self.field_extractor.fillConFieldGlyphs2D(
                field_name,
                centroids_addr,
                volume_scaling_factors_addr,
                scalar_value_at_com_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfield":
            self.field_extractor.fillScalarFieldGlyphs2D(
                field_name,
                centroids_addr,
                volume_scaling_factors_addr,
                scalar_value_at_com_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        elif field_type == "scalarfieldcelllevel":
            self.field_extractor.fillScalarFieldCellLevelGlyphs2D(
                field_name,
                centroids_addr,
                volume_scaling_factors_addr,
                scalar_value_at_com_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        circle = self.get_vtk_circle_source(radius=1.0)

        try:
            actor = list(actor_specs.actors_dict.values())[0]
        except IndexError:
            print("Could not find any actor for listed cell types")
            return

        glyphs = vtk.vtkGlyph3D()

        glyphs.SetInputData(centroid_polydata)
        glyphs.SetSourceConnection(0, circle.GetOutputPort())

        glyphs.ScalingOn()
        glyphs.SetScaleModeToScaleByScalar()
        glyphs.SetColorModeToColorByScalar()

        # Overall scaling factor
        glyphs.SetScaleFactor(1)

        # Tell it to index into the glyph table according to scalars
        glyphs.SetIndexModeToScalar()

        # Tell glyph which attribute arrays to use for what
        # see also https://stackoverflow.com/questions/29768049/
        # python-vtk-glyphs-with-independent-position-orientation-color-and-height
        glyphs.SetInputArrayToProcess(0, 0, 0, 0, "volume_scaling_factors")  # 0 - scalars for scaling
        # glyphs.SetInputArrayToProcess(3, 0, 0, 0, 'cell_types')  # 3 - color
        glyphs.SetInputArrayToProcess(3, 0, 0, 0, "scalar_value_at_com")  # 3 - color

        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata

        range_array = scalar_value_at_com_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict["MinRangeFixed"]
        max_range_fixed = min_max_dict["MaxRangeFixed"]
        min_range = min_max_dict["MinRange"]
        max_range = min_max_dict["MaxRange"]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        lut = self.clut

        mapper.SetInputConnection(glyphs.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(min_con, max_con)
        mapper.SetColorModeToMapScalars()

        actor.SetMapper(mapper)
        if self.is_lattice_hex(drawing_params=drawing_params):
            actor.SetScale(self.xScaleHex, self.yScaleHex, 1.0)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper': mapper}
        else:
            actor_specs.metadata['mapper'] = mapper


    def get_vtk_circle_source(self, radius=1.0, num_sides=50):
        """
        generates equivalent of vtkCircleSource using vtk.vtkRegularPolygonSource()
        vtk does not have vtkCircleSource so we need to emulate it
        """
        polygon_source = vtk.vtkRegularPolygonSource()
        # Comment this line to generate a disk instead of a circle.
        # polygonSource.GeneratePolygonOff()
        polygon_source.SetNumberOfSides(num_sides)
        polygon_source.SetRadius(radius)
        polygon_source.SetCenter(0.0, 0.0, 0.0)
        return polygon_source

    def init_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        scr_data = drawing_params.screenshot_data
        if scr_data.cell_glyphs_on:
            self.init_cell_field_glyphs_actors(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            if self.is_lattice_hex(drawing_params=drawing_params):
                self.init_cell_field_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
            else:
                if self.pixelized_cartesian_field:
                    self.init_cell_field_actors_cartesian_pixelized(actor_specs=actor_specs, drawing_params=drawing_params)
                else:
                    self.init_cell_field_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)

    def init_cell_field_actors_hex(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors for hex lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        # field_dim = self.currentDrawingParameters.bsd.fieldDim
        # dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        # mdata = MetadataHandler(mdata=scene_metadata)

        # dim = self.planeMapper(dim_order,
        #                             (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")
        cell_type_int_addr = extract_address_int_from_vtk_object(vtkObj=cell_type_array)

        hex_cells_array = vtk.vtkCellArray()

        hex_cells_int_addr = extract_address_int_from_vtk_object(vtkObj=hex_cells_array)

        hex_cells_poly_data = vtk.vtkPolyData()
        # **********************************************

        hex_points = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        hex_points_int_addr = extract_address_int_from_vtk_object(vtkObj=hex_points)

        self.field_extractor.fillCellFieldData2DHex(
            cell_type_int_addr,
            hex_cells_int_addr,
            hex_points_int_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos,
        )

        hex_cells_poly_data.GetCellData().SetScalars(cell_type_array)
        hex_cells_poly_data.SetPoints(hex_points)
        hex_cells_poly_data.SetPolys(hex_cells_array)

        if VTK_MAJOR_VERSION >= 6:
            self.hex_cells_mapper.SetInputData(hex_cells_poly_data)
        else:
            self.hex_cells_mapper.SetInput(hex_cells_poly_data)

        cell_type_lut = self.get_type_lookup_table()
        cell_type_lut_max = cell_type_lut.GetNumberOfTableValues() - 1

        self.hex_cells_mapper.ScalarVisibilityOn()
        self.hex_cells_mapper.SetLookupTable(cell_type_lut)
        self.hex_cells_mapper.SetScalarRange(0, cell_type_lut_max)

        cells_actor = actors_dict["cellsActor"]
        cells_actor.SetMapper(self.hex_cells_mapper)

    def init_cell_field_actors_cartesian_pixelized(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        actors_dict = actor_specs.actors_dict
        # field_dim = self.currentDrawingParameters.bsd.fieldDim
        # dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        # mdata = MetadataHandler(mdata=scene_metadata)

        # dim = self.planeMapper(dim_order,
        #                             (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")
        cell_type_int_addr = extract_address_int_from_vtk_object(vtkObj=cell_type_array)

        cartesian_cells_array = vtk.vtkCellArray()

        cartesian_cells_int_addr = extract_address_int_from_vtk_object(vtkObj=cartesian_cells_array)

        cartesian_cells_poly_data = vtk.vtkPolyData()
        # **********************************************

        cartesian_points = vtk.vtkPoints()
        cartesian_points_int_addr = extract_address_int_from_vtk_object(vtkObj=cartesian_points)

        self.field_extractor.fillCellFieldData2DCartesian(
            cell_type_int_addr,
            cartesian_cells_int_addr,
            cartesian_points_int_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos,
        )

        cartesian_cells_poly_data.GetCellData().SetScalars(cell_type_array)
        cartesian_cells_poly_data.SetPoints(cartesian_points)
        cartesian_cells_poly_data.SetPolys(cartesian_cells_array)

        if VTK_MAJOR_VERSION >= 6:
            self.cellsMapper.SetInputData(cartesian_cells_poly_data)
        else:
            self.cellsMapper.SetInput(cartesian_cells_poly_data)

        cell_type_lut = self.get_type_lookup_table(scene_metadata=scene_metadata)
        cell_type_lut_max = cell_type_lut.GetNumberOfTableValues() - 1

        self.cellsMapper.ScalarVisibilityOn()
        self.cellsMapper.SetLookupTable(cell_type_lut)
        self.cellsMapper.SetScalarRange(0, cell_type_lut_max)

        cells_actor = actors_dict["cellsActor"]
        cells_actor.SetMapper(self.cellsMapper)

    def init_cell_field_actors_cartesian(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        # mdata = MetadataHandler(mdata=scene_metadata)

        # [fieldDim.x, fieldDim.y, fieldDim.z]
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")

        cell_type_int_addr = extract_address_int_from_vtk_object(vtkObj=cell_type_array)

        self.field_extractor.fillCellFieldData2D(
            cell_type_int_addr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos
        )

        dim_new = [dim[0] + 1, dim[1] + 1, dim[2] + 1]

        u_grid_conc = vtk.vtkStructuredPoints()
        u_grid_conc.SetDimensions(dim_new[0], dim_new[1], dim_new[2])

        u_grid_conc.GetPointData().SetScalars(cell_type_array)

        cells_plane = vtk.vtkImageDataGeometryFilter()

        cells_plane.SetExtent(0, dim_new[0], 0, dim_new[1], 0, 0)
        if VTK_MAJOR_VERSION >= 6:
            cells_plane.SetInputData(u_grid_conc)
        else:
            cells_plane.SetInput(u_grid_conc)

        self.cellsMapper.SetInputConnection(cells_plane.GetOutputPort())
        self.cellsMapper.ScalarVisibilityOn()

        cell_type_lut = self.get_type_lookup_table()
        cell_type_lut_max = cell_type_lut.GetNumberOfTableValues() - 1

        self.cellsMapper.SetLookupTable(cell_type_lut)  # def'd in parent class
        self.cellsMapper.SetScalarRange(0, cell_type_lut_max)

        cells_actor = actors_dict["cellsActor"]
        cells_actor.SetMapper(self.cellsMapper)
        # cells_actor.GetProperty().SetInterpolationToFlat()

    def init_borders_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        # actors_dict = actor_specs.actors_dict
        # field_dim = self.currentDrawingParameters.bsd.fieldDim
        # dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(vtkObj=points)

        lines_int_addr = extract_address_int_from_vtk_object(vtkObj=lines)

        hex_flag = False
        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() =='hexagonal' and drawing_params.plane.lower()=="xy":
        if self.is_lattice_hex(drawing_params=drawing_params):
            hex_flag = True

        if hex_flag:
            self.field_extractor.fillBorderData2DHex(
                points_int_addr,
                lines_int_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        else:
            self.field_extractor.fillBorderData2D(
                points_int_addr,
                lines_int_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.borderMapper.SetInputData(borders)
        else:
            self.borderMapper.SetInput(borders)

        border_actor = actor_specs.actors_dict["border_actor"]
        # actors = list(actor_specs.actors_dict.values())

        border_actor.SetMapper(self.borderMapper)

        border_color = to_vtk_rgb(mdata.get("BorderColor", data_type="color"))
        # coloring borders
        border_actor.GetProperty().SetColor(*border_color)

    def init_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(vtkObj=points)
        lines_int_addr = extract_address_int_from_vtk_object(vtkObj=lines)

        hex_flag = False
        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() =='hexagonal' and drawing_params.plane.lower()=="xy":
        if self.is_lattice_hex(drawing_params=drawing_params):
            hex_flag = True

        if hex_flag:
            self.field_extractor.fillClusterBorderData2DHex(
                points_int_addr,
                lines_int_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )
        else:
            self.field_extractor.fillClusterBorderData2D(
                points_int_addr,
                lines_int_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos,
            )

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.clusterBorderMapper.SetInputData(borders)
        else:
            self.clusterBorderMapper.SetInput(borders)

        cluster_border_actor = actor_specs.actors_dict["cluster_border_actor"]

        cluster_border_actor.SetMapper(self.clusterBorderMapper)

        cluster_border_color = to_vtk_rgb(mdata.get("ClusterBorderColor", data_type="color"))
        # coloring borders
        cluster_border_actor.GetProperty().SetColor(*cluster_border_color)

    def init_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        initializes fpp links actors
        :param actor_specs:
        :param drawing_params
        :return: None
        """
        _drawing_params = self.currentDrawingParameters if drawing_params is None else drawing_params

        mdata = MetadataHandler(mdata=_drawing_params.screenshot_data.metadata)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_ad = extract_address_int_from_vtk_object(vtkObj=points)
        lines_ad = extract_address_int_from_vtk_object(vtkObj=lines)

        self.field_extractor.fillLinksField2D(points_ad, lines_ad, _drawing_params.plane, _drawing_params.planePosition)

        fpp_links_pd = vtk.vtkPolyData()
        fpp_links_pd.SetPoints(points)
        fpp_links_pd.SetLines(lines)

        fpp_links_actor = actor_specs.actors_dict["fpp_links_actor"]

        if VTK_MAJOR_VERSION >= 6:
            self.FPPLinksMapper.SetInputData(fpp_links_pd)
        else:
            fpp_links_pd.Update()
            self.FPPLinksMapper.SetInput(fpp_links_pd)

        fpp_links_actor.SetMapper(self.FPPLinksMapper)
        # coloring borders
        fpp_links_actor.GetProperty().SetColor(*to_vtk_rgb(mdata.get("FPPLinksColor", data_type="color")))

    @staticmethod
    def is_within_lattice(coord, coord_dim, eps=0.01):
        """
        checks if a coordinate is within the lattice
        :param coord:
        :param coord_dim:
        :param eps:
        :return:
        """
        return 0 - eps <= coord <= coord_dim + eps

    # Optimize code?
    def dimOrder(self, plane):
        plane = plane.lower()
        order = (0, 1, 2)
        if plane == "xy":
            order = (0, 1, 2)
        elif plane == "xz":
            order = (0, 2, 1)
        elif plane == "yz":
            order = (1, 2, 0)

        return order

    # Optimize code?
    def pointOrder(self, plane):
        plane = string.lower(plane)
        order = (0, 1, 2)
        if plane == "xy":
            order = (0, 1, 2)
        elif plane == "xz":
            order = (0, 2, 1)
        elif plane == "yz":
            order = (2, 0, 1)

        return order

    def planeMapper(self, order, tuple):
        return [tuple[order[0]], tuple[order[1]], tuple[order[2]]]

    def HexCoordXY(self, x, y, z):

        if z % 2:
            if y % 2:
                return [x, sqrt(3.0) / 2.0 * (y + 2.0 / 3.0), z * sqrt(6.0) / 3.0]
            else:
                return [x + 0.5, sqrt(3.0) / 2.0 * (y + 2.0 / 3.0), z * sqrt(6.0) / 3.0]

        else:
            if y % 2:
                return [x, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0]
            else:
                return [x + 0.5, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0]
