from .MVCDrawModelBase import MVCDrawModelBase
import vtk
import math
from math import sqrt
import string
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object, to_vtk_rgb
from cc3d.core.GraphicsOffScreen.MetadataHandler import MetadataHandler
from cc3d.core.iterators import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
from cc3d.cpp import CompuCell

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()

MODULENAME = '----- MVCDrawModel2D.py:  '


class MVCDrawModel2D(MVCDrawModelBase):
    def __init__(self):
        MVCDrawModelBase.__init__(self)

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

        ## Set up the mappers (2D) for cell vis.
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

        ## Set up the mappers (2D) for concentration field.
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
        if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
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
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata

        outlineData = vtk.vtkImageData()

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))

        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
        if self.is_lattice_hex(drawing_params=drawing_params):

            outlineData.SetDimensions(self.dim[0] + 1, int(self.dim[1] * math.sqrt(3.0) / 2.0) + 2, 1)
            # print "self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
        else:
            outlineData.SetDimensions(self.dim[0] + 1, self.dim[1] + 1, 1)

        outline = vtk.vtkOutlineFilter()

        if VTK_MAJOR_VERSION >= 6:
            outline.SetInputData(outlineData)
        else:
            outline.SetInput(outlineData)

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        outline_actor = actors_dict['outline_actor']
        outline_actor.SetMapper(outlineMapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)

        outline_color = to_vtk_rgb(scene_metadata['BoundingBoxColor'])

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

        axes_actor = actors_dict['axes_actor']

        lattice_type_str = self.get_lattice_type_str()

        dim_array = [field_dim.x, field_dim.y, field_dim.z]
        # if lattice_type_str.lower() == 'hexagonal':
        if self.is_lattice_hex(drawing_params=drawing_params):
            dim_array = [field_dim.x, field_dim.y * math.sqrt(3.0) / 2.0, field_dim.z * math.sqrt(6.0) / 3.0]

        axes_labels = ['X', 'Y', 'Z']

        horizontal_length = dim_array[dim_order[0]]  # x-axis - equivalent
        vertical_length = dim_array[dim_order[1]]  # y-axis - equivalent
        horizontal_label = axes_labels[dim_order[0]]  # x-axis - equivalent
        vertical_label = axes_labels[dim_order[1]]  # y-axis - equivalent

        # eventually do this smarter (only get/update when it changes)
        # color = Configuration.getSetting("AxesColor")

        # color = (float(color.red()) / 255, float(color.green()) / 255, float(color.blue()) / 255)

        axes_color = to_vtk_rgb(scene_metadata['AxesColor'])
        axes_actor.GetProperty().SetColor(axes_color)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(axes_color)
        # tprop.ShadowOn()

        # axesActor.SetNumberOfLabels(4) # number of labels
        axes_actor.SetUse2DMode(1)
        # axesActor.SetScreenSize(50.0) # for labels and axes titles
        # axesActor.SetLabelScaling(True,0,0,0)

        if scene_metadata['ShowHorizontalAxesLabels']:
            axes_actor.SetXAxisLabelVisibility(1)
        else:
            axes_actor.SetXAxisLabelVisibility(0)

        if scene_metadata['ShowVerticalAxesLabels']:
            axes_actor.SetYAxisLabelVisibility(1)
        else:
            axes_actor.SetYAxisLabelVisibility(0)

        # axesActor.SetAxisLabels(1,[0,10])

        # this was causing problems when x and y dimensions were different
        # axesActor.SetXLabelFormat("%6.4g")
        # axesActor.SetYLabelFormat("%6.4g")

        axes_actor.SetBounds(0, horizontal_length, 0, vertical_length, 0, 0)

        axes_actor.SetXTitle(horizontal_label)
        axes_actor.SetYTitle(vertical_label)
        # axesActor.SetFlyModeToOuterEdges()

        label_prop = axes_actor.GetLabelTextProperty(0)
        # print 'label_prop=',label_prop

        # print 'axesActor.GetXTitle()=',axesActor.GetXTitle()
        title_prop_x = axes_actor.GetTitleTextProperty(0)
        # title_prop_x.SetLineOffset()
        # print 'axesActor.GetTitleTextProperty(0)=',axesActor.GetTitleTextProperty(0)

        # axesActor.GetTitleTextProperty(0).SetFontSize(50)

        # axesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
        # axesActor.GetTitleTextProperty(0).SetFontSize(6.0)
        # axesActor.GetTitleTextProperty(0).SetLineSpacing(0.5)
        # print 'axesActor.GetTitleTextProperty(0)=',axesActor.GetTitleTextProperty(0)

        axes_actor.XAxisMinorTickVisibilityOff()
        axes_actor.YAxisMinorTickVisibilityOff()

        axes_actor.SetTickLocationToOutside()

        axes_actor.GetTitleTextProperty(0).SetColor(axes_color)
        axes_actor.GetLabelTextProperty(0).SetColor(axes_color)

        axes_actor.GetXAxesLinesProperty().SetColor(axes_color)
        axes_actor.GetYAxesLinesProperty().SetColor(axes_color)
        # axesActor.GetLabelTextProperty(0).SetColor(axes_color)

        axes_actor.GetTitleTextProperty(1).SetColor(axes_color)
        axes_actor.GetLabelTextProperty(1).SetColor(axes_color)

        # axesActor.DrawXGridlinesOn()
        # axesActor.DrawYGridlinesOn()
        # axesActor.XAxisVisibilityOn()
        # axesActor.YAxisVisibilityOn()

        # print 'axesActor.GetViewAngleLODThreshold()=',axesActor.GetViewAngleLODThreshold()
        # axesActor.SetViewAngleLODThreshold(1.0)

        # print 'axesActor.GetEnableViewAngleLOD()=',axesActor.GetEnableViewAngleLOD()
        # axesActor.SetEnableViewAngleLOD(0)
        #
        # print 'axesActor.GetEnableDistanceLOD()=',axesActor.GetEnableDistanceLOD()
        # axesActor.SetEnableDistanceLOD(0)

        # axesActor.SetLabelFormat("%6.4g")
        # axesActor.SetFlyModeToOuterEdges()
        # axesActor.SetFlyModeToNone()
        # axesActor.SetFontFactor(1.5)

        # axesActor.SetXAxisVisibility(1)
        # axesActor.SetYAxisVisibility(1)
        # axesActor.SetZAxisVisibility(0)

        # axesActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
        # axesActor.GetProperty().SetColor(color)

        # xAxisActor = axesActor.GetXAxisActor2D()
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetRulerDistance(40)
        # xAxisActor.SetRulerMode(20)
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetNumberOfMinorTicks(3)

    def init_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes vector field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        dim = self.planeMapper(dim_order,
                               (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        field_type = drawing_params.fieldType.lower()
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
        # lattice_type_str = self.get_lattice_type_str()
        #
        # if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
        if self.is_lattice_hex(drawing_params=drawing_params):
            if field_type == 'vectorfield':
                fill_successful = self.field_extractor.fillVectorFieldData2DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos
                )
            elif field_type == 'vectorfieldcelllevel':
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData2DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos
                )
        else:
            if field_type == 'vectorfield':
                fill_successful = self.field_extractor.fillVectorFieldData2D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos
                )
            elif field_type == 'vectorfieldcelllevel':
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData2D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                    self.currentDrawingParameters.plane,
                    self.currentDrawingParameters.planePos
                )

        if not fill_successful:
            return

        vector_grid.SetPoints(points)
        vector_grid.GetPointData().SetVectors(vectors)

        cone = vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        # cone.SetRadius(4)

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_magnitude_fixed = min_max_dict['MinRangeFixed']
        max_magnitude_fixed = min_max_dict['MaxRangeFixed']
        min_magnitude_read = min_max_dict['MinRange']
        max_magnitude_read = min_max_dict['MaxRange']

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
        arrowScalingFactor = scene_metadata['ArrowLength']

        vector_field_actor = actors_dict['vector_field_actor']
        if mdata.get('FixedArrowColorOn', data_type='bool'):
            glyphs.SetScaleModeToScaleByVector()
            # rangeSpan = maxMagnitude - minMagnitude
            dataScalingFactor = max(abs(min_magnitude), abs(max_magnitude))
            #            print MODULENAME,"initVectorFieldCellLevelActors():  self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor == 0.0:
                dataScalingFactor = 1.0  # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)
            # coloring arrows
            # arrow_color = to_vtk_rgb(scene_metadata['ArrowColor'])

            arrow_color = to_vtk_rgb(mdata.get('ArrowColor', data_type='color'))
            vector_field_actor.GetProperty().SetColor(arrow_color)



        else:

            if mdata.get('ScaleArrowsOn', data_type='bool'):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                rangeSpan = max_magnitude - min_magnitude
                dataScalingFactor = max(abs(min_magnitude), abs(max_magnitude))
                #                print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

                if dataScalingFactor == 0.0:
                    dataScalingFactor = 1.0  # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)

        self.glyphs_mapper.SetScalarRange([min_magnitude, max_magnitude])

        vector_field_actor.SetMapper(self.glyphs_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

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
        if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
            self.init_concentration_field_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            if self.pixelized_cartesian_field:
                self.init_concentration_field_actors_cartesian_pixelized(actor_specs=actor_specs,
                                                                         drawing_params=drawing_params)
            else:

                self.init_concentration_field_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)

        if mdata.get('LegendEnable', default=True):
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
        dim = self.planeMapper(dim_order,
                               (field_dim.x, field_dim.y, field_dim.z))
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

        field_type = drawing_params.fieldType.lower()
        if field_type == 'confield':
            fill_successful = self.field_extractor.fillConFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        else:
            print(("unsuported field type {}".format(field_type)))
            return

        if not fill_successful:
            return

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict['MinRangeFixed']
        max_range_fixed = min_max_dict['MaxRangeFixed']
        min_range = min_max_dict['MinRange']
        max_range = min_max_dict['MaxRange']

        range_array = con_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        if mdata.get('ContoursOn', default=False):
            contour_actor = actors_dict['contour_actor']
            num_contour_lines = mdata.get('NumberOfContourLines', default=3)
            self.initialize_contours_pixelized([dim[0], dim[1]], con_array, [min_con, max_con],
                                               contour_actor, num_contour_lines=num_contour_lines, hex_flag=True)

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

        concentration_actor = actors_dict['concentration_actor']

        concentration_actor.SetMapper(self.con_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper': self.con_mapper}
        else:
            actor_specs.metadata['mapper'] = self.con_mapper

    def initialize_contours_pixelized(self, dim, con_array, min_max, contour_actor, num_contour_lines=2,
                                      hex_flag=False):
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

        iso_contour.GenerateValues(
            num_contour_lines + 2,
            min_max)

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
        dim = self.planeMapper(dim_order,
                               (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]
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

        field_type = drawing_params.fieldType.lower()
        if field_type == 'confield':
            fill_successful = self.field_extractor.fillConFieldData2DCartesian(
                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData2DCartesian(

                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2DCartesian(
                con_array_int_addr,
                cells_con_int_addr,
                points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )

        else:
            print(("unsuported field type {}".format(field_type)))
            return

        if not fill_successful:
            return

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict['MinRangeFixed']
        max_range_fixed = min_max_dict['MaxRangeFixed']
        min_range = min_max_dict['MinRange']
        max_range = min_max_dict['MaxRange']

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

        if mdata.get('ContoursOn', default=False):
            contour_actor = actors_dict['contour_actor']
            num_contour_lines = mdata.get('NumberOfContourLines', default=3)
            self.initialize_contours_pixelized([dim[0], dim[1]], con_array, [min_con, max_con],
                                               contour_actor, num_contour_lines=num_contour_lines, hex_flag=False)

        self.con_mapper.ScalarVisibilityOn()
        self.con_mapper.SetLookupTable(self.clut)
        self.con_mapper.SetScalarRange(min_con, max_con)

        concentration_actor = actors_dict['concentration_actor']

        concentration_actor.SetMapper(self.con_mapper)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper': self.con_mapper}
        else:
            actor_specs.metadata['mapper'] = self.con_mapper

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
        dim = self.planeMapper(dim_order,
                               (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(vtkObj=con_array)
        # # todo - make it flexible

        field_type = drawing_params.fieldType.lower()
        if field_type == 'confield':
            fill_successful = self.field_extractor.fillConFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type == 'scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )

        else:
            print(("unsuported field type {}".format(field_type)))
            return

        if not fill_successful:
            return

        # # todo 5 - revisit later
        # numIsos = Configuration.getSetting("NumberOfContourLines", field_name)
        # #        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict['MinRangeFixed']
        max_range_fixed = min_max_dict['MaxRangeFixed']
        min_range = min_max_dict['MinRange']
        max_range = min_max_dict['MaxRange']

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

        if mdata.get('ContoursOn', default=False):
            contour_actor = actors_dict['contour_actor']
            num_contour_lines = mdata.get('NumberOfContourLines', default=3)
            self.initialize_contours_cartesian(
                field_image_data,
                [min_con, max_con],
                contour_actor,
                num_contour_lines=num_contour_lines,
                scene_metadata=scene_metadata
            )

        self.clut.SetTableRange([min_con, max_con])

        self.con_mapper.SetInputConnection(field_image_data.GetOutputPort())  # port index = 0

        self.con_mapper.ScalarVisibilityOn()

        self.con_mapper.SetLookupTable(self.clut)
        # 0, self.clut.GetNumberOfColors()) # may manually set range so that
        # type reassignment will not be scalled dynamically when one type is missing
        self.con_mapper.SetScalarRange(min_con, max_con)

        self.con_mapper.SetScalarModeToUsePointData()

        concentration_actor = actors_dict['concentration_actor']
        concentration_actor.SetMapper(self.con_mapper)  # concentration actor

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper': self.con_mapper}
        else:
            actor_specs.metadata['mapper'] = self.con_mapper

    def initialize_contours_cartesian(self, field_image_data, min_max, contour_actor, num_contour_lines=2,
                                      scene_metadata=None):

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
        #        isoContour.GenerateValues(Configuration.getSetting("NumberOfContourLines",self.currentDrawingParameters.fieldName)+2, [self.minCon, self.maxCon])

        self.contour_mapper.SetInputConnection(iso_contour.GetOutputPort())
        self.contour_mapper.SetLookupTable(self.ctlut)
        self.contour_mapper.SetScalarRange(min_con, max_con)
        # this is required to do a SetColor on the actor's property
        self.contour_mapper.ScalarVisibilityOff()

        contour_actor.SetMapper(self.contour_mapper)

        color = to_vtk_rgb(mdata.get('ContourColor', data_type='color'))
        contour_actor.GetProperty().SetColor(color)

    def init_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
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
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

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
            self.currentDrawingParameters.planePos
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

        cells_actor = actors_dict['cellsActor']
        cells_actor.SetMapper(self.hex_cells_mapper)

    def init_cell_field_actors_cartesian_pixelized(self, actor_specs, drawing_params=None):
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
        mdata = MetadataHandler(mdata=scene_metadata)

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
            self.currentDrawingParameters.planePos
        )

        cartesian_cells_poly_data.GetCellData().SetScalars(cell_type_array)
        cartesian_cells_poly_data.SetPoints(cartesian_points)
        cartesian_cells_poly_data.SetPolys(cartesian_cells_array)

        if VTK_MAJOR_VERSION >= 6:
            self.cellsMapper.SetInputData(cartesian_cells_poly_data)
        else:
            self.cellsMapper.SetInput(cartesian_cells_poly_data)

        cell_type_lut = self.get_type_lookup_table()
        cell_type_lut_max = cell_type_lut.GetNumberOfTableValues() - 1

        self.cellsMapper.ScalarVisibilityOn()
        self.cellsMapper.SetLookupTable(cell_type_lut)
        self.cellsMapper.SetScalarRange(0, cell_type_lut_max)

        cells_actor = actors_dict['cellsActor']
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
        mdata = MetadataHandler(mdata=scene_metadata)

        # [fieldDim.x, fieldDim.y, fieldDim.z]
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")

        cell_type_int_addr = extract_address_int_from_vtk_object(vtkObj=cell_type_array)

        self.field_extractor.fillCellFieldData2D(
            cell_type_int_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos
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

        cells_actor = actors_dict['cellsActor']
        cells_actor.SetMapper(self.cellsMapper)
        # cells_actor.GetProperty().SetInterpolationToFlat()

    def init_borders_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
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
            self.field_extractor.fillBorderData2DHex(points_int_addr, lines_int_addr,
                                                     self.currentDrawingParameters.plane,
                                                     self.currentDrawingParameters.planePos)
        else:
            self.field_extractor.fillBorderData2D(points_int_addr, lines_int_addr, self.currentDrawingParameters.plane,
                                                  self.currentDrawingParameters.planePos)

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.borderMapper.SetInputData(borders)
        else:
            self.borderMapper.SetInput(borders)

        border_actor = actor_specs.actors_dict['border_actor']
        # actors = list(actor_specs.actors_dict.values())

        border_actor.SetMapper(self.borderMapper)

        border_color = to_vtk_rgb(mdata.get('BorderColor', data_type='color'))
        # coloring borders
        border_actor.GetProperty().SetColor(*border_color)

    def init_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
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
                self.currentDrawingParameters.planePos
            )
        else:
            self.field_extractor.fillClusterBorderData2D(
                points_int_addr,
                lines_int_addr,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.clusterBorderMapper.SetInputData(borders)
        else:
            self.clusterBorderMapper.SetInput(borders)

        cluster_border_actor = actor_specs.actors_dict['cluster_border_actor']

        cluster_border_actor.SetMapper(self.clusterBorderMapper)

        cluster_border_color = to_vtk_rgb(mdata.get('ClusterBorderColor', data_type='color'))
        # coloring borders
        cluster_border_actor.GetProperty().SetColor(*cluster_border_color)

    def init_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        initializes fpp links actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
        # if (fppPlugin == 0):  # bogus check
        if not fppPlugin:  # bogus check
            print('    fppPlugin is null, returning')
            return

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        xdim = field_dim.x
        ydim = field_dim.y

        try:
            cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
            inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        except AttributeError:
            raise AttributeError('Could not access Potts object')

        cellList = CellList(inventory)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        beginPt = 0
        lineNum = 0

        for cell in cellList:
            vol = cell.volume
            if vol < self.eps: continue

            xmid0 = cell.xCOM
            ymid0 = cell.yCOM

            points.InsertNextPoint(xmid0, ymid0, 0)
            endPt = beginPt + 1

            for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
                xmid = fppd.neighborAddress.xCOM
                ymid = fppd.neighborAddress.yCOM

                xdiff = xmid - xmid0
                ydiff = ymid - ymid0
                actualDist = math.sqrt((xdiff * xdiff) + (ydiff * ydiff))
                if actualDist > fppd.maxDistance:  # implies we have wraparound (via periodic BCs)
                    # add dangling "out" line to beginning cell
                    if abs(xdiff) > abs(ydiff):  # wraps around in x-direction
                        if xdiff < 0:
                            xmid0end = xmid0 + self.stubSize
                        else:
                            xmid0end = xmid0 - self.stubSize
                        ymid0end = ymid0
                        points.InsertNextPoint(xmid0end, ymid0end, 0)
                        lines.InsertNextCell(2)  # our line has 2 points
                        lines.InsertCellPoint(beginPt)
                        lines.InsertCellPoint(endPt)

                        actualDist = xdim - actualDist  # compute (approximate) real actualDist
                        lineNum += 1
                        endPt += 1
                    else:  # wraps around in y-direction
                        xmid0end = xmid0
                        if ydiff < 0:
                            ymid0end = ymid0 + self.stubSize
                        else:
                            ymid0end = ymid0 - self.stubSize
                        points.InsertNextPoint(xmid0end, ymid0end, 0)
                        lines.InsertNextCell(2)  # our line has 2 points
                        lines.InsertCellPoint(beginPt)
                        lines.InsertCellPoint(endPt)

                        actualDist = ydim - actualDist  # compute (approximate) real actualDist

                        lineNum += 1

                        endPt += 1

                # link didn't wrap around on lattice
                else:
                    points.InsertNextPoint(xmid, ymid, 0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    lineNum += 1
                    endPt += 1
            for fppd in FocalPointPlasticityDataList(fppPlugin, cell):

                xmid = fppd.neighborAddress.xCOM
                ymid = fppd.neighborAddress.yCOM

                xdiff = xmid - xmid0
                ydiff = ymid - ymid0
                actualDist = math.sqrt((xdiff * xdiff) + (ydiff * ydiff))
                if actualDist > fppd.maxDistance:  # implies we have wraparound (via periodic BCs)

                    # add dangling "out" line to beginning cell
                    if abs(xdiff) > abs(ydiff):  # wraps around in x-direction
                        #                    print '>>>>>> wraparound X'
                        if xdiff < 0:
                            xmid0end = xmid0 + self.stubSize
                        else:
                            xmid0end = xmid0 - self.stubSize
                        ymid0end = ymid0
                        points.InsertNextPoint(xmid0end, ymid0end, 0)
                        lines.InsertNextCell(2)  # our line has 2 points
                        lines.InsertCellPoint(beginPt)
                        lines.InsertCellPoint(endPt)

                        # coloring the FPP links
                        actualDist = xdim - actualDist  # compute (approximate) real actualDist

                        lineNum += 1

                        endPt += 1
                    else:  # wraps around in y-direction
                        xmid0end = xmid0
                        if ydiff < 0:
                            ymid0end = ymid0 + self.stubSize
                        else:
                            ymid0end = ymid0 - self.stubSize
                        points.InsertNextPoint(xmid0end, ymid0end, 0)
                        lines.InsertNextCell(2)  # our line has 2 points
                        lines.InsertCellPoint(beginPt)
                        lines.InsertCellPoint(endPt)

                        # coloring the FPP links
                        actualDist = ydim - actualDist  # compute (approximate) real actualDist

                        lineNum += 1

                        endPt += 1

                # link didn't wrap around on lattice
                else:
                    points.InsertNextPoint(xmid, ymid, 0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    lineNum += 1
                    endPt += 1
            beginPt = endPt  # update point index

        # -----------------------
        if lineNum == 0:
            return

        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)

        fpp_links_actor = actors_dict['fpp_links_actor']

        if VTK_MAJOR_VERSION >= 6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:
            FPPLinksPD.Update()
            self.FPPLinksMapper.SetInput(FPPLinksPD)

        fpp_links_actor.SetMapper(self.FPPLinksMapper)
        fpp_links_color = to_vtk_rgb(mdata.get('FPPLinksColor', data_type='color'))
        # coloring borders
        fpp_links_actor.GetProperty().SetColor(*fpp_links_color)

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

        if (z % 2):
            if (y % 2):
                return [x, sqrt(3.0) / 2.0 * (y + 2.0 / 3.0), z * sqrt(6.0) / 3.0]
            else:
                return [x + 0.5, sqrt(3.0) / 2.0 * (y + 2.0 / 3.0), z * sqrt(6.0) / 3.0]

        else:
            if (y % 2):
                return [x, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0]
            else:
                return [x + 0.5, sqrt(3.0) / 2.0 * y, z * sqrt(6.0) / 3.0]
