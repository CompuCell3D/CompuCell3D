from vtk.util.numpy_support import vtk_to_numpy
from .MVCDrawModelBase import MVCDrawModelBase
import vtk
import numpy as np
import math
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object, to_vtk_rgb
from cc3d.core.GraphicsOffScreen.MetadataHandler import MetadataHandler
from cc3d.cpp import PlayerPython
from cc3d.core.iterators import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
from cc3d.cpp import CompuCell
import sys

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()
epsilon = sys.float_info.epsilon
MODULENAME = '------  MVCDrawModel3D.py'


class MVCDrawModel3D(MVCDrawModelBase):
    def __init__(self, boundary_strategy=None, ren=None):
        MVCDrawModelBase.__init__(self, boundary_strategy=boundary_strategy, ren=ren)

        self.initArea()
        self.setParams()

        self.usedDraw3DFlag = False

    # Sets up the VTK simulation area 
    def initArea(self):
        # Zoom items
        self.zitems = []

        self.outlineDim = [0, 0, 0]

        self.numberOfTableColors = 1024
        self.scalarLUT = vtk.vtkLookupTable()
        self.scalarLUT.SetHueRange(0.67, 0.0)
        self.scalarLUT.SetSaturationRange(1.0, 1.0)
        self.scalarLUT.SetValueRange(1.0, 1.0)
        self.scalarLUT.SetAlphaRange(1.0, 1.0)
        self.scalarLUT.SetNumberOfColors(self.numberOfTableColors)
        self.scalarLUT.Build()

        self.lowTableValue = self.scalarLUT.GetTableValue(0)
        self.highTableValue = self.scalarLUT.GetTableValue(self.numberOfTableColors - 1)

        ## Set up the mapper and actor (3D) for concentration field.
        self.conMapper = vtk.vtkPolyDataMapper()
        # self.conActor = vtk.vtkActor()

        # self.glyphsActor=vtk.vtkActor()
        self.glyphsMapper = vtk.vtkPolyDataMapper()

        self.cellGlyphsMapper = vtk.vtkPolyDataMapper()
        self.FPPLinksMapper = vtk.vtkPolyDataMapper()

        # Weird attributes
        # self.typeActors             = {} # vtkActor
        self.smootherFilters = {}  # vtkSmoothPolyDataFilter
        self.polyDataNormals = {}  # vtkPolyDataNormals
        self.typeExtractors = {}  # vtkDiscreteMarchingCubes
        self.typeExtractorMappers = {}  # vtkPolyDataMapper

    # def setDim(self, fieldDim):
    #     # self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
    #     self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    def is_lattice_hex(self, drawing_params):
        """
        returns if flag that states if the lattice is hex or not. Notice
        In 2D we may use cartesian coordinates for certain projections
        :return: {bool}
        """
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            return True
        else:
            return False

    def init_cell_field_actors_borderless(self, actor_specs, drawing_params=None):

        hex_flag = False
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            hex_flag = True

        # todo 5 - check if this should be called earlier
        # self.extractCellFieldData() # initializes self.usedCellTypesList

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        cell_type_image_data = vtk.vtkImageData()

        cell_type_image_data.SetDimensions(field_dim.x + 2, field_dim.y + 2,
                                           field_dim.z + 2)  # adding 1 pixel border around the lattice to make rendering smooth at lattice borders
        cell_type_image_data.GetPointData().SetScalars(self.cell_type_array)
        voi = vtk.vtkExtractVOI()

        if VTK_MAJOR_VERSION >= 6:
            voi.SetInputData(cell_type_image_data)
        else:
            voi.SetInput(cell_type_image_data)

        #        voi.SetVOI(1,self.dim[0]-1, 1,self.dim[1]-1, 1,self.dim[2]-1 )  # crop out the artificial boundary layer that we created
        voi.SetVOI(0, 249, 0, 189, 0, 170)

        # # todo 5- check if it is possible to call it once
        # self.usedCellTypesList = self.extractCellFieldData()

        number_of_actors = len(self.used_cell_types_list)

        # creating and initializing filters, smoothers and mappers - one for each cell type

        filterList = [vtk.vtkDiscreteMarchingCubes() for i in range(number_of_actors)]
        smootherList = [vtk.vtkSmoothPolyDataFilter() for i in range(number_of_actors)]
        normalsList = [vtk.vtkPolyDataNormals() for i in range(number_of_actors)]
        mapperList = [vtk.vtkPolyDataMapper() for i in range(number_of_actors)]

        # actorCounter=0
        # for i in usedCellTypesList:
        for actorCounter, actor_number in enumerate(self.used_cell_types_list):
            # for actorCounter in xrange(len(self.usedCellTypesList)):

            if VTK_MAJOR_VERSION >= 6:
                filterList[actorCounter].SetInputData(cell_type_image_data)
            else:
                filterList[actorCounter].SetInput(cell_type_image_data)

            #            filterList[actorCounter].SetInputConnection(voi.GetOutputPort())

            # filterList[actorCounter].SetValue(0, usedCellTypesList[actorCounter])
            filterList[actorCounter].SetValue(0, self.used_cell_types_list[actorCounter])
            smootherList[actorCounter].SetInputConnection(filterList[actorCounter].GetOutputPort())
            #            smootherList[actorCounter].SetNumberOfIterations(200)
            normalsList[actorCounter].SetInputConnection(smootherList[actorCounter].GetOutputPort())
            normalsList[actorCounter].SetFeatureAngle(45.0)
            mapperList[actorCounter].SetInputConnection(normalsList[actorCounter].GetOutputPort())
            mapperList[actorCounter].ScalarVisibilityOff()

            actors_dict = actor_specs.actors_dict

            cell_type_lut = self.get_type_lookup_table()
            cell_type_lut_max = cell_type_lut.GetNumberOfTableValues() - 1

            if actor_number in list(actors_dict.keys()):
                actor = actors_dict[actor_number]
                actor.SetMapper(mapperList[actorCounter])

                actor.GetProperty().SetDiffuseColor(
                    cell_type_lut.GetTableValue(self.used_cell_types_list[actorCounter])[0:3])

                # actor.GetProperty().SetDiffuseColor(
                #     # self.celltypeLUT.GetTableValue(self.usedCellTypesList[actorCounter])[0:3])
                #     self.celltypeLUT.GetTableValue(actor_number)[0:3])
                if hex_flag:
                    actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)
                    # actor.GetProperty().SetOpacity(0.5)

    def init_cell_field_borders_actors(self, actor_specs, drawing_params=None):
        """
        initializes cell field actors where each cell is rendered individually as a separate spatial domain
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        field_dim = self.currentDrawingParameters.bsd.fieldDim

        hex_flag = self.is_lattice_hex(drawing_params=drawing_params)

        cell_type_image_data = vtk.vtkImageData()

        # adding 1 pixel border around the lattice to make rendering smooth at lattice borders
        cell_type_image_data.SetDimensions(field_dim.x + 2, field_dim.y + 2, field_dim.z + 2)

        cell_type_image_data.GetPointData().SetScalars(self.cell_id_array)

        # create a different actor for each cell type
        number_of_actors = len(self.used_cell_types_list)

        # creating and initializing filters, smoothers and mappers - one for each cell type
        filter_list = [vtk.vtkDiscreteMarchingCubes() for i in range(number_of_actors)]
        smoother_list = [vtk.vtkSmoothPolyDataFilter() for i in range(number_of_actors)]
        normals_list = [vtk.vtkPolyDataNormals() for i in range(number_of_actors)]
        mapper_list = [vtk.vtkPolyDataMapper() for i in range(number_of_actors)]

        for actor_counter, actor_number in enumerate(self.used_cell_types_list):

            if VTK_MAJOR_VERSION >= 6:
                filter_list[actor_counter].SetInputData(cell_type_image_data)
            else:
                filter_list[actor_counter].SetInput(cell_type_image_data)

            if self.used_cell_types_list[actor_counter] >= 1:
                ct_all = vtk_to_numpy(self.cell_type_array)
                cid_all = vtk_to_numpy(self.cell_id_array)

                cid_unique = np.unique(cid_all[ct_all == actor_number])

                for idx in range(len(cid_unique)):
                    filter_list[actor_counter].SetValue(idx, cid_unique[idx])

            else:
                filter_list[actor_counter].SetValue(0, 13)  # rwh: what the??

            smoother_list[actor_counter].SetInputConnection(filter_list[actor_counter].GetOutputPort())
            normals_list[actor_counter].SetInputConnection(smoother_list[actor_counter].GetOutputPort())
            normals_list[actor_counter].SetFeatureAngle(45.0)
            mapper_list[actor_counter].SetInputConnection(normals_list[actor_counter].GetOutputPort())
            mapper_list[actor_counter].ScalarVisibilityOff()

            actors_dict = actor_specs.actors_dict
            if actor_number in list(actors_dict.keys()):
                actor = actors_dict[actor_number]
                actor.SetMapper(mapper_list[actor_counter])

                cell_type_lut = self.get_type_lookup_table()

                actor.GetProperty().SetDiffuseColor(cell_type_lut.GetTableValue(actor_number)[0:3])

                if hex_flag:
                    actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

    # original rendering technique (and still used if Vis->Cell Borders not checked) - vkDiscreteMarchingCubes
    # on celltype
    def init_cell_field_actors(self, actor_specs, drawing_params=None):

        if drawing_params.screenshot_data.cell_borders_on:
            self.init_cell_field_borders_actors(actor_specs=actor_specs)
        else:
            self.init_cell_field_actors_borderless(actor_specs=actor_specs)

    def init_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """

        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        # dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        # dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
        dim = [field_dim.x, field_dim.y, field_dim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        try:
            isovalues = mdata.get('ScalarIsoValues', default=[])
            isovalues = list([float(x) for x in isovalues])
        except:
            print('Could not process isovalue list ')
            isovalues = []

        try:
            numIsos = mdata.get('NumberOfContourLines', default=3)
        except:
            print('could not process NumberOfContourLines setting')
            numIsos = 0

        hex_flag = False
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            hex_flag = True

        types_invisible = PlayerPython.vectorint()
        for type_label in drawing_params.screenshot_data.invisible_types:
            types_invisible.append(int(type_label))

        # self.isovalStr = Configuration.getSetting("ScalarIsoValues", field_name)
        # if type(self.isovalStr) == QVariant:
        #     self.isovalStr = str(self.isovalStr.toString())
        # else:
        #     self.isovalStr = str(self.isovalStr)

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(vtkObj=con_array)

        cell_type_con = vtk.vtkIntArray()
        cell_type_con.SetName("concelltype")
        cell_type_con_int_addr = extract_address_int_from_vtk_object(vtkObj=cell_type_con)

        field_type = drawing_params.fieldType.lower()
        if field_type == 'confield':
            fill_successful = self.field_extractor.fillConFieldData3D(con_array_int_addr, cell_type_con_int_addr,
                                                                      field_name, types_invisible)
        elif field_type == 'scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData3D(con_array_int_addr, cell_type_con_int_addr,
                                                                         field_name, types_invisible)
        elif field_type == 'scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData3D(con_array_int_addr,
                                                                                  cell_type_con_int_addr, field_name,
                                                                                  types_invisible)

        if not fill_successful:
            return

        range_array = con_array.GetRange()
        min_con = range_array[0]
        max_con = range_array[1]
        field_max = range_array[1]
        #        print MODULENAME, '  initScalarFieldDataActors(): min,maxCon=',self.minCon,self.maxCon

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict['MinRangeFixed']
        max_range_fixed = min_max_dict['MaxRangeFixed']
        min_range = min_max_dict['MinRange']
        max_range = min_max_dict['MaxRange']

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        uGrid = vtk.vtkStructuredPoints()
        uGrid.SetDimensions(dim[0] + 2, dim[1] + 2, dim[
            2] + 2)  # only add 2 if we're filling in an extra boundary (rf. FieldExtractor.cpp)
        #        uGrid.SetDimensions(self.dim[0],self.dim[1],self.dim[2])
        #        uGrid.GetPointData().SetScalars(self.cellTypeCon)   # cellType scalar field
        uGrid.GetPointData().SetScalars(con_array)
        #        uGrid.GetPointData().AddArray(self.conArray)        # additional scalar field

        voi = vtk.vtkExtractVOI()
        ##        voi.SetInputConnection(uGrid.GetOutputPort())
        #        voi.SetInput(uGrid.GetOutput())

        if VTK_MAJOR_VERSION >= 6:
            voi.SetInputData(uGrid)
        else:
            voi.SetInput(uGrid)

        voi.SetVOI(1, dim[0] - 1, 1, dim[1] - 1, 1,
                   dim[2] - 1)  # crop out the artificial boundary layer that we created

        isoContour = vtk.vtkContourFilter()
        # skinExtractorColor = vtk.vtkDiscreteMarchingCubes()
        # skinExtractorColor = vtk.vtkMarchingCubes()
        #        isoContour.SetInput(uGrid)
        isoContour.SetInputConnection(voi.GetOutputPort())

        isoNum = 0
        for isoNum, isoVal in enumerate(isovalues):
            try:
                isoContour.SetValue(isoNum, isoVal)
            except:
                print(MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ', self.isovalStr[idx])

        if isoNum > 0:
            isoNum += 1

        delIso = (max_con - min_con) / (numIsos + 1)  # exclude the min,max for isovalues
        isoVal = min_con + delIso
        for idx in range(numIsos):
            isoContour.SetValue(isoNum, isoVal)
            isoNum += 1
            isoVal += delIso

        # UGLY hack to NOT display anything since our attempt to RemoveActor (below) don't seem to work
        if isoNum == 0:
            isoVal = field_max + 1.0  # go just outside valid range
            isoContour.SetValue(isoNum, isoVal)

        #        concLut = vtk.vtkLookupTable()
        # concLut.SetTableRange(conc_vol.GetScalarRange())
        #        concLut.SetTableRange([self.minCon,self.maxCon])
        self.scalarLUT.SetTableRange([min_con, max_con])
        #        concLut.SetNumberOfColors(256)
        #        concLut.Build()
        # concLut.SetTableValue(39,0,0,0,0)

        #        skinColorMapper = vtk.vtkPolyDataMapper()
        # skinColorMapper.SetInputConnection(skinNormals.GetOutputPort())
        #        self.conMapper.SetInputConnection(skinExtractorColor.GetOutputPort())
        self.conMapper.SetInputConnection(isoContour.GetOutputPort())
        self.conMapper.ScalarVisibilityOn()
        self.conMapper.SetLookupTable(self.scalarLUT)
        # # # print " this is conc_vol.GetScalarRange()=",conc_vol.GetScalarRange()
        # self.conMapper.SetScalarRange(conc_vol.GetScalarRange())
        self.conMapper.SetScalarRange([min_con, max_con])
        # self.conMapper.SetScalarRange(0,1500)

        # rwh - what does this do?
        #        self.conMapper.SetScalarModeToUsePointFieldData()
        #        self.conMapper.ColorByArrayComponent("concentration",0)

        #        print MODULENAME,"initScalarFieldDataActors():  Plotting 3D Scalar field"
        # self.conMapper      = vtk.vtkPolyDataMapper()
        # self.conActor       = vtk.vtkActor()

        concentration_actor = actors_dict['concentration_actor']

        concentration_actor.SetMapper(self.conMapper)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if hex_flag:
            concentration_actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper': self.conMapper}
        else:
            actor_specs.metadata['mapper'] = self.conMapper

        if mdata.get('LegendEnable', default=False):
            self.init_legend_actors(actor_specs=actor_specs, drawing_params=drawing_params)

    def init_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes vector field actors for cartesian lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        field_name = drawing_params.fieldName
        field_type = drawing_params.fieldType.lower()
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        dim = [field_dim.x, field_dim.y, field_dim.z]

        vector_grid = vtk.vtkUnstructuredGrid()

        points = vtk.vtkPoints()
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        points_int_addr = extract_address_int_from_vtk_object(vtkObj=points)
        vectors_int_addr = extract_address_int_from_vtk_object(vtkObj=vectors)

        fill_successful = False

        hex_flag = False

        if self.is_lattice_hex(drawing_params=drawing_params):
            hex_flag = True
            if field_type == 'vectorfield':
                fill_successful = self.field_extractor.fillVectorFieldData3DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name
                )
            elif field_type == 'vectorfieldcelllevel':
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData3DHex(
                    points_int_addr,
                    vectors_int_addr,
                    field_name
                )
        else:
            if field_type == 'vectorfield':
                fill_successful = self.field_extractor.fillVectorFieldData3D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
                )
            elif field_type == 'vectorfieldcelllevel':
                fill_successful = self.field_extractor.fillVectorFieldCellLevelData3D(
                    points_int_addr,
                    vectors_int_addr,
                    field_name,
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

        range_array = vectors.GetRange(-1)

        min_magnitude = range_array[0]
        max_magnitude = range_array[1]

        min_max_dict = self.get_min_max_metadata(scene_metadata=scene_metadata, field_name=field_name)
        min_range_fixed = min_max_dict['MinRangeFixed']
        max_range_fixed = min_max_dict['MaxRangeFixed']
        min_range = min_max_dict['MinRange']
        max_range = min_max_dict['MaxRange']

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_magnitude = min_range

        if max_range_fixed:
            max_magnitude = max_range

        glyphs = vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION >= 6:
            glyphs.SetInputData(vector_grid)
        else:
            glyphs.SetInput(vector_grid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        # glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # scaling arrows here ArrowLength indicates scaling factor not actual length
        # glyphs.SetScaleFactor(Configuration.getSetting("ArrowLength"))

        vector_field_actor = actors_dict['vector_field_actor']

        # scaling factor for an arrow - ArrowLength indicates scaling factor not actual length
        arrowScalingFactor = mdata.get('ArrowLength', default=1.0)

        if mdata.get('FixedArrowColorOn', default=False):
            glyphs.SetScaleModeToScaleByVector()

            dataScalingFactor = max(abs(min_magnitude), abs(max_magnitude))

            if dataScalingFactor == 0.0:
                # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                dataScalingFactor = 1.0

            glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)

            # coloring arrows
            arrow_color = to_vtk_rgb(mdata.get('ArrowColor', data_type='color'))
            vector_field_actor.GetProperty().SetColor(arrow_color)

        else:
            glyphs.SetColorModeToColorByVector()
            glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphsMapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphsMapper.SetLookupTable(self.scalarLUT)

        self.glyphsMapper.SetScalarRange([min_magnitude, max_magnitude])

        vector_field_actor.SetMapper(self.glyphsMapper)

        self.init_min_max_actor(min_max_actor=actors_dict['min_max_text_actor'], range_array=range_array)

        if hex_flag:
            vector_field_actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

    def init_outline_actors(self, actor_specs, drawing_params=None):
        """
        Initializes outline actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        outline_data = vtk.vtkImageData()

        outline_data.SetDimensions(field_dim.x + 1, field_dim.y + 1, field_dim.z + 1)

        outline = vtk.vtkOutlineFilter()

        if VTK_MAJOR_VERSION >= 6:
            outline.SetInputData(outline_data)
        else:
            outline.SetInput(outline_data)

        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())

        outline_actor = actors_dict['outline_actor']

        outline_actor.SetMapper(outline_mapper)

        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() == 'hexagonal':
        if self.is_lattice_hex(drawing_params=drawing_params):
            outline_actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

        outline_color = to_vtk_rgb(mdata.get('BoundingBoxColor', data_type='color'))
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
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        axes_actor = actors_dict['axes_actor']
        axes_color = to_vtk_rgb(mdata.get('AxesColor', data_type='color'))

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(axes_color)
        tprop.ShadowOn()

        axes_actor.SetNumberOfLabels(4)  # number of labels

        # lattice_type_str = self.get_lattice_type_str()
        # if lattice_type_str.lower() == 'hexagonal':
        if self.is_lattice_hex(drawing_params=drawing_params):
            axes_actor.SetBounds(0, field_dim.x, 0, field_dim.y * math.sqrt(3.0) / 2.0, 0,
                                 field_dim.z * math.sqrt(6.0) / 3.0)
        else:
            axes_actor.SetBounds(0, field_dim.x, 0, field_dim.y, 0, field_dim.z)

        axes_actor.SetLabelFormat("%6.4g")
        axes_actor.SetFlyModeToOuterEdges()
        axes_actor.SetFontFactor(1.5)

        # axesActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
        axes_actor.GetProperty().SetColor(axes_color)

        xAxisActor = axes_actor.GetXAxisActor2D()
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetRulerDistance(40)
        # xAxisActor.SetRulerMode(20)
        # xAxisActor.RulerModeOn()
        xAxisActor.SetNumberOfMinorTicks(3)

    def init_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        initializes fpp links actors
        :param actor_specs:
        :param drawing_params:
        :return: None
        """

        fpp_plugin = CompuCell.getFocalPointPlasticityPlugin()
        if not fpp_plugin:
            print('    fppPlugin is null, returning')
            return

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        try:
            inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        except AttributeError:
            raise AttributeError('Could not access Potts object')

        cell_list = CellList(inventory)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        pt_counter = 0

        for cell in cell_list:
            vol = cell.volume
            if vol < self.eps:
                continue

            mid_com = np.array([cell.xCOM, cell.yCOM, cell.zCOM], dtype=float)

            for fppd in InternalFocalPointPlasticityDataList(fpp_plugin, cell):
                pt_counter = self.add_link(field_dim=field_dim,
                                           fppd=fppd, mid_com=mid_com, pt_counter=pt_counter,
                                           lines=lines, points=points)

            for fppd in FocalPointPlasticityDataList(fpp_plugin, cell):
                pt_counter = self.add_link(field_dim=field_dim,
                                           fppd=fppd, mid_com=mid_com, pt_counter=pt_counter, lines=lines,
                                           points=points)

        fpp_links_pd = vtk.vtkPolyData()
        fpp_links_pd.SetPoints(points)
        fpp_links_pd.SetLines(lines)

        fpp_links_actor = actors_dict['fpp_links_actor']

        if VTK_MAJOR_VERSION >= 6:
            self.FPPLinksMapper.SetInputData(fpp_links_pd)
        else:
            fpp_links_pd.Update()
            self.FPPLinksMapper.SetInput(fpp_links_pd)

        fpp_links_actor.SetMapper(self.FPPLinksMapper)
        fpp_links_color = to_vtk_rgb(mdata.get('FPPLinksColor', data_type='color'))

        # coloring borders
        fpp_links_actor.GetProperty().SetColor(*fpp_links_color)

    def add_link(self, field_dim, fppd, mid_com, pt_counter, lines, points) -> int:
        """
        Draws single link in 3D. Returns updated point counter

        :param field_dim:
        :param fppd:
        :param mid_com: link begin
        :param pt_counter: point counter
        :param lines: vtk lines structure that holds line specification
        :param points: vtk points sstructure
        :return: end point counter
        """

        n_cell = fppd.neighborAddress

        n_mid_com = np.array([n_cell.xCOM, n_cell.yCOM, n_cell.zCOM], dtype=float)

        naive_actual_dist = np.linalg.norm(mid_com - n_mid_com)
        inv_dist = self.invariant_distance(p1=mid_com, p2=n_mid_com, dim=field_dim)
        if abs((inv_dist - naive_actual_dist) / (naive_actual_dist + epsilon)) > 10 * epsilon:
            # if naive distance is different than invariant distance then we are dealing with link that is wrapped
            # if naive_actual_dist > fppd.maxDistance:  # implies we have wraparound (via periodic BCs)
            # we are drawing links that currently stick out of the lattice.
            inv_dist_vec = self.invariant_distance_vector(p1=mid_com,
                                                          p2=n_mid_com,
                                                          dim=field_dim)

            # inv_dist_vec = self.unconditional_invariant_distance_vector(p1=mid_com,
            #                                                             p2=n_mid_com,
            #                                                             dim=field_dim)
            link_begin = mid_com
            link_end = link_begin + inv_dist_vec

            points.InsertNextPoint(mid_com[0], mid_com[1], mid_com[2])
            points.InsertNextPoint(link_end[0], link_end[1], link_end[2])
            # our line has 2 points
            lines.InsertNextCell(2)
            lines.InsertCellPoint(pt_counter)
            lines.InsertCellPoint(pt_counter + 1)
            pt_counter += 2

        # link didn't wrap around on lattice
        else:
            points.InsertNextPoint(mid_com[0], mid_com[1], mid_com[2])
            points.InsertNextPoint(n_mid_com[0], n_mid_com[1], n_mid_com[2])
            # our line has 2 points
            lines.InsertNextCell(2)
            lines.InsertCellPoint(pt_counter)
            lines.InsertCellPoint(pt_counter + 1)

            pt_counter += 2

        return pt_counter
