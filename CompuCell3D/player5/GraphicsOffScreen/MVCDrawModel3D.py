from vtk.util.numpy_support import vtk_to_numpy
from MVCDrawModelBase import MVCDrawModelBase
import Configuration
import vtk, math

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()
# from Utilities.utils import extract_address_int_from_vtk_object
from CompuCell3D.player5.Utilities.utils import extract_address_int_from_vtk_object, to_vtk_rgb

MODULENAME='------  MVCDrawModel3D.py'


class MVCDrawModel3D(MVCDrawModelBase):
    def __init__(self):
        MVCDrawModelBase.__init__(self)

        self.initArea()
        self.setParams()

        self.usedDraw3DFlag = False

    # Sets up the VTK simulation area 
    def initArea(self):
        # Zoom items
        self.zitems = []

        # self.cellTypeActors={}
        # self.outlineActor = vtk.vtkActor()
        self.outlineDim = [0, 0, 0]

        # self.invisibleCellTypes={}
        # self.typesInvisibleStr=""
        # self.set3DInvisibleTypes()

        # axesActor = vtk.vtkActor()
        # axisTextActor = vtk.vtkFollower()

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
        

    def setDim(self, fieldDim):
        # self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    # def prepareOutlineActors(self, _actors):
    #
    #     outlineData = vtk.vtkImageData()
    #
    #     fieldDim = self.currentDrawingParameters.bsd.fieldDim
    #
    #     outlineData.SetDimensions(fieldDim.x+1,fieldDim.y+1,fieldDim.z+1)
    #
    #     # if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.currentDrawingParameters.plane=="XY":
    #         # import math
    #         # outlineData.SetDimensions(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
    #         # print "self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
    #     # else:
    #         # outlineData.SetDimensions(self.dim[0]+1, self.dim[1]+1, 1)
    #
    #     # outlineDimTmp=_imageData.GetDimensions()
    #     # # print "\n\n\n this is outlineDimTmp=",outlineDimTmp," self.outlineDim=",self.outlineDim
    #     # if self.outlineDim[0] != outlineDimTmp[0] or self.outlineDim[1] != outlineDimTmp[1] or self.outlineDim[2] != outlineDimTmp[2]:
    #         # self.outlineDim=outlineDimTmp
    #
    #     outline = vtk.vtkOutlineFilter()
    #
    #     if VTK_MAJOR_VERSION>=6:
    #         outline.SetInputData(outlineData)
    #     else:
    #         outline.SetInput(outlineData)
    #
    #
    #     outlineMapper = vtk.vtkPolyDataMapper()
    #     outlineMapper.SetInputConnection(outline.GetOutputPort())
    #
    #     _actors[0].SetMapper(outlineMapper)
    #     if self.hexFlag:
    #         _actors[0].SetScale(self.xScaleHex,self.yScaleHex,self.zScaleHex)
    #     _actors[0].GetProperty().SetColor(1, 1, 1)
    #     # self.outlineDim=_imageData.GetDimensions()
    #
    #     color = Configuration.getSetting("BoundingBoxColor")   # eventually do this smarter (only get/update when it changes)
    #     _actors[0].GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)

    def prepareAxesActors(self, _mappers, _actors):

        axesActor=_actors[0]
        color = Configuration.getSetting("AxesColor")   # eventually do this smarter (only get/update when it changes)
        color = (float(color.red())/255,float(color.green())/255,float(color.blue())/255)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(color)
        tprop.ShadowOn()
        dim = self.currentDrawingParameters.bsd.fieldDim

        axesActor.SetNumberOfLabels(4) # number of labels

        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"]:
            axesActor.SetBounds(0, dim.x, 0, dim.y*math.sqrt(3.0)/2.0, 0, dim.z*math.sqrt(6.0)/3.0)
        else:
            axesActor.SetBounds(0, dim.x, 0, dim.y, 0, dim.z)

        axesActor.SetLabelFormat("%6.4g")
        axesActor.SetFlyModeToOuterEdges()
        axesActor.SetFontFactor(1.5)

        # axesActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
        axesActor.GetProperty().SetColor(color)

        xAxisActor = axesActor.GetXAxisActor2D()
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetRulerDistance(40)
        # xAxisActor.SetRulerMode(20)
        # xAxisActor.RulerModeOn()
        xAxisActor.SetNumberOfMinorTicks(3)

        # setting camera fot he actor is vey important to get axes working properly
#         axesActor.SetCamera(self.graphicsFrameWidget.ren.GetActiveCamera())
#         self.graphicsFrameWidget.ren.AddActor(axesActor)

    def init_cell_field_actors_borderless(self,actor_specs, drawing_params=None):

        hex_flag = False
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() =='hexagonal':
            hex_flag = True

        # todo 5 - check if this should be called earlier
        # self.extractCellFieldData() # initializes self.usedCellTypesList

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        cell_type_image_data = vtk.vtkImageData()

        cell_type_image_data.SetDimensions(field_dim.x+2,field_dim.y+2,field_dim.z+2) # adding 1 pixel border around the lattice to make rendering smooth at lattice borders
        cell_type_image_data.GetPointData().SetScalars(self.cell_type_array)
        voi = vtk.vtkExtractVOI()

        if VTK_MAJOR_VERSION>=6:
            voi.SetInputData(cell_type_image_data)
        else:
            voi.SetInput(cell_type_image_data)

#        voi.SetVOI(1,self.dim[0]-1, 1,self.dim[1]-1, 1,self.dim[2]-1 )  # crop out the artificial boundary layer that we created
        voi.SetVOI(0,249, 0,189, 0,170)

        # # todo 5- check if it is possible to call it once
        # self.usedCellTypesList = self.extractCellFieldData()

        number_of_actors = len(self.used_cell_types_list)

        # creating and initializing filters, smoothers and mappers - one for each cell type

        filterList = [vtk.vtkDiscreteMarchingCubes() for i in xrange(number_of_actors)]
        smootherList = [vtk.vtkSmoothPolyDataFilter() for i in xrange(number_of_actors)]
        normalsList = [vtk.vtkPolyDataNormals() for i in xrange(number_of_actors)]
        mapperList = [vtk.vtkPolyDataMapper() for i in xrange(number_of_actors)]

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

            if actor_number in actors_dict.keys():
                actor = actors_dict[actor_number]
                actor.SetMapper(mapperList[actorCounter])

                actor.GetProperty().SetDiffuseColor(self.celltypeLUT.GetTableValue(self.used_cell_types_list[actorCounter])[0:3])

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

        hex_flag = False
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            hex_flag = True

        cell_type_image_data = vtk.vtkImageData()

        # adding 1 pixel border around the lattice to make rendering smooth at lattice borders
        cell_type_image_data.SetDimensions(field_dim.x + 2, field_dim.y + 2, field_dim.z + 2)

        cell_type_image_data.GetPointData().SetScalars(self.cell_id_array)

        # create a different actor for each cell type
        number_of_actors = len(self.used_cell_types_list)

        # creating and initializing filters, smoothers and mappers - one for each cell type
        filter_list = [vtk.vtkDiscreteMarchingCubes() for i in xrange(number_of_actors)]
        smoother_list = [vtk.vtkSmoothPolyDataFilter() for i in xrange(number_of_actors)]
        normals_list = [vtk.vtkPolyDataNormals() for i in xrange(number_of_actors)]
        mapper_list = [vtk.vtkPolyDataMapper() for i in xrange(number_of_actors)]

        for actor_counter, actor_number in enumerate(self.used_cell_types_list):

            if VTK_MAJOR_VERSION >= 6:
                filter_list[actor_counter].SetInputData(cell_type_image_data)
            else:
                filter_list[actor_counter].SetInput(cell_type_image_data)

            if self.used_cell_types_list[actor_counter] >= 1:
                ct_all = vtk_to_numpy(self.cell_type_array)
                cid_all = vtk_to_numpy(self.cell_id_array)

                cid_unique = []
                for idx in range(len(ct_all)):
                    if ct_all[idx] == self.used_cell_types_list[actor_counter]:
                        cid = cid_all[idx]
                        if cid not in cid_unique:
                            cid_unique.append(cid_all[idx])

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
            if actor_number in actors_dict.keys():
                actor = actors_dict[actor_number]
                actor.SetMapper(mapper_list[actor_counter])

                actor.GetProperty().SetDiffuseColor(
                    self.celltypeLUT.GetTableValue(actor_number)[0:3])

                if hex_flag:
                    actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

    # original rendering technique (and still used if Vis->Cell Borders not checked) - vkDiscreteMarchingCubes on celltype
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
        dim = [field_dim.x,field_dim.y, field_dim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata


        try:
            isovalues = scene_metadata['ScalarIsoValues']
            isovalues = list(map(lambda x: float(x), isovalues))
        except:
            print('Could not process isovalue list ')
            isovalues = []

        try:
            numIsos = scene_metadata['NumberOfContourLines']
        except:
            print('could not process NumberOfContourLines setting')
            numIsos = 0

        hex_flag = False
        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() =='hexagonal':
            hex_flag = True

        import PlayerPython

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
        con_array_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=con_array)

        cell_type_con = vtk.vtkIntArray()
        cell_type_con.SetName("concelltype")
        cell_type_con_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=cell_type_con)

        field_type = drawing_params.fieldType.lower()
        if field_type == 'confield':
            fill_successful = self.field_extractor.fillConFieldData3D(con_array_int_addr, cell_type_con_int_addr, field_name,types_invisible)
        elif field_type =='scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData3D(con_array_int_addr, cell_type_con_int_addr, field_name,types_invisible)
        elif field_type =='scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData3D(con_array_int_addr, cell_type_con_int_addr, field_name,types_invisible)

        if not fill_successful:
            return

        range = con_array.GetRange()
        min_con = range[0]
        max_con = range[1]
        field_max = range[1]
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

        # if Configuration.getSetting("MinRangeFixed", field_name):
        #     min_con = Configuration.getSetting("MinRange", field_name)
        #
        # if Configuration.getSetting("MaxRangeFixed", field_name):
        #     max_con = Configuration.getSetting("MaxRange", field_name)

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
                print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ', self.isovalStr[idx]

        if isoNum > 0:
            isoNum += 1

        delIso = (max_con - min_con) / (numIsos + 1)  # exclude the min,max for isovalues
        isoVal = min_con + delIso
        for idx in xrange(numIsos):
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

        if hex_flag:
            concentration_actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

        if actor_specs.metadata is None:
            actor_specs.metadata = {'mapper':self.conMapper}
        else:
            actor_specs.metadata['mapper'] = self.conMapper


        if scene_metadata['LegendEnable']:
            print 'Enabling legend'
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

        dim = [field_dim.x,field_dim.y, field_dim.z]

        scene_metadata = drawing_params.screenshot_data.metadata



        vector_grid = vtk.vtkUnstructuredGrid()

        points = vtk.vtkPoints()
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        points_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=points)
        vectors_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=vectors)


        fill_successful = False
        lattice_type_str = self.get_lattice_type_str()

        hex_flag = False
        if lattice_type_str.lower() == 'hexagonal':
            hex_flag = True
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

        range = vectors.GetRange(-1)

        min_magnitude = range[0]
        max_magnitude = range[1]

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

        # if Configuration.getSetting("MinRangeFixed", field_name):
        #     min_magnitude = Configuration.getSetting("MinRange", field_name)
        #
        # if Configuration.getSetting("MaxRangeFixed", field_name):
        #     max_magnitude = Configuration.getSetting("MaxRange", field_name)


        glyphs = vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION >= 6:
            glyphs.SetInputData(vector_grid)
        else:
            glyphs.SetInput(vector_grid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        # glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # glyphs.SetScaleFactor(Configuration.getSetting("ArrowLength")) # scaling arrows here ArrowLength indicates scaling factor not actual length


        vector_field_actor = actors_dict['vector_field_actor']

        # scaling factor for an arrow - ArrowLength indicates scaling factor not actual length
        arrowScalingFactor = Configuration.getSetting("ArrowLength",field_name)

        if Configuration.getSetting("ScaleArrowsOn", field_name):
            glyphs.SetScaleModeToScaleByVector()
            rangeSpan = self.maxMagnitude - self.minMagnitude
            dataScalingFactor = max(abs(self.minMagnitude), abs(self.maxMagnitude))
            #            print MODULENAME,"self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor == 0.0:
                dataScalingFactor = 1.0  # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)
            # coloring arrows

            color = Configuration.getSetting("ArrowColor", field_name)
            r, g, b = color.red(), color.green(), color.blue()

            vector_field_actor.GetProperty().SetColor(r, g, b)
        else:
            glyphs.SetColorModeToColorByVector()
            glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphsMapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphsMapper.SetLookupTable(self.scalarLUT)

        self.glyphsMapper.SetScalarRange([min_magnitude, max_magnitude])

        vector_field_actor.SetMapper(self.glyphsMapper)

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


        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            outline_actor.SetScale(self.xScaleHex, self.yScaleHex, self.zScaleHex)

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
        scene_metadata = drawing_params.screenshot_data.metadata


        axesActor = actors_dict['axes_actor']
        axes_color = to_vtk_rgb(scene_metadata['AxesColor'])

        lattice_type_str = self.get_lattice_type_str()


        tprop = vtk.vtkTextProperty()
        tprop.SetColor(axes_color)
        tprop.ShadowOn()
        dim = self.currentDrawingParameters.bsd.fieldDim

        axesActor.SetNumberOfLabels(4) # number of labels

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal':
            axesActor.SetBounds(0, dim.x, 0, dim.y*math.sqrt(3.0)/2.0, 0, dim.z*math.sqrt(6.0)/3.0)
        else:
            axesActor.SetBounds(0, dim.x, 0, dim.y, 0, dim.z)

        axesActor.SetLabelFormat("%6.4g")
        axesActor.SetFlyModeToOuterEdges()
        axesActor.SetFontFactor(1.5)

        # axesActor.GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)
        axesActor.GetProperty().SetColor(axes_color)

        xAxisActor = axesActor.GetXAxisActor2D()
        # xAxisActor.RulerModeOn()
        # xAxisActor.SetRulerDistance(40)
        # xAxisActor.SetRulerMode(20)
        # xAxisActor.RulerModeOn()
        xAxisActor.SetNumberOfMinorTicks(3)




    def __zoomStep(self, delta):
        # # # print "ZOOM STEP"
        if self.ren:
            # renderer = self.GetCurrentRenderer()
            camera = self.ren.GetActiveCamera()
            
            zoomFactor = math.pow(1.02,(0.5*(delta/8)))

            # I don't know why I might need the parallel projection
            if camera.GetParallelProjection(): 
                parallelScale = camera.GetParallelScale()/zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                self.ren.ResetCameraClippingRange()

            self.Render()

    def takeSimShot(self, fileName):
        renderLarge = vtk.vtkRenderLargeImage()
        if VTK_MAJOR_VERSION>=6:
            renderLarge.SetInputData(self.graphicsFrameWidget.ren)
        else:    
            renderLarge.SetInput(self.graphicsFrameWidget.ren)
        

        renderLarge.SetMagnification(1)

        # We write out the image which causes the rendering to occur. If you
        # watch your screen you might see the pieces being rendered right
        # after one another.
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(renderLarge.GetOutputPort())        
        # # # print "GOT HERE fileName=",fileName
        writer.SetFileName(fileName)
        
        writer.Write()

    def initSizeDim(self, dataSet, x, y, z):
        (xloc, yloc, zloc) = (x, y, z)
        if x == 1:
            xloc += 2
        if y == 1:
            yloc += 2
        if z == 1:
            zloc += 2

        dataSet.SetDimensions(xloc, yloc, zloc)

    # this function is used during prototyping. in production code it is replaced by C++ counterpart    
    def fillCellFieldData_old(self,_cellFieldG):
        import CompuCell
        
        pt = CompuCell.Point3D() 
        cell = CompuCell.CellG() 
        fieldDim = _cellFieldG.getDim()

        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]
        # # # print "FILLCELLFIELDDATA 3D"
        # # # print "self.dim=",self.dim
        offset=0

        #will add 1 pixel border to celltype vtkImage data so that rendering will look smooth at the borders
        self.cellType = vtk.vtkIntArray()
        self.cellType.SetName("celltype")
        self.cellType.SetNumberOfValues((self.dim[2]+2)*(self.dim[1]+2)*(self.dim[0]+2))
        self.cell_id_array=[[[0 for k in range(self.dim[2])] for j in range(self.dim[1])] for i in range(self.dim[0])]
        
        usedCellTypes={}
        
        # For some reasons the points x=0 are eaten up (don't know why).
        # So we just populate empty cellIds.
        # for i in range(self.dim[0]+1):
            # self.cellType.SetValue(offset, 0)
            # offset += 1
                
        for k in range(self.dim[2]+2):
            for j in range(self.dim[1]+2):
                for i in range(self.dim[0]+2):                
                    if i==0 or i ==self.dim[0]+1 or j==0 or j ==self.dim[1]+1 or k==0 or k ==self.dim[2]+1:
                        self.cellType.InsertValue(offset, 0)
                        offset+=1
                    else:
                        pt.x = i-1
                        pt.y = j-1
                        pt.z = k-1
                        cell = _cellFieldG.get(pt)
                        if cell is not None:
                            type    = int(cell.type)
                            id      = int(cell.id)
                            if not type in usedCellTypes:
                                usedCellTypes[type]=0
                        else:
                            type    = 0
                            id      = 0
                        self.cellType.InsertValue(offset, type)
                        # print "inserting type ",type," offset ",offset
                        # print "pt=",pt," type=",type
                        
                        offset += 1
                        
                        self.cell_id_array[pt.x][pt.y][pt.z] = id

        usedCellTypesList=usedCellTypes.keys()
        usedCellTypesList.sort()
        return usedCellTypesList
        
    # this function is used during prototyping. in production code it is replaced by C++ counterpart            
    def fillConFieldData(self,_cellFieldG,_conField):
        import CompuCell
        
        pt = CompuCell.Point3D(0,0,0) 
        cell = CompuCell.CellG() 
        fieldDim = _cellFieldG.getDim()

        self.dim  = [fieldDim.x, fieldDim.y, fieldDim.z]         

        # # # print "FILL CONFIELDDATA 3D"
        # # # print "self.dim=",self.dim

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        zdim = self.dim[2]+2
        ydim = self.dim[1]+2
        xdim = self.dim[0]+2
        self.conArray.SetNumberOfValues(xdim*ydim*zdim)

        self.cellTypeCon = vtk.vtkIntArray()
        self.cellTypeCon.SetName("concelltype")
        self.cellTypeCon.SetNumberOfValues((self.dim[2]+2)*(self.dim[1]+2)*(self.dim[0]+2))
        
        offset=0        
        # # For some reasons the points x=0 are eaten up (don't know why).
        # # So we just populate empty cellIds.
        # for i in range(self.dim[0]+1):
            # self.conArray.SetValue(offset, 0.0)    
            # offset += 1
        
        
        maxCon = float(_conField.get(pt)) # concentration at pt=0,0,0
        minCon = float(_conField.get(pt)) # concentration at pt=0,0,0
        
        con=0.0
        for k in range(zdim):
            for j in range(ydim):
                for i in range(xdim):                
                    # if padding bogus boundary
#                    if i==0 or i ==self.dim[0]+1 or j==0 or j ==self.dim[1]+1 or k==0 or k ==self.dim[2]+1:
#                        con=0.0
#                        self.conArray.SetValue(offset, con)
#                        type=0
#                        self.cellTypeCon.SetValue(offset,type)
#                    else:
#                        pt.x = i-1
#                        pt.y = j-1
#                        pt.z = k-1
                        pt.x = i
                        pt.y = j
                        pt.z = k
                        
                        # con = float(_conField.get(pt))
                        # con = float((self.dim[1]-pt.y)*(self.dim[0]-pt.x))
                        con = float(pt.y*pt.x)
                        self.conArray.SetValue(offset, con)
                        
                        cell = _cellFieldG.get(pt)
                        
                        if cell is not None:
                            type = int(cell.type)
                            if type in self.invisibleCellTypes:
                                type = 0
                        else:
                            type = 0
                     
                        self.cellTypeCon.SetValue(offset,type)
                        
                        if maxCon < con:
                            maxCon = con
                        
                        if minCon > con:
                            minCon = con
                        
                        
                        offset += 1
#                    offset += 1
        
        return (minCon, maxCon)

    def initCellGlyphsActor3D(self, _glyphActor, _invisibleCellTypes):
#        print MODULENAME,'  ---initCellGlyphsActor3D'
#        print MODULENAME,'    _invisibleCellTypes=', _invisibleCellTypes

        from PySteppables import CellList

        fieldDim=self.currentDrawingParameters.bsd.fieldDim
        sim = self.currentDrawingParameters.bsd.sim
        if (sim == None):
          print 'MVCDrawModel3D.py: initCellGlyphsActor3D(),  sim is empty'
          return
      
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList=CellList(inventory)
        centroidPoints = vtk.vtkPoints()
        cellTypes = vtk.vtkIntArray()
        cellTypes.SetName("CellTypes")

#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")

#        if self.scaleGlyphsByVolume:
        cellScalars = vtk.vtkFloatArray()
        cellScalars.SetName("CellScalars")
        
        cellCount = 0
        
#        if self.hexFlag:
#          print MODULENAME,'   initCellGlyphsActor3D(): doing hex'
#          for cell in cellList:
#              if cell.type in _invisibleCellTypes: continue   # skip invisible cell types
#
#              #print 'cell.id=',cell.id  # = 2,3,4,...
#              #print 'cell.type=',cell.type
#              #print 'cell.volume=',cell.volume
#              xmid = cell.xCOM/1.122
#              ymid = cell.yCOM/1.122
##              zmid = cell.zCOM/1.07457
#              zmid = cell.zCOM
#    #          if cellCount < 50:  print cellCount,' glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
#    #          if cell.volume > 1: print cellCount,' ** glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
#    #          cellCount += 1
#              centroidPoints.InsertNextPoint(xmid,ymid,zmid)
#              cellTypes.InsertNextValue(cell.type)
#    
#    #          if self.scaleGlyphsByVolume:
#              if Configuration.getSetting("CellGlyphScaleByVolumeOn"):       # todo: make class attrib; update only when changes
#                cellScalars.InsertNextValue(cell.volume ** 0.333)   # take cube root of V, to get ~radius
#              else:
#                cellScalars.InsertNextValue(1.0)      # lame way of doing this
#        else:
#        print MODULENAME,'   initCellGlyphsActor3D(): self.offset=',self.offset
        for cell in cellList:
              if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

              #print 'cell.id=',cell.id  # = 2,3,4,...
              #print 'cell.type=',cell.type
              #print 'cell.volume=',cell.volume
              xmid = cell.xCOM     # + self.offset
              ymid = cell.yCOM
              zmid = cell.zCOM
    #          if cellCount < 50:  print cellCount,' glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
    #          if cell.volume > 1: print cellCount,' ** glyph x,y,z,vol=',xmid,ymid,zmid,cell.volume
    #          cellCount += 1
              centroidPoints.InsertNextPoint(xmid,ymid,zmid)
              cellTypes.InsertNextValue(cell.type)
    
    #          if self.scaleGlyphsByVolume:
              if Configuration.getSetting("CellGlyphScaleByVolumeOn"):       # todo: make class attrib; update only when changes
                cellScalars.InsertNextValue(cell.volume ** 0.333)   # take cube root of V, to get ~radius
              else:
                cellScalars.InsertNextValue(1.0)      # lame way of doing this


        centroidsPD = vtk.vtkPolyData()
        centroidsPD.SetPoints(centroidPoints)
        centroidsPD.GetPointData().SetScalars(cellTypes)

#        if self.scaleGlyphsByVolume:
        centroidsPD.GetPointData().AddArray(cellScalars)

        centroidGS = vtk.vtkSphereSource()
        thetaRes = Configuration.getSetting("CellGlyphThetaRes")     # todo: make class attrib; update only when changes
        phiRes = Configuration.getSetting("CellGlyphPhiRes")            
        centroidGS.SetThetaResolution(thetaRes)  # increase these values for a higher-res sphere glyph
        centroidGS.SetPhiResolution(phiRes)

        centroidGlyph = vtk.vtkGlyph3D()
        
        if VTK_MAJOR_VERSION>=6:
            centroidGlyph.SetInputData(centroidsPD)
        else:    
            centroidGlyph.SetInput(centroidsPD)

        try:
            centroidGlyph.SetSource(centroidGS.GetOutput())
        except AttributeError:
            centroidGlyph.SetSourceData(centroidGS.GetOutput())

        glyphScale = Configuration.getSetting("CellGlyphScale")            
        centroidGlyph.SetScaleFactor( glyphScale )
        centroidGlyph.SetIndexModeToScalar()
        centroidGlyph.SetRange(0,self.celltypeLUTMax)

        centroidGlyph.SetColorModeToColorByScalar()
#        if self.scaleGlyphsByVolume:
        centroidGlyph.SetScaleModeToScaleByScalar()
        
#        centroidGlyph.SetScaleModeToDataScalingOff()  # call this to disable scaling by scalar value
#        centroidGlyph.SetScaleModeToDataScalingOn()   # method doesn't even exist?!

        centroidGlyph.SetInputArrayToProcess(3,0,0,0,"CellTypes")
        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellScalars")

        if VTK_MAJOR_VERSION>=6:
            self.cellGlyphsMapper.SetInputData(centroidGlyph.GetOutput())
        else:    
            self.cellGlyphsMapper.SetInput(centroidGlyph.GetOutput())


        
        self.cellGlyphsMapper.SetScalarRange(0,self.celltypeLUTMax)
        self.cellGlyphsMapper.ScalarVisibilityOn()
        
        self.cellGlyphsMapper.SetLookupTable(self.celltypeLUT)   # defined in parent class
#        print MODULENAME,' usedCellTypesList=' ,self.usedCellTypesList

        _glyphActor.SetMapper(self.cellGlyphsMapper)  # Note: we don't need to scale actor for hex lattice here since using cell info

#---------------------------------------------------------------------------
    def initFPPLinksActor3D(self, _fppActor, _invisibleCellTypes):
#        print MODULENAME,'  initFPPLinksActor3D'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor3D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):  # bogus check
          print MODULENAME,'    fppPlugin is null, returning'
          return

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print 'fieldDim, fieldDim.x =',fieldDim,fieldDim.x
        xdim = fieldDim.x
        ydim = fieldDim.y
        zdim = fieldDim.z

        # To test if links should be stubs (for wraparound on periodic BCs)
        xdim_delta = xdim/2
        ydim_delta = ydim/2
        zdim_delta = zdim/2

        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList = CellList(inventory)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0

        for cell in cellList:
          if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue
          xmid0 = cell.xCOM    # + self.offset
          ymid0 = cell.yCOM
          zmid0 = cell.zCOM
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,zmid0)

          endPt = beginPt + 1

#2345678901234
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):  # First pass (Internal list)
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM    # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
#                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
                numStubs = 0
                if abs(xdiff) > xdim_delta:   # wraps around in x-direction
                    numStubs += 1
#                    print '>>>>>> wraparound X'
                    ymid0end = ymid0
                    zmid0end = zmid0
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1
#                else:   # wraps around in y-direction
                if abs(ydiff) > ydim_delta:   # wraps around in y-direction
                    numStubs += 1
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    zmid0end = zmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                if abs(zdiff) > zdim_delta:   # wraps around in z-direction
                    numStubs += 1
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    ymid0end = ymid0
                    if zdiff < 0:
                      zmid0end = zmid0 + self.stubSize
                    else:
                      zmid0end = zmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = zdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)

                if numStubs > 1: print MODULENAME,"  --------------  numStubs = ",numStubs

            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' (internal link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#         ---------------------------------------
#2345678901234
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):   # Second pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   #  + self.offset   # used to do: float(fppd.neighborAddress.xCM) / vol + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
#                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
                numStubs = 0
                if abs(xdiff) > xdim_delta:   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    numStubs += 1
                    ymid0end = ymid0
                    zmid0end = zmid0
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                if abs(ydiff) > ydim_delta:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    numStubs += 1
                    xmid0end = xmid0
                    zmid0end = zmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                if abs(zdiff) > zdim_delta:   # wraps around in z-direction
#                    print '>>>>>> wraparound Z'
                    numStubs += 1
                    xmid0end = xmid0
                    ymid0end = ymid0
                    if zdiff < 0:
                      zmid0end = zmid0 + self.stubSize
                    else:
                      zmid0end = zmid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' ----- (external link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#2345678901234
#          print 'after external links: beginPt, endPt=',beginPt,endPt
          beginPt = endPt  # update point index

        #-----------------------
        if lineNum == 0:  return
#        print '---------- # links=',lineNum

        # create Blue-Red LUT
#        lutBlueRed = vtk.vtkLookupTable()
#        lutBlueRed.SetHueRange(0.667,0.0)
#        lutBlueRed.Build()

#        print '---------- # links,scalarValMin,Max =',lineNum,scalarValMin,scalarValMax
        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)



        if VTK_MAJOR_VERSION>=6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:
            FPPLinksPD.Update()
            self.FPPLinksMapper.SetInput(FPPLinksPD)




#        self.FPPLinksMapper.SetScalarModeToUseCellFieldData()

        _fppActor.SetMapper(self.FPPLinksMapper)  # Note: we don't need to scale actor for hex lattice here since using cell info

        
#---------------------------------------------------------------------------
    def initFPPLinksColorActor3D(self, _fppActor, _invisibleCellTypes):
#        print MODULENAME,'  initFPPLinksActor3D_color'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell
        
        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor3D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):  # bogus check
          print '    fppPlugin is null, returning'
          return

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print 'fieldDim, fieldDim.x =',fieldDim,fieldDim.x
        xdim = fieldDim.x
        ydim = fieldDim.y
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList = CellList(inventory)
        
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colorScalars = vtk.vtkFloatArray()
        colorScalars.SetName("fpp_scalar")

#        cellTypes = vtk.vtkIntArray()
#        cellTypes.SetName("CellTypes")
#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0
        scalarValMin = 1000.0
        scalarValMax = -scalarValMin

        for cell in cellList:
          if cell.type in _invisibleCellTypes: continue   # skip invisible cell types

#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue
          xmid0 = cell.xCOM    # + self.offset
          ymid0 = cell.yCOM
          zmid0 = cell.zCOM
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,zmid0)
          
          endPt = beginPt + 1
          
#2345678901
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):  # First pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
#            d2 = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff)  # compute dist^2 and avoid sqrt
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))  
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                zmid0end = zmid0 
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                scalarVal = d2/targetDist2    # actual^2/target^2
#                scalarVal = actualDist / fppd.targetDistance    # actual/target
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist 
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)
                
                lineNum += 1
                endPt += 1

        
#         ---------------------------------------
#2345678901
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):   # Second pass
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:  
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid=fppd.neighborAddress.xCOM   # + self.offset
            ymid=fppd.neighborAddress.yCOM
            zmid=fppd.neighborAddress.zCOM
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            zdiff = zmid-zmid0
#            d2 = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff)  # compute dist^2 and avoid sqrt
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff)+(zdiff*zdiff))
#            if beginPt < 10:  
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                zmid0end = zmid0 
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,zmid0end)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                    scalarVal = d2/targetDist2    # actual^2/target^2
#                    scalarVal = actualDist / fppd.targetDistance    # actual/target
                    scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                    scalarVal = actualDist 
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
                    colorScalars.InsertNextValue(scalarVal)
                    
                    lineNum += 1
                    endPt += 1

                    # add dangling "in" line to end cell
#                    beginPt = endPt 
#                    lines.InsertNextCell(2)  # our line has 2 points
#                    lines.InsertCellPoint(beginPt)
#                    lines.InsertCellPoint(endPt)


            else:   # link didn't wrap around on lattice
#                print '>>> No wraparound'
                points.InsertNextPoint(xmid,ymid,zmid)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
#                scalarVal = d2/targetDist2    # actual^2/target^2
#                scalarVal = actualDist / fppd.targetDistance    # actual/target
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist 
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)
                
                lineNum += 1
                endPt += 1

          beginPt = endPt  # update point index 
          
          #--------------------------------------

        if lineNum == 0:  return
        
        # create Blue-Red LUT
#        lutBlueRed = vtk.vtkLookupTable()
#        lutBlueRed.SetHueRange(0.667,0.0)
#        lutBlueRed.Build()

#        print '---------- # links,scalarValMin,Max =',lineNum,scalarValMin,scalarValMax
        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)

        
        
        if VTK_MAJOR_VERSION>=6:
            pass
        else:    
            FPPLinksPD.Update()
        
        FPPLinksPD.GetCellData().SetScalars(colorScalars)
        
        if VTK_MAJOR_VERSION>=6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:    
            self.FPPLinksMapper.SetInput(FPPLinksPD)
        
        
        

        self.FPPLinksMapper.SetScalarModeToUseCellFieldData()
        self.FPPLinksMapper.SelectColorArray("fpp_scalar")
        self.FPPLinksMapper.SetScalarRange(scalarValMin,scalarValMax)

        self.FPPLinksMapper.SetLookupTable(self.lutBlueRed)
        
        _fppActor.SetMapper(self.FPPLinksMapper)

        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(self.lutBlueRed)
        #scalarBar.SetTitle("Stress")
        scalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        #scalarBar.GetPositionCoordinate().SetValue(0.8,0.05)
        scalarBar.SetOrientationToVertical()
        scalarBar.SetWidth(0.1)
        scalarBar.SetHeight(0.9)
        scalarBar.SetPosition(0.88,0.1)
        #scalarBar.SetLabelFormat("%-#6.3f")
        scalarBar.SetLabelFormat("%-#3.1f")
        scalarBar.GetLabelTextProperty().SetColor(1,1,1)
        #scalarBar.GetTitleTextProperty().SetColor(1,0,0)

#        self.graphicsFrameWidget.ren.AddActor2D(scalarBar)    

    def configsChanged(self):
        self.populateLookupTable()
        #reassign which types are invisible        
        self.set3DInvisibleTypes()
        self.parentWidget.requestRedraw()
