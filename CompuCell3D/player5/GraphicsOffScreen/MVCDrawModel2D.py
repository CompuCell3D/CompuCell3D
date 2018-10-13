# -*- coding: utf-8 -*-
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# import
# from GraphicsNew import GraphicsNew
import Configuration
from MVCDrawModelBase import MVCDrawModelBase
import vtk
import math
import string
from CompuCell3D.player5.Utilities.utils import extract_address_int_from_vtk_object, to_vtk_rgb

VTK_MAJOR_VERSION=vtk.vtkVersion.GetVTKMajorVersion()


MODULENAME='----- MVCDrawModel2D.py:  '
from Messaging import dbgMsg


class MVCDrawModel2D(MVCDrawModelBase):
    def __init__(self):
        MVCDrawModelBase.__init__(self)
        
        self.initArea()
        self.setParams()

        
    # Sets up the VTK simulation area 
    def initArea(self):
        ## Set up the mappers (2D) for cell vis.
        self.cellsMapper    = vtk.vtkPolyDataMapper()
        self.hex_cells_mapper = vtk.vtkPolyDataMapper()
        self.cartesianCellsMapper = vtk.vtkPolyDataMapper()
        self.borderMapper   = vtk.vtkPolyDataMapper()
        self.borderMapperHex   = vtk.vtkPolyDataMapper()
        self.clusterBorderMapper   = vtk.vtkPolyDataMapper()
        self.clusterBorderMapperHex   = vtk.vtkPolyDataMapper()
        self.cellGlyphsMapper  = vtk.vtkPolyDataMapper()
        self.FPPLinksMapper  = vtk.vtkPolyDataMapper()

        self.outlineDim=[0,0,0]

        ## Set up the mappers (2D) for concentration field.
        self.conMapper      = vtk.vtkPolyDataMapper()
        self.hex_con_mapper   = vtk.vtkPolyDataMapper()
        self.cartesianConMapper   = vtk.vtkPolyDataMapper()
        self.contour_mapper  = vtk.vtkPolyDataMapper()
        self.glyphs_mapper   = vtk.vtkPolyDataMapper()

        # # Concentration lookup table
        self.numberOfTableColors=1024
        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0,1.0)
        self.clut.SetValueRange(1.0,1.0)
        self.clut.SetAlphaRange(1.0,1.0)
        self.clut.SetNumberOfColors(self.numberOfTableColors)
        self.clut.Build()

        self.lowTableValue=self.clut.GetTableValue(0)
        self.highTableValue=self.clut.GetTableValue(self.numberOfTableColors-1)

        # Contour lookup table
        # Do I need lookup table? May be just one color?
        self.ctlut = vtk.vtkLookupTable()
        self.ctlut.SetHueRange(0.6, 0.6)
        self.ctlut.SetSaturationRange(0,1.0)
        self.ctlut.SetValueRange(1.0,1.0)
        self.ctlut.SetAlphaRange(1.0,1.0)
        self.ctlut.SetNumberOfColors(self.numberOfTableColors)
        self.ctlut.Build()

    def setDim(self, fieldDim):
        self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]

    def prepareOutlineActors(self,_actors):
        outlineData = vtk.vtkImageData()

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))

        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.currentDrawingParameters.plane=="XY":
            import math
            outlineData.SetDimensions(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
            # print "self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
        else:
            outlineData.SetDimensions(self.dim[0]+1, self.dim[1]+1, 1)

        # outlineDimTmp=_imageData.GetDimensions()
        # # print "\n\n\n this is outlineDimTmp=",outlineDimTmp," self.outlineDim=",self.outlineDim
        # if self.outlineDim[0] != outlineDimTmp[0] or self.outlineDim[1] != outlineDimTmp[1] or self.outlineDim[2] != outlineDimTmp[2]:
            # self.outlineDim=outlineDimTmp

        outline = vtk.vtkOutlineFilter()

        if VTK_MAJOR_VERSION>=6:
            outline.SetInputData(outlineData)
        else:
            outline.SetInput(outlineData)


        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        _actors[0].SetMapper(outlineMapper)
        _actors[0].GetProperty().SetColor(1, 1, 1)
        # self.outlineDim=_imageData.GetDimensions()

        color = Configuration.getSetting("BoundingBoxColor")   # eventually do this smarter (only get/update when it changes)
        _actors[0].GetProperty().SetColor(float(color.red())/255,float(color.green())/255,float(color.blue())/255)


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
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":

            outlineData.SetDimensions(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
            # print "self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1= ",(self.dim[0]+1,int(self.dim[1]*math.sqrt(3.0)/2.0)+2,1)
        else:
            outlineData.SetDimensions(self.dim[0]+1, self.dim[1]+1, 1)

        outline = vtk.vtkOutlineFilter()

        if VTK_MAJOR_VERSION>=6:
            outline.SetInputData(outlineData)
        else:
            outline.SetInput(outlineData)

        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())

        outline_actor = actors_dict['outlineActor']
        outline_actor.SetMapper(outlineMapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)

        outline_color = to_vtk_rgb(scene_metadata['BoundingBoxColor'])

        outline_actor.GetProperty().SetColor(*outline_color)


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
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        field_type = drawing_params.fieldType.lower()
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

        if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
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

        minMagnitude = range[0]
        maxMagnitude = range[1]

        if Configuration.getSetting("MinRangeFixed", field_name):
            minMagnitude = Configuration.getSetting("MinRange", field_name)

        if Configuration.getSetting("MaxRangeFixed", field_name):
            maxMagnitude = Configuration.getSetting("MaxRange", field_name)

        glyphs = vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION >= 6:
            glyphs.SetInputData(vector_grid)
        else:
            glyphs.SetInput(vector_grid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        # glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # rwh: should use of this factor depend on the state of the "Scale arrow length" checkbox?
        arrowScalingFactor = Configuration.getSetting("ArrowLength",
                                                      field_name)  # scaling factor for an arrow (ArrowLength indicates scaling factor not actual length)

        vector_field_actor = actors_dict['vector_field_actor']
        if Configuration.getSetting("FixedArrowColorOn", field_name):
            glyphs.SetScaleModeToScaleByVector()
            # rangeSpan = maxMagnitude - minMagnitude
            dataScalingFactor = max(abs(minMagnitude), abs(maxMagnitude))
            #            print MODULENAME,"initVectorFieldCellLevelActors():  self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor == 0.0:
                dataScalingFactor = 1.0  # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)
            # coloring arrows
            color = Configuration.getSetting("ArrowColor", field_name)
            r, g, b = color.red(), color.green(), color.blue()
            vector_field_actor.GetProperty().SetColor(r, g, b)
        else:
            if Configuration.getSetting("ScaleArrowsOn", field_name):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                rangeSpan = maxMagnitude - minMagnitude
                dataScalingFactor = max(abs(minMagnitude), abs(maxMagnitude))
                #                print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

                if dataScalingFactor == 0.0:
                    dataScalingFactor = 1.0  # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                glyphs.SetScaleFactor(arrowScalingFactor / dataScalingFactor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)

        self.glyphs_mapper.SetScalarRange([minMagnitude, maxMagnitude])

        vector_field_actor.SetMapper(self.glyphs_mapper)

    def init_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() =='hexagonal' and drawing_params.plane.lower()=="xy":
            self.init_concentration_field_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            self.init_concentration_field_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)

    def init_concentration_field_actors_hex(self, actor_specs, drawing_params=None):
        """
        initializes concentration field actors for hex lattice
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """
        print "init_concentration_field_actors_hex"
        actors_dict = actor_specs.actors_dict

        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=con_array)
        hex_points_con = vtk.vtkPoints()
        hex_points_con_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=hex_points_con)


        hex_cells_con = vtk.vtkCellArray()
        hex_cells_con_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=hex_cells_con)
        hex_cells_con_poly_data = vtk.vtkPolyData()


        field_type = drawing_params.fieldType.lower()
        if field_type =='confield':
            fill_successful = self.field_extractor.fillConFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type =='scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type =='scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2DHex(
                con_array_int_addr,
                hex_cells_con_int_addr,
                hex_points_con_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        else:
            print ("unsuported field type {}".format(field_type))
            return


        # fill_successful = self.field_extractor.fillConFieldData2DHex(
        #     con_array_int_addr,
        #     hex_cells_con_int_addr,
        #     hex_points_con_int_addr,
        #     field_name,
        #     self.currentDrawingParameters.plane,
        #     self.currentDrawingParameters.planePos
        # )

        if not fill_successful:
            return

        if set(['MinRangeFixed',"MaxRangeFixed",'MinRange','MaxRange']).issubset( set(scene_metadata.keys())):
            min_range_fixed = scene_metadata['MinRangeFixed']
            max_range_fixed = scene_metadata['MaxRangeFixed']
            min_range = scene_metadata['MinRange']
            max_range = scene_metadata['MaxRange']
        else:
            min_range_fixed = Configuration.getSetting("MinRangeFixed", field_name)
            max_range_fixed = Configuration.getSetting("MaxRangeFixed", field_name)
            min_range = Configuration.getSetting("MinRange", field_name)
            max_range = Configuration.getSetting("MaxRange", field_name)


        range =con_array.GetRange()
        min_con = range[0]
        max_con = range[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering; only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        if scene_metadata['ContoursOn']:
            contour_actor = actors_dict['contour_actor']
            num_contour_lines = scene_metadata['NumberOfContourLines']
            self.initialize_contours_hex([dim[0], dim[1]], con_array, [min_con, max_con],
                                       contour_actor,num_contour_lines=num_contour_lines)

        hex_cells_con_poly_data.GetCellData().SetScalars(con_array)
        hex_cells_con_poly_data.SetPoints(hex_points_con)
        hex_cells_con_poly_data.SetPolys(hex_cells_con)

        if VTK_MAJOR_VERSION >= 6:
            self.hex_con_mapper.SetInputData(hex_cells_con_poly_data)
        else:
            self.hex_con_mapper.SetInput(hex_cells_con_poly_data)

        self.hex_con_mapper.ScalarVisibilityOn()
        self.hex_con_mapper.SetLookupTable(self.clut)
        self.hex_con_mapper.SetScalarRange(min_con, max_con)

        concentration_actor = actors_dict['concentration_actor']

        concentration_actor.SetMapper(self.hex_con_mapper)

    def initialize_contours_hex(self, dim, con_array, min_max, contour_actor, num_contour_lines=2):
        """
        INitializes contour actor
        :param dim: {tuple}
        :param con_array: {vtkDoubleArray}
        :param min_max: {tuple (float, float)} concentration min, max
        :param contour_actor: {vtkActor} conrour actor
        :param num_contour_lines: {int} number of contour lines
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

        transform.Scale(1, math.sqrt(3.0) / 2.0, 1)
        if self.currentDrawingParameters.planePos % 3 == 0:
            transform.Translate(0.5, 0, 0)  # z%3==0
        elif self.currentDrawingParameters.planePos % 3 == 1:
            transform.Translate(0, math.sqrt(3.0) / 4.0, 0)  # z%3==1
        else:
            transform.Translate(0.0, -math.sqrt(3.0) / 4.0, 0)  # z%3==2

        isoContour = vtk.vtkContourFilter()

        isoContour.SetInputConnection(field_image_data.GetOutputPort())

        isoContour.GenerateValues(
            num_contour_lines+2,
            min_max)

        tpd1 = vtk.vtkTransformPolyDataFilter()
        tpd1.SetInputConnection(isoContour.GetOutputPort())
        tpd1.SetTransform(transform)

        # self.contourMapper.SetInputConnection(contour.GetOutputPort())
        self.contour_mapper.SetInputConnection(tpd1.GetOutputPort())
        self.contour_mapper.SetLookupTable(self.ctlut)
        self.contour_mapper.SetScalarRange(min_max)
        self.contour_mapper.ScalarVisibilityOff()
        contour_actor.SetMapper(self.contour_mapper)

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
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
        field_name = drawing_params.fieldName
        scene_metadata = drawing_params.screenshot_data.metadata

        con_array = vtk.vtkDoubleArray()
        con_array.SetName("concentration")
        con_array_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=con_array)
        # todo - make it flexible

        field_type = drawing_params.fieldType.lower()
        if field_type =='confield':
            fill_successful = self.field_extractor.fillConFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type =='scalarfield':
            fill_successful = self.field_extractor.fillScalarFieldData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )
        elif field_type =='scalarfieldcelllevel':
            fill_successful = self.field_extractor.fillScalarFieldCellLevelData2D(
                con_array_int_addr,
                field_name,
                self.currentDrawingParameters.plane,
                self.currentDrawingParameters.planePos
            )

        else:
            print ("unsuported field type {}".format(field_type))
            return

        if not fill_successful:
            return

        # # todo 5 - revisit later
        # numIsos = Configuration.getSetting("NumberOfContourLines", field_name)
        # #        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)

        if set(['MinRangeFixed',"MaxRangeFixed",'MinRange','MaxRange']).issubset( set(scene_metadata.keys())):
            min_range_fixed = scene_metadata['MinRangeFixed']
            max_range_fixed = scene_metadata['MaxRangeFixed']
            min_range = scene_metadata['MinRange']
            max_range = scene_metadata['MaxRange']
        else:
            min_range_fixed = Configuration.getSetting("MinRangeFixed", field_name)
            max_range_fixed = Configuration.getSetting("MaxRangeFixed", field_name)
            min_range = Configuration.getSetting("MinRange", field_name)
            max_range = Configuration.getSetting("MaxRange", field_name)


        range =con_array.GetRange()
        min_con = range[0]
        max_con = range[1]

        # Note! should really avoid doing a getSetting with each step to speed up the rendering;
        # only update when changed in Prefs
        if min_range_fixed:
            min_con = min_range

        if max_range_fixed:
            max_con = max_range

        dim_0 = dim[0] + 1
        dim_1 = dim[1] + 1

        # print 'dim_0,dim_1=',(dim_0,dim_1)
        dbgMsg('dim_0,dim_1=', (dim_0, dim_1))

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

        if scene_metadata['ContoursOn']:

            contour_actor = actors_dict['contour_actor']
            num_contour_lines = scene_metadata['NumberOfContourLines']
            self.initialize_contours_cartesian(
                field_image_data,
                [min_con, max_con],
                contour_actor,
                num_contour_lines=num_contour_lines
            )

        self.conMapper.SetInputConnection(field_image_data.GetOutputPort())  # port index = 0

        self.conMapper.ScalarVisibilityOn()
        self.conMapper.SetLookupTable(self.clut)
        # 0, self.clut.GetNumberOfColors()) # may manually set range so that type reassignment will not be scalled dynamically when one type is missing
        self.conMapper.SetScalarRange(min_con,max_con)

        self.conMapper.SetScalarModeToUsePointData()

        concentration_actor = actors_dict['concentration_actor']
        concentration_actor.SetMapper(self.conMapper)  # concentration actor

    def initialize_contours_cartesian(self,field_image_data, min_max, contour_actor, num_contour_lines=2):

        min_con, max_con = min_max[0], min_max[1]
        iso_contour = vtk.vtkContourFilter()
        iso_contour.SetInputConnection(field_image_data.GetOutputPort())


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
        for idx in xrange(num_contour_lines):
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

        color = Configuration.getSetting("ContourColor")  # want to avoid this; only update when Prefs changes
        contour_actor.GetProperty().SetColor(float(color.red()) / 255, float(color.green()) / 255,
                                            float(color.blue()) / 255)

    def init_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() == 'hexagonal' and drawing_params.plane.lower() == "xy":
            self.init_cell_field_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
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

        # dim = self.planeMapper(dim_order,
        #                             (field_dim.x, field_dim.y, field_dim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")
        cell_type_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                                 vtkObj=cell_type_array)

        hex_cells_array = vtk.vtkCellArray()

        hex_cells_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                                 vtkObj=hex_cells_array)

        hex_cells_poly_data = vtk.vtkPolyData()
        # **********************************************

        hex_points = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        hex_points_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor, vtkObj=hex_points)

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

        self.hex_cells_mapper.ScalarVisibilityOn()
        self.hex_cells_mapper.SetLookupTable(self.celltypeLUT)
        self.hex_cells_mapper.SetScalarRange(0, self.celltypeLUTMax)

        cells_actor = actors_dict['cellsActor']
        cells_actor.SetMapper(self.hex_cells_mapper)

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

        # [fieldDim.x, fieldDim.y, fieldDim.z]
        dim = self.planeMapper(dim_order, (field_dim.x, field_dim.y, field_dim.z))

        cell_type_array = vtk.vtkIntArray()
        cell_type_array.SetName("celltype")

        cell_type_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                                 vtkObj=cell_type_array)

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

        # concMapper=self.cellsMapper

        self.cellsMapper.SetInputConnection(cells_plane.GetOutputPort())
        self.cellsMapper.ScalarVisibilityOn()

        self.cellsMapper.SetLookupTable(self.celltypeLUT)  # def'd in parent class
        self.cellsMapper.SetScalarRange(0, self.celltypeLUTMax)

        cells_actor = actors_dict['cellsActor']
        cells_actor.SetMapper(self.cellsMapper)

    def init_borders_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() =='hexagonal' and drawing_params.plane.lower()=="xy":
            self.init_borders_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            self.init_borders_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)


    def init_borders_actors_hex(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(
            field_extractor=self.field_extractor,
            vtkObj=points
        )

        lines_int_addr = extract_address_int_from_vtk_object(
            field_extractor=self.field_extractor,
            vtkObj=lines
        )

        self.field_extractor.fillBorderData2DHex(
            points_int_addr,
            lines_int_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos
        )

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        border_actor = actor_specs.actors_dict['borderActor']

        if VTK_MAJOR_VERSION >= 6:
            self.borderMapperHex.SetInputData(borders)
        else:
            self.borderMapperHex.SetInput(borders)

        border_actor.SetMapper(self.borderMapperHex)

        border_color = to_vtk_rgb(scene_metadata['BorderColor'])
        # coloring borders
        border_actor.GetProperty().SetColor(*border_color)

    def init_borders_actors_cartesian(self, actor_specs, drawing_params=None):
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

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                              vtkObj=points)

        lines_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                             vtkObj=lines)

        self.field_extractor.fillBorderData2D(points_int_addr, lines_int_addr, self.currentDrawingParameters.plane,
                                              self.currentDrawingParameters.planePos)

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.borderMapper.SetInputData(borders)
        else:
            self.borderMapper.SetInput(borders)

        border_actor = actor_specs.actors_dict['borderActor']
        # actors = list(actor_specs.actors_dict.values())

        border_actor.SetMapper(self.borderMapper)

        border_color = to_vtk_rgb(scene_metadata['BorderColor'])
        # coloring borders
        border_actor.GetProperty().SetColor(*border_color)


        # self.setBorderColor()

        # print "self.currentActors.keys()=",self.currentActors.keys()

        # # print "self.currentActors[BorderActor]=",self.currentActors["BorderActor"].GetClassName()
        # if not self.currentActors.has_key("BorderActor"):
        # self.currentActors["BorderActor"]=self.borderActor
        # self.graphicsFrameWidget.ren.AddActor(self.borderActor)
        # print "ADDING BORDER ACTOR"

        # else:
        # # will ensure that borders is the last item to draw
        # actorsCollection=self.graphicsFrameWidget.ren.GetActors()
        # if actorsCollection.GetLastItem()!=self.borderActor:
        # self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
        # self.graphicsFrameWidget.ren.AddActor(self.borderActor)
        # print "self.currentActors.keys()=",self.currentActors.keys()

    def init_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for cartesian actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        lattice_type_str = self.get_lattice_type_str()
        if lattice_type_str.lower() =='hexagonal' and drawing_params.plane.lower()=="xy":
            self.init_cluster_border_actors_hex(actor_specs=actor_specs, drawing_params=drawing_params)
        else:
            self.init_cluster_border_actors_cartesian(actor_specs=actor_specs, drawing_params=drawing_params)


    def init_cluster_border_actors_cartesian(self, actor_specs, drawing_params=None):
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


        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                             vtkObj=points)
        lines_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                             vtkObj=lines)
        self.field_extractor.fillClusterBorderData2D(
            points_int_addr,
            lines_int_addr,
            self.currentDrawingParameters.plane,
            self.currentDrawingParameters.planePos
        )


        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION>=6:
            self.clusterBorderMapper.SetInputData(borders)
        else:
            self.clusterBorderMapper.SetInput(borders)

        cluster_border_actor = actor_specs.actors_dict['cluster_border_actor']

        cluster_border_actor.SetMapper(self.clusterBorderMapper)

        cluster_border_color = to_vtk_rgb(scene_metadata['ClusterBorderColor'])
        # coloring borders
        cluster_border_actor.GetProperty().SetColor(*cluster_border_color)

    def init_cluster_border_actors_hex(self, actor_specs, drawing_params=None):
        """
        Initializes cell borders actors for hex actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :return: None
        """

        actors_dict = actor_specs.actors_dict
        field_dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        scene_metadata = drawing_params.screenshot_data.metadata

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        points_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                              vtkObj=points)
        lines_int_addr = extract_address_int_from_vtk_object(field_extractor=self.field_extractor,
                                                             vtkObj=lines)

        self.field_extractor.fillClusterBorderData2DHex(
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

        cluster_border_color = to_vtk_rgb(scene_metadata['ClusterBorderColor'])
        # coloring borders
        cluster_border_actor.GetProperty().SetColor(*cluster_border_color)




    # def initBordersActors2D(self,_actors):
    #     points = vtk.vtkPoints()
    #     lines = vtk.vtkCellArray()
    #     pointsIntAddr = extractAddressIntFromVtkObject(field_extractor=self.field_extractor,
    #                                                           vtkObj=points)
    #
    #     linesIntAddr = extractAddressIntFromVtkObject(field_extractor=self.field_extractor,
    #                                                           vtkObj=lines)
    #
    #     # pointsIntAddr=self.extractAddressIntFromVtkObject(points)
    #     # linesIntAddr=self.extractAddressIntFromVtkObject(lines)
    #
    #     # self.parentWidget.fieldExtractor.fillBorderData2D(pointsIntAddr , linesIntAddr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
    #
    #     self.field_extractor.fillBorderData2D(pointsIntAddr , linesIntAddr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
    #
    #
    #     borders = vtk.vtkPolyData()
    #
    #     borders.SetPoints(points)
    #     borders.SetLines(lines)
    #
    #     if VTK_MAJOR_VERSION>=6:
    #         self.borderMapper.SetInputData(borders)
    #     else:
    #         self.borderMapper.SetInput(borders)
    #
    #
    #     _actors[0].SetMapper(self.borderMapper)
    #     # self.setBorderColor()
    #
    #     # print "self.currentActors.keys()=",self.currentActors.keys()
    #
    #     # # print "self.currentActors[BorderActor]=",self.currentActors["BorderActor"].GetClassName()
    #     # if not self.currentActors.has_key("BorderActor"):
    #         # self.currentActors["BorderActor"]=self.borderActor
    #         # self.graphicsFrameWidget.ren.AddActor(self.borderActor)
    #         # print "ADDING BORDER ACTOR"
    #
    #     # else:
    #         # # will ensure that borders is the last item to draw
    #         # actorsCollection=self.graphicsFrameWidget.ren.GetActors()
    #         # if actorsCollection.GetLastItem()!=self.borderActor:
    #             # self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
    #             # self.graphicsFrameWidget.ren.AddActor(self.borderActor)
    #     # print "self.currentActors.keys()=",self.currentActors.keys()

    def initBordersActors2DHex(self, _actors):

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointsIntAddr = self.extractAddressIntFromVtkObject(points)
        linesIntAddr = self.extractAddressIntFromVtkObject(lines)

        self.parentWidget.fieldExtractor.fillBorderData2DHex(pointsIntAddr, linesIntAddr,
                                                             self.currentDrawingParameters.plane,
                                                             self.currentDrawingParameters.planePos)

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION >= 6:
            self.borderMapperHex.SetInputData(borders)
        else:
            self.borderMapperHex.SetInput(borders)

        _actors[0].SetMapper(self.borderMapperHex)
        # self.setBorderColor()
        # if not self.currentActors.has_key("BorderActor"):
        # self.currentActors["BorderActor"]=self.borderActor
        # self.graphicsFrameWidget.ren.AddActor(self.borderActor)
        # else:
        # # will ensure that borders is the last item to draw
        # actorsCollection=self.graphicsFrameWidget.ren.GetActors()
        # if actorsCollection.GetLastItem()!=self.borderActor:
        # self.graphicsFrameWidget.ren.RemoveActor(self.borderActor)
        # self.graphicsFrameWidget.ren.AddActor(self.borderActor)

    # self.initCellFieldActorsData(list(actor_specs.actors_dict.values()))

    # def init_cell_field_actors(self, actors):
    #
    #     fieldDim = self.currentDrawingParameters.bsd.fieldDim
    #     dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
    #     self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
    #
    #     self.cellType = vtk.vtkIntArray()
    #     self.cellType.SetName("celltype")
    #
    #     self.cellTypeIntAddr = extractAddressIntFromVtkObject(field_extractor=self.field_extractor, vtkObj=self.cellType)
    #     self.field_extractor.fillCellFieldData2D(self.cellTypeIntAddr,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
    #
    #     self.initCellFieldActorsData(actors)

    def initCellFieldHexActors(self, _actors):
        # cellField  = sim.getPotts().getCellFieldG()
        # # # # print "INSIDE drawCellFieldHex"
        # # # # print "drawing plane ",self.plane," planePos=",self.planePos
        # fieldDim = cellField.getDim()
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]

        self.cellType = vtk.vtkIntArray()
        self.cellType.SetName("celltype")
        self.cellTypeIntAddr = self.extractAddressIntFromVtkObject(self.cellType)
		# a=21
        self.hexCells = vtk.vtkCellArray()

        self.hexCellsIntAddr = self.extractAddressIntFromVtkObject(self.hexCells)

        self.hexCellsPolyData = vtk.vtkPolyData()
		# **********************************************

        self.hexPoints = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        self.hexPointsIntAddr = self.extractAddressIntFromVtkObject(self.hexPoints)

        self.parentWidget.fieldExtractor.fillCellFieldData2DHex(self.cellTypeIntAddr,self.hexCellsIntAddr,self.hexPointsIntAddr,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
		# self.parentWidget.fieldExtractor.fillCellFieldData2DHex(self.cellTypeIntAddr,self.hexPointsIntAddr,self.plane, self.planePos)

        self.hexCellsPolyData.GetCellData().SetScalars(self.cellType)
        self.hexCellsPolyData.SetPoints(self.hexPoints)
        self.hexCellsPolyData.SetPolys(self.hexCells)

        if VTK_MAJOR_VERSION>=6:
            self.hex_cells_mapper.SetInputData(self.hexCellsPolyData)
        else:
            self.hex_cells_mapper.SetInput(self.hexCellsPolyData)


        self.hex_cells_mapper.ScalarVisibilityOn()
        self.hex_cells_mapper.SetLookupTable(self.celltypeLUT)
        self.hex_cells_mapper.SetScalarRange(0, self.celltypeLUTMax)

        _actors[0].SetMapper(self.hex_cells_mapper)


    def prepareAxesActors(self, _mappers, _actors):
        axesActor = _actors[0]

        dim = self.currentDrawingParameters.bsd.fieldDim
        dim_order = self.dimOrder(self.currentDrawingParameters.plane)
        dim_array=[dim.x, dim.y, dim.z]
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"]:
            dim_array=[dim.x, dim.y*math.sqrt(3.0)/2.0, dim.z*math.sqrt(6.0)/3.0]

        axes_labels = ['X','Y','Z']

        horizontal_length = dim_array [dim_order[0]] # x-axis - equivalent
        vertical_length = dim_array [dim_order[1]] # y-axis - equivalent
        horizontal_label = axes_labels [dim_order[0]] # x-axis - equivalent
        vertical_label = axes_labels [dim_order[1]] # y-axis - equivalent

        color = Configuration.getSetting("AxesColor")   # eventually do this smarter (only get/update when it changes)

        color = (float(color.red())/255, float(color.green())/255, float(color.blue())/255)

        axesActor.GetProperty().SetColor(color)

        tprop = vtk.vtkTextProperty()
        tprop.SetColor(color)
        # tprop.ShadowOn()


        # axesActor.SetNumberOfLabels(4) # number of labels
        axesActor.SetUse2DMode(1)
        # axesActor.SetScreenSize(50.0) # for labels and axes titles
        # axesActor.SetLabelScaling(True,0,0,0)

        if Configuration.getSetting('ShowHorizontalAxesLabels'):
            axesActor.SetXAxisLabelVisibility(1)
        else:
            axesActor.SetXAxisLabelVisibility(0)

        if Configuration.getSetting('ShowVerticalAxesLabels'):
            axesActor.SetYAxisLabelVisibility(1)
        else:
            axesActor.SetYAxisLabelVisibility(0)


        # axesActor.SetAxisLabels(1,[0,10])




        # this was causing problems when x and y dimensions were different
        # axesActor.SetXLabelFormat("%6.4g")
        # axesActor.SetYLabelFormat("%6.4g")

        axesActor.SetBounds(0, horizontal_length, 0, vertical_length, 0, 0)

        axesActor.SetXTitle(horizontal_label)
        axesActor.SetYTitle(vertical_label)
        # axesActor.SetFlyModeToOuterEdges()

        label_prop = axesActor.GetLabelTextProperty(0)
        # print 'label_prop=',label_prop


        # print 'axesActor.GetXTitle()=',axesActor.GetXTitle()
        title_prop_x = axesActor.GetTitleTextProperty(0)
        # title_prop_x.SetLineOffset()
        # print 'axesActor.GetTitleTextProperty(0)=',axesActor.GetTitleTextProperty(0)

        # axesActor.GetTitleTextProperty(0).SetFontSize(50)

        # axesActor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
        # axesActor.GetTitleTextProperty(0).SetFontSize(6.0)
        # axesActor.GetTitleTextProperty(0).SetLineSpacing(0.5)
        # print 'axesActor.GetTitleTextProperty(0)=',axesActor.GetTitleTextProperty(0)

        axesActor.XAxisMinorTickVisibilityOff()
        axesActor.YAxisMinorTickVisibilityOff()

        axesActor.SetTickLocationToOutside()

        axesActor.GetTitleTextProperty(0).SetColor(color)
        axesActor.GetLabelTextProperty(0).SetColor(color)

        axesActor.GetXAxesLinesProperty().SetColor(color)
        axesActor.GetYAxesLinesProperty().SetColor(color)
        # axesActor.GetLabelTextProperty(0).SetColor(color)


        axesActor.GetTitleTextProperty(1).SetColor(color)
        axesActor.GetLabelTextProperty(1).SetColor(color)





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



    def initializeContoursHex(self,_dim,_conArray,_minMax,_contourActor):
#        print MODULENAME,'   initializeContoursHex():  _conArray=',_conArray
        data = vtk.vtkImageData()
        data.SetDimensions(_dim[0], _dim[1], 1)


        if VTK_MAJOR_VERSION >= 6:
            data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        else:
            data.SetScalarTypeToUnsignedChar()

        # data.SetScalarTypeToUnsignedChar()

        data.GetPointData().SetScalars(_conArray)
        field       = vtk.vtkImageDataGeometryFilter()
        # field.SetExtent(0, _dim[0], 0, int((_dim[1])*math.sqrt(3.0)), 0, 0)
        # field.SetExtent(0, _dim[0], 0, _dim[1]/2, 0, 0)

        if VTK_MAJOR_VERSION>=6:
            field.SetInputData(data)
        else:
            field.SetInput(data)


        transform = vtk.vtkTransform()
        # transform.Scale(1,math.sqrt(3.0)/2.0,1)
        # transform.Translate(0,math.sqrt(3.0)/2.0,0)


        transform.Scale(1,math.sqrt(3.0)/2.0,1)
        if self.currentDrawingParameters.planePos % 3 == 0:
            transform.Translate(0.5,0,0) #z%3==0
        elif self.currentDrawingParameters.planePos % 3 == 1:
            transform.Translate(0,math.sqrt(3.0)/4.0,0)#z%3==1
        else:
            transform.Translate(0.0,-math.sqrt(3.0)/4.0,0) #z%3==2



        isoContour = vtk.vtkContourFilter()

#        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):
        if True:
            isoContour.SetInputConnection(field.GetOutputPort())

            isoContour.GenerateValues(Configuration.getSetting("NumberOfContourLines",self.currentDrawingParameters.fieldName)+2, _minMax)

            tpd1 = vtk.vtkTransformPolyDataFilter()
            tpd1.SetInputConnection(isoContour.GetOutputPort())
            tpd1.SetTransform(transform)

            # self.contourMapper.SetInputConnection(contour.GetOutputPort())
            self.contour_mapper.SetInputConnection(tpd1.GetOutputPort())
            self.contour_mapper.SetLookupTable(self.ctlut)
            self.contour_mapper.SetScalarRange(_minMax)
            self.contour_mapper.ScalarVisibilityOff()
            _contourActor.SetMapper(self.contour_mapper)


    def initConFieldHexActors(self,_actors):
        # cellField  = sim.getPotts().getCellFieldG()
        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        conFieldName = self.currentDrawingParameters.fieldName
        print MODULENAME,'   initConFieldHexActors():  conFieldName=',conFieldName

        # # # print "drawing plane ",self.plane," planePos=",self.planePos
        # fieldDim = cellField.getDim()

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr=self.extractAddressIntFromVtkObject(self.conArray)
        self.hexPointsCon = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        self.hexPointsConIntAddr=self.extractAddressIntFromVtkObject(self.hexPointsCon)

        # ***************************************************************************
        self.hexCellsCon = vtk.vtkCellArray()
        self.hexCellsConIntAddr = self.extractAddressIntFromVtkObject(self.hexCellsCon)
        self.hexCellsConPolyData = vtk.vtkPolyData()

        # ***************************************************************************
        fillSuccessful = self.parentWidget.fieldExtractor.fillConFieldData2DHex(self.conArrayIntAddr,self.hexCellsConIntAddr,self.hexPointsConIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        # fillSuccessful=self.parentWidget.fieldExtractor.fillConFieldData2DHex(self.conArrayIntAddr,self.hexPointsIntAddr,conFieldName,self.plane, self.planePos)

        # fillSuccessful=self.parentWidget.fieldExtractor.fillConFieldData2DHex(self.conArrayIntAddr,self.hexPointsIntAddr,conFieldName,self.plane, self.planePos)
        if not fillSuccessful:
            return

        range = self.conArray.GetRange()
        self.minCon = range[0]
        self.maxCon = range[1]
        dim_0 = self.dim[0]+1
        dim_1 = self.dim[1]+1

        if Configuration.getSetting("MinRangeFixed",conFieldName):
            self.minCon = Configuration.getSetting("MinRange",conFieldName)

        if Configuration.getSetting("MaxRangeFixed",conFieldName):
            self.maxCon = Configuration.getSetting("MaxRange",conFieldName)


#        if Configuration.getSetting("ContoursOn",conFieldName):
        if True:
            contourActor = _actors[1]
            self.initializeContoursHex([self.dim[0], self.dim[1]],self.conArray,[self.minCon, self.maxCon],contourActor)

        self.hexCellsConPolyData.GetCellData().SetScalars(self.conArray)
        self.hexCellsConPolyData.SetPoints(self.hexPointsCon)
        self.hexCellsConPolyData.SetPolys(self.hexCellsCon)

        if VTK_MAJOR_VERSION>=6:
            self.hex_con_mapper.SetInputData(self.hexCellsConPolyData)
        else:
            self.hex_con_mapper.SetInput(self.hexCellsConPolyData)




        self.hex_con_mapper.ScalarVisibilityOn()
        self.hex_con_mapper.SetLookupTable(self.clut)
        self.hex_con_mapper.SetScalarRange(self.minCon, self.maxCon)

        _actors[0].SetMapper(self.hex_con_mapper)


    def initScalarFieldCellLevelHexActors(self,_actors):
        # cellField  = sim.getPotts().getCellFieldG()
        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        conFieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initScalarFieldCellLevelHexActors():  conFieldName=',conFieldName

        # # # print "drawing plane ",self.plane," planePos=",self.planePos
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr = self.extractAddressIntFromVtkObject(self.conArray)
        self.hexPointsCon = vtk.vtkPoints()

        self.hexPointsConIntAddr = self.extractAddressIntFromVtkObject(self.hexPointsCon)

        # ***************************************************************************
        self.hexCellsCon = vtk.vtkCellArray()
        self.hexCellsConIntAddr = self.extractAddressIntFromVtkObject(self.hexCellsCon)
        self.hexCellsConPolyData = vtk.vtkPolyData()

        # ***************************************************************************
        fillSuccessful = self.parentWidget.fieldExtractor.fillScalarFieldCellLevelData2DHex(self.conArrayIntAddr,self.hexCellsConIntAddr,self.hexPointsConIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        if not fillSuccessful:
            return

        range=self.conArray.GetRange()
        self.minCon = range[0]
        self.maxCon = range[1]
        dim_0 = self.dim[0]+1
        dim_1 = self.dim[1]+1

        if Configuration.getSetting("MinRangeFixed",conFieldName):
            self.minCon=Configuration.getSetting("MinRange",conFieldName)

        if Configuration.getSetting("MaxRangeFixed",conFieldName):
            self.maxCon=Configuration.getSetting("MaxRange",conFieldName)

#        if Configuration.getSetting("ContoursOn",conFieldName):
        if True:
            contourActor=_actors[1]
            self.initializeContoursHex([self.dim[0], self.dim[1]],self.conArray,[self.minCon, self.maxCon],contourActor)

        self.hexCellsConPolyData.GetCellData().SetScalars(self.conArray)
        self.hexCellsConPolyData.SetPoints(self.hexPointsCon)
        self.hexCellsConPolyData.SetPolys(self.hexCellsCon)

        if VTK_MAJOR_VERSION>=6:
            self.hex_con_mapper.SetInputData(self.hexCellsConPolyData)
        else:
            self.hex_con_mapper.SetInput(self.hexCellsConPolyData)

        self.hex_con_mapper.ScalarVisibilityOn()
        self.hex_con_mapper.SetLookupTable(self.clut)
        self.hex_con_mapper.SetScalarRange(self.minCon, self.maxCon)

        _actors[0].SetMapper(self.hex_con_mapper)


    def initScalarFieldHexActors(self,_actors):
        # cellField  = sim.getPotts().getCellFieldG()
        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        conFieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initScalarFieldHexActors():  conFieldName=',conFieldName

        # # # print "drawing plane ",self.plane," planePos=",self.planePos
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr=self.extractAddressIntFromVtkObject(self.conArray)
        self.hexPointsCon = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        self.hexPointsConIntAddr=self.extractAddressIntFromVtkObject(self.hexPointsCon)

        # ***************************************************************************
        self.hexCellsCon=vtk.vtkCellArray()

        self.hexCellsConIntAddr=self.extractAddressIntFromVtkObject(self.hexCellsCon)

        self.hexCellsConPolyData=vtk.vtkPolyData()

        # ***************************************************************************
        fillSuccessful=self.parentWidget.fieldExtractor.fillScalarFieldData2DHex(self.conArrayIntAddr,self.hexCellsConIntAddr,self.hexPointsConIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        if not fillSuccessful:
            return

        range=self.conArray.GetRange()
        self.minCon=range[0]
        self.maxCon=range[1]
        dim_0=self.dim[0]+1
        dim_1=self.dim[1]+1

        if Configuration.getSetting("MinRangeFixed",self.currentDrawingParameters.fieldName):
            self.minCon=Configuration.getSetting("MinRange",self.currentDrawingParameters.fieldName)

        if Configuration.getSetting("MaxRangeFixed",self.currentDrawingParameters.fieldName):
            self.maxCon=Configuration.getSetting("MaxRange",self.currentDrawingParameters.fieldName)

#        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):
        if True:
            contourActor=_actors[1]
            self.initializeContoursHex([self.dim[0], self.dim[1]],self.conArray,[self.minCon, self.maxCon],contourActor)


        self.hexCellsConPolyData.GetCellData().SetScalars(self.conArray)
        self.hexCellsConPolyData.SetPoints(self.hexPointsCon)
        self.hexCellsConPolyData.SetPolys(self.hexCellsCon)
        if VTK_MAJOR_VERSION>=6:
            self.hex_con_mapper.SetInputData(self.hexCellsConPolyData)
        else:
            self.hex_con_mapper.SetInput(self.hexCellsConPolyData)


        self.hex_con_mapper.ScalarVisibilityOn()
        self.hex_con_mapper.SetLookupTable(self.clut)
        self.hex_con_mapper.SetScalarRange(self.minCon, self.maxCon)

        _actors[0].SetMapper(self.hex_con_mapper)

    def drawConField(self, sim, fieldType):
        print MODULENAME,'  drawConField()'
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawConFieldHex(sim,fieldType)
            return

        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData2D") # this is simply a "pointer" to function
        self.drawScalarFieldData(sim,fieldType,fillScalarField)  # in MVCDrawView2D

    def drawScalarFieldCellLevel(self, sim, fieldType):
        print MODULENAME,'  drawScalarFieldCellLevel()'
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawScalarFieldCellLevelHex(sim,fieldType)
            return
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldCellLevelData2D") # this is simply a "pointer" to function
        self.drawScalarFieldData(sim,fieldType,fillScalarField)  # in MVCDrawView2D

    # def drawScalarField(self, sim, fieldType):
        # print MODULENAME,'  drawScalarField()'
        # if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            # self.drawScalarFieldHex(sim,fieldType)
            # return
        # fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillScalarFieldData2D") # this is simply a "pointer" to function
        # self.drawScalarFieldData(sim,fieldType,fillScalarField)  # in MVCDrawView2D

    def drawScalarField(self, sim, fieldType):
        # print MODULENAME,'  drawScalarField()'
        # print '\n\n\n\n\n drawScalarField',MODULENAME
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawScalarFieldHex(sim,fieldType)
            return
        fillScalarField = getattr(self.parentWidget.fieldExtractor, "fillConFieldData2D") # this is simply a "pointer" to function
        self.drawScalarFieldData(sim,fieldType,fillScalarField)  # in MVCDrawView2D


    def initializeContoursCartesian(self,_dim,_conArray,_minMax,_contourActor):

#        print MODULENAME,'   initializeContoursHex():  _conArray=',_conArray
        data = vtk.vtkImageData()
        data.SetDimensions(_dim[0], _dim[1], 1)

        if VTK_MAJOR_VERSION  >= 6:
            data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,3)
        else:
            data.SetScalarTypeToUnsignedChar()


        data.GetPointData().SetScalars(_conArray)
        field = vtk.vtkImageDataGeometryFilter()
        # field.SetExtent(0, _dim[0], 0, int((_dim[1])*math.sqrt(3.0)), 0, 0)
        # field.SetExtent(0, _dim[0], 0, _dim[1]/2, 0, 0)
        if VTK_MAJOR_VERSION>=6:
            field.SetInputData(data)
        else:
            field.SetInput(data)

        transform = vtk.vtkTransform()
        # transform.Scale(1,math.sqrt(3.0)/2.0,1)
        transform.Translate(0.5,0.5,0) # for some reason there is   is offset  of (0.5,0.5,0) when generating contours

        isoContour = vtk.vtkContourFilter()

#        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):

        if True:
            isoContour.SetInputConnection(field.GetOutputPort())
            isoContour.GenerateValues(Configuration.getSetting("NumberOfContourLines",self.currentDrawingParameters.fieldName)+2, _minMax)

            tpd1 = vtk.vtkTransformPolyDataFilter()
            tpd1.SetInputConnection(isoContour.GetOutputPort())
            tpd1.SetTransform(transform)

            # self.contourMapper.SetInputConnection(contour.GetOutputPort())
            self.contour_mapper.SetInputConnection(tpd1.GetOutputPort())
            self.contour_mapper.SetLookupTable(self.ctlut)
            self.contour_mapper.SetScalarRange(_minMax)
            self.contour_mapper.ScalarVisibilityOff()
            _contourActor.SetMapper(self.contour_mapper)


    # def initScalarFieldCartesianActors(self,_actors):
    def initScalarFieldCartesianActors(self, _fillScalarField,_actors):

        # cellField  = sim.getPotts().getCellFieldG()
        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        conFieldName = self.currentDrawingParameters.fieldName
        # print MODULENAME,'   initScalarFieldHexActors():  conFieldName=',conFieldName

        # # # print "drawing plane ",self.plane," planePos=",self.planePos
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr=self.extractAddressIntFromVtkObject(self.conArray)
        self.cartesianPointsCon = vtk.vtkPoints()
        # self.hexPoints.SetName("hexpoints")
        self.cartesianPointsConIntAddr=self.extractAddressIntFromVtkObject(self.cartesianPointsCon)

        # ***************************************************************************
        self.cartesianCellsCon=vtk.vtkCellArray()

        self.cartesianCellsConIntAddr=self.extractAddressIntFromVtkObject(self.cartesianCellsCon)

        self.cartesianCellsConPolyData=vtk.vtkPolyData()

        # ***************************************************************************
        # # # fillSuccessful=self.parentWidget.fieldExtractor.fillConFieldData2DCartesian(self.conArrayIntAddr,self.cartesianCellsConIntAddr,self.cartesianPointsConIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        fillSuccessful=_fillScalarField(self.conArrayIntAddr,self.cartesianCellsConIntAddr,self.cartesianPointsConIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
        # print 'fillSuccessful=',fillSuccessful
        if not fillSuccessful:
            return

        range=self.conArray.GetRange()
        self.minCon=range[0]
        self.maxCon=range[1]
        dim_0=self.dim[0]+1
        dim_1=self.dim[1]+1


        if Configuration.getSetting("MinRangeFixed",self.currentDrawingParameters.fieldName):
            self.minCon=Configuration.getSetting("MinRange",self.currentDrawingParameters.fieldName)

        if Configuration.getSetting("MaxRangeFixed",self.currentDrawingParameters.fieldName):
            self.maxCon=Configuration.getSetting("MaxRange",self.currentDrawingParameters.fieldName)

#        if Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName):
        if True:
            contourActor=_actors[1]
            self.initializeContoursCartesian([self.dim[0], self.dim[1]],self.conArray,[self.minCon, self.maxCon],contourActor)


        self.cartesianCellsConPolyData.GetCellData().SetScalars(self.conArray)
        self.cartesianCellsConPolyData.SetPoints(self.cartesianPointsCon)
        self.cartesianCellsConPolyData.SetPolys(self.cartesianCellsCon)

        if VTK_MAJOR_VERSION>=6:
            self.conMapper.SetInputData(self.cartesianCellsConPolyData)
        else:
            self.conMapper.SetInput(self.cartesianCellsConPolyData)

        self.conMapper.ScalarVisibilityOn()
        self.conMapper.SetLookupTable(self.clut)
        self.conMapper.SetScalarRange(self.minCon, self.maxCon)

        _actors[0].SetMapper(self.conMapper)






    def initScalarFieldActors(self, _fillScalarField,_actors):
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        # fieldDim   = cellField.getDim()
        # conField   = CompuCell.getConcentrationField(sim, fieldType[0])
        # conFieldName=fieldType[0]

        #print self._statusBar.currentMessage()
        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        conFieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initScalarFieldActors():  conFieldName=',conFieldName
        # conFieldName=fieldType[0]

        self.dim    = [fieldDim.x, fieldDim.y, fieldDim.z]
#        print MODULENAME,'   initScalarFieldActors():  self.dim=',self.dim

        # Leave it for testing
        assert self.currentDrawingParameters.plane in ("XY", "XZ", "YZ"), "Plane is not XY, XZ or YZ"

        # fieldDim = cellField.getDim()
        dimOrder = self.dimOrder(self.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        self.conArray = vtk.vtkDoubleArray()
        self.conArray.SetName("concentration")
        self.conArrayIntAddr = self.extractAddressIntFromVtkObject(self.conArray)
        fillSuccessful = _fillScalarField(self.conArrayIntAddr,conFieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
        if not fillSuccessful:
            return

        numIsos = Configuration.getSetting("NumberOfContourLines",conFieldName)
#        self.isovalStr = Configuration.getSetting("ScalarIsoValues",conFieldName)

        range = self.conArray.GetRange()
        self.minCon = range[0]
        self.maxCon = range[1]
#        print MODULENAME,'   initScalarFieldActors():  (before) self.minCon, maxCon=',self.minCon,self.maxCon
#        print MODULENAME,'   initScalarFieldActors():  numberOfTableColors, highTableValue=',self.numberOfTableColors,self.highTableValue  # 1024 (1.0, 0.0, 0.0, 1.0)
#        print MODULENAME,'  initScalarFieldActors():  doing Config-.getSetting(MinRange*)'

        # Note! should really avoid doing a getSetting with each step to speed up the rendering; only update when changed in Prefs
        if Configuration.getSetting("MinRangeFixed",conFieldName):
            self.minCon = Configuration.getSetting("MinRange",conFieldName)
#            self.clut.SetTableValue(0,[0,0,0,1])   # this will cause values < minCon to be black
#        else:
#            self.clut.SetTableValue(0,self.lowTableValue)

        if Configuration.getSetting("MaxRangeFixed",conFieldName):
            self.maxCon = Configuration.getSetting("MaxRange",conFieldName)


        dim_0 = self.dim[0]+1
        dim_1 = self.dim[1]+1

        # print 'dim_0,dim_1=',(dim_0,dim_1)
        dbgMsg('dim_0,dim_1=',(dim_0,dim_1))

        data = vtk.vtkImageData()
        data.SetDimensions(dim_0, dim_1, 1)
        # print "dim_0,dim_1",(dim_0,dim_1)

#
        if VTK_MAJOR_VERSION >= 6:
            data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,3)
        else:
            data.SetScalarTypeToUnsignedChar()

        data.GetPointData().SetScalars(self.conArray)

        field = vtk.vtkImageDataGeometryFilter()

        if VTK_MAJOR_VERSION>=6:
            field.SetInputData(data)
        else:
            field.SetInput(data)


        field.SetExtent(0, dim_0, 0, dim_1, 0, 0)

#        spoints = vtk.vtkStructuredPoints()
#        spoints.SetDimensions(self.dim[0]+2, self.dim[1]+2, self.dim[2]+2)  #  only add 2 if we're filling in an extra boundary (rf. FieldExtractor.cpp)
#        spoints.GetPointData().SetScalars(self.conArray)

#        voi = vtk.vtkExtractVOI()
#        voi.SetInput(spoints)
#        voi.SetVOI(1,self.dim[0]-1, 1,self.dim[1]-1, 1,self.dim[2]-1 )

        isoContour = vtk.vtkContourFilter()
#        isoContour.SetInputConnection(voi.GetOutputPort())
        isoContour.SetInputConnection(field.GetOutputPort())


        isoValList = self.getIsoValues(conFieldName)
#        print MODULENAME, 'initScalarFieldActors():  getIsoValues=',isoValList

        printIsoValues = False
#        if printIsoValues:  print MODULENAME, ' isovalues= ',
        isoNum = 0
        for isoVal in isoValList:
            try:
                if printIsoValues:  print MODULENAME, '  initScalarFieldActors(): setting (specific) isoval= ',isoVal
                isoContour.SetValue(isoNum, isoVal)
                isoNum += 1
            except:
                print MODULENAME, '  initScalarFieldDataActors(): cannot convert to float: ',self.isovalStr[idx]
        if isoNum > 0:  isoNum += 1
#        print MODULENAME, '  after specific isovalues, isoNum=',isoNum
#        numIsos = Configuration.getSetting("NumberOfContourLines")
#        print MODULENAME, '  Next, do range of isovalues: min,max, # isos=',self.minCon,self.maxCon,numIsos
        delIso = (self.maxCon - self.minCon)/(numIsos+1)  # exclude the min,max for isovalues
#        print MODULENAME, '  initScalarFieldActors(): delIso= ',delIso
        isoVal = self.minCon + delIso
        for idx in xrange(numIsos):
            if printIsoValues:  print MODULENAME, '  initScalarFieldDataActors(): isoNum, isoval= ',isoNum,isoVal
            isoContour.SetValue(isoNum, isoVal)
            isoNum += 1
            isoVal += delIso
        if printIsoValues:  print


        isoContour.SetInputConnection(field.GetOutputPort())  # rwh?
#        isoContour.GenerateValues(Configuration.getSetting("NumberOfContourLines",self.currentDrawingParameters.fieldName)+2, [self.minCon, self.maxCon])

        self.contour_mapper.SetInputConnection(isoContour.GetOutputPort())
        self.contour_mapper.SetLookupTable(self.ctlut)
        self.contour_mapper.SetScalarRange(self.minCon, self.maxCon)
        self.contour_mapper.ScalarVisibilityOff()  # this is required to do a SetColor on the actor's property
#            print MODULENAME,' initScalarFieldActors:  setColor=1,0,0'
#            contourActor.GetProperty().SetColor(1.,0.,0.)
#        if Configuration.getSetting("ContoursOn",conFieldName):
        contourActor = _actors[1]
        contourActor.SetMapper(self.contour_mapper)

        color = Configuration.getSetting("ContourColor")  # want to avoid this; only update when Prefs changes
        contourActor.GetProperty().SetColor(float(color.red())/255, float(color.green())/255, float(color.blue())/255)


        self.conMapper.SetInputConnection(field.GetOutputPort()) # port index = 0

        self.conMapper.ScalarVisibilityOn()
        self.conMapper.SetLookupTable(self.clut)
        self.conMapper.SetScalarRange(self.minCon, self.maxCon) #0, self.clut.GetNumberOfColors()) # may manually set range so that type reassignment will not be scalled dynamically when one type is missing

        self.conMapper.SetScalarModeToUsePointData()

        _actors[0].SetMapper(self.conMapper)   # concentration actor


    def drawVectorField(self, bsd, fieldType):
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawVectorFieldDataHex(bsd,fieldType)
            return
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldData2D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData2D
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)

    def drawVectorFieldCellLevel(self, bsd, fieldType):
        if self.parentWidget.latticeType==Configuration.LATTICE_TYPES["Hexagonal"] and self.plane=="XY": # drawing in other planes will be done on a rectangular lattice
            self.drawVectorFieldCellLevelDataHex(bsd,fieldType)
            return
        fillVectorField = getattr(self.parentWidget.fieldExtractor, "fillVectorFieldCellLevelData2D") # this is simply a "pointer" to function self.parentWidget.fieldExtractor.fillVectorFieldData2D
        self.drawVectorFieldData(bsd,fieldType,fillVectorField)

    def initVectorFieldDataHexActors(self,_actors):
        # # # print "INSIDE drawVectorFieldDataHex"
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        fieldDim   = self.currentDrawingParameters.bsd.fieldDim

        fieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initVectorFieldDataHexActors():  fieldName=',fieldName


        #print self._statusBar.currentMessage()
        self.dim  = [fieldDim.x, fieldDim.y, fieldDim.z]

        assert self.currentDrawingParameters.plane in ("XY", "XZ", "YZ"), "Plane is not XY, XZ or YZ"


        # fieldDim = cellField.getDim()
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        dim_0=self.dim[0]+1
        dim_1=self.dim[1]+1


        vectorGrid=vtk.vtkUnstructuredGrid()

        points=vtk.vtkPoints()
        vectors=vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        vectorsIntAddr=self.extractAddressIntFromVtkObject(vectors)

        fillSuccessful=self.parentWidget.fieldExtractor.fillVectorFieldData2DHex(pointsIntAddr,vectorsIntAddr,fieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
        if not fillSuccessful:
            return

        vectorGrid.SetPoints(points)
        vectorGrid.GetPointData().SetVectors(vectors)

        cone=vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        #cone.SetRadius(4)

        range=vectors.GetRange(-1)

        self.minMagnitude=range[0]
        self.maxMagnitude=range[1]


        if Configuration.getSetting("MinRangeFixed",fieldName):
            self.minMagnitude = Configuration.getSetting("MinRange",fieldName)
#            self.clut.SetTableValue(0,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(0,self.lowTableValue)

        if Configuration.getSetting("MaxRangeFixed",fieldName):
            self.maxMagnitude = Configuration.getSetting("MaxRange",fieldName)
#            self.clut.SetTableValue(self.numberOfTableColors-1,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(self.numberOfTableColors-1,self.highTableValue)



        glyphs=vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION>=6:
            glyphs.SetInputData(vectorGrid)
        else:
            glyphs.SetInput(vectorGrid)


        glyphs.SetSourceConnection(cone.GetOutputPort())
        #glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()
        # glyphs.SetScaleFactor(Configuration.getSetting("ArrowLength")) # scaling arrows here ArrowLength indicates scaling factor not actual length

        arrowScalingFactor=Configuration.getSetting("ArrowLength",fieldName) # scaling factor for and arrow  -  ArrowLength indicates scaling factor not actual length

        if Configuration.getSetting("FixedArrowColorOn",fieldName):
            glyphs.SetScaleModeToScaleByVector()
            rangeSpan=self.maxMagnitude-self.minMagnitude
            dataScalingFactor=max(abs(self.minMagnitude),abs(self.maxMagnitude))
#            print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor==0.0:
                dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)
            #coloring arrows

            r,g,b = Configuration.getSetting("ArrowColor",fieldName)
#            r = arrowColor.red()
#            g = arrowColor.green()
#            b = arrowColor.blue()
#            _actors[0].GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
            _actors[0].GetProperty().SetColor(r, g, b)
        else:
            if Configuration.getSetting("ScaleArrowsOn",fieldName):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                rangeSpan=self.maxMagnitude-self.minMagnitude
                dataScalingFactor=max(abs(self.minMagnitude),abs(self.maxMagnitude))
#                print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

                if dataScalingFactor==0.0:
                    dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)

        # # # print "range=",range
        # # # print "vectors.GetNumberOfTuples()=",vectors.GetNumberOfTuples()
        # self.glyphsMapper.SetScalarRange(vectors.GetRange(-1)) # this will return the range of magnitudes of all the vectors store int vtkFloatArray
        # self.glyphsMapper.SetScalarRange(range)
        self.glyphs_mapper.SetScalarRange([self.minMagnitude, self.maxMagnitude])

        _actors[0].SetMapper(self.glyphs_mapper)


    def initVectorFieldCellLevelDataHexActors(self,_actors):
        # # # print "INSIDE drawVectorFieldDataHex"
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        fieldDim   = self.currentDrawingParameters.bsd.fieldDim

        fieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initVectorFieldCellLevelDataHexActors():  fieldName=',fieldName


        #print self._statusBar.currentMessage()
        self.dim    = [fieldDim.x, fieldDim.y, fieldDim.z]

        assert self.currentDrawingParameters.plane in ("XY", "XZ", "YZ"), "Plane is not XY, XZ or YZ"


        # fieldDim = cellField.getDim()
        dimOrder    = self.dimOrder(self.currentDrawingParameters.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        dim_0=self.dim[0]+1
        dim_1=self.dim[1]+1

        vectorGrid=vtk.vtkUnstructuredGrid()

        points=vtk.vtkPoints()
        vectors=vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        vectorsIntAddr=self.extractAddressIntFromVtkObject(vectors)

        fillSuccessful=self.parentWidget.fieldExtractor.fillVectorFieldCellLevelData2DHex(pointsIntAddr,vectorsIntAddr,fieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
        if not fillSuccessful:
            return

        vectorGrid.SetPoints(points)
        vectorGrid.GetPointData().SetVectors(vectors)

        cone=vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        #cone.SetRadius(4)

        range=vectors.GetRange(-1)

        self.minMagnitude=range[0]
        self.maxMagnitude=range[1]

        if Configuration.getSetting("MinRangeFixed",fieldName):
            self.minMagnitude = Configuration.getSetting("MinRange",fieldName)
#            self.clut.SetTableValue(0,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(0,self.lowTableValue)

        if Configuration.getSetting("MaxRangeFixed",fieldName):
            self.maxMagnitude = Configuration.getSetting("MaxRange",fieldName)
#            self.clut.SetTableValue(self.numberOfTableColors-1,[0,0,0,1])
#        else:
#            self.clut.SetTableValue(self.numberOfTableColors-1,self.highTableValue)

        glyphs=vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION>=6:
            glyphs.SetInputData(vectorGrid)
        else:
            glyphs.SetInput(vectorGrid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        #glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()
        # glyphs.SetScaleFactor(Configuration.getSetting("ArrowLength")) # scaling arrows here ArrowLength indicates scaling factor not actual length

        arrowScalingFactor=Configuration.getSetting("ArrowLength",fieldName) # sscaling factor for and arrow  -  ArrowLength indicates scaling factor not actual length

        if Configuration.getSetting("FixedArrowColorOn",fieldName):
            glyphs.SetScaleModeToScaleByVector()
            rangeSpan=self.maxMagnitude-self.minMagnitude
            dataScalingFactor=max(abs(self.minMagnitude),abs(self.maxMagnitude))
#            print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor==0.0:
                dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)
            #coloring arrows

            r,g,b = Configuration.getSetting("ArrowColor",fieldName)
#            r = arrowColor.red()
#            g = arrowColor.green()
#            b = arrowColor.blue()
#            _actors[0].GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
            _actors[0].GetProperty().SetColor(r, g, b)
        else:
            if Configuration.getSetting("ScaleArrowsOn",fieldName):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                rangeSpan=self.maxMagnitude-self.minMagnitude
                dataScalingFactor=max(abs(self.minMagnitude),abs(self.maxMagnitude))
#                print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

                if dataScalingFactor==0.0:
                    dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrowScalingFactor)


        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)

        # # # print "range=",range
        # # # print "vectors.GetNumberOfTuples()=",vectors.GetNumberOfTuples()
        # self.glyphsMapper.SetScalarRange(vectors.GetRange(-1)) # this will return the range of magnitudes of all the vectors store int vtkFloatArray
        # self.glyphsMapper.SetScalarRange(range)
        self.glyphs_mapper.SetScalarRange([self.minMagnitude, self.maxMagnitude])

        _actors[0].SetMapper(self.glyphs_mapper)

    def initVectorFieldCellLevelActors(self, _fillVectorFieldFcn, _actors):
        # potts      = sim.getPotts()
        # cellField  = potts.getCellFieldG()
        # fieldDim   = cellField.getDim()

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
        fieldName = self.currentDrawingParameters.fieldName
#        print MODULENAME,'   initVectorFieldCellLevelActors():  fieldName=',fieldName

        #print self._statusBar.currentMessage()
        self.dim    = [fieldDim.x, fieldDim.y, fieldDim.z]

        assert self.plane in ("XY", "XZ", "YZ"), "Plane is not XY, XZ or YZ"


        # fieldDim = cellField.getDim()
        dimOrder    = self.dimOrder(self.plane)
        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]

        dim_0=self.dim[0]+1
        dim_1=self.dim[1]+1

        vectorGrid=vtk.vtkUnstructuredGrid()

        points=vtk.vtkPoints()
        vectors=vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("visVectors")

        pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        vectorsIntAddr=self.extractAddressIntFromVtkObject(vectors)

        fillSuccessful=_fillVectorFieldFcn(pointsIntAddr,vectorsIntAddr,fieldName,self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)
        if not fillSuccessful:
            return

        vectorGrid.SetPoints(points)
        vectorGrid.GetPointData().SetVectors(vectors)

        cone=vtk.vtkConeSource()
        cone.SetResolution(5)
        cone.SetHeight(2)
        cone.SetRadius(0.5)
        #cone.SetRadius(4)

        range = vectors.GetRange(-1)

        self.minMagnitude = range[0]
        self.maxMagnitude = range[1]

#        if Configuration.getSetting("MinMagnitudeFixed"):
#            self.minMagnitude=Configuration.getSetting("MinMagnitude")
        if Configuration.getSetting("MinRangeFixed",fieldName):
            self.minMagnitude = Configuration.getSetting("MinRange",fieldName)

        if Configuration.getSetting("MaxRangeFixed",fieldName):
            self.maxMagnitude = Configuration.getSetting("MaxRange",fieldName)

        glyphs=vtk.vtkGlyph3D()

        if VTK_MAJOR_VERSION>=6:
            glyphs.SetInputData(vectorGrid)
        else:
            glyphs.SetInput(vectorGrid)

        glyphs.SetSourceConnection(cone.GetOutputPort())
        #glyphs.SetScaleModeToScaleByVector()
        # glyphs.SetColorModeToColorByVector()

        # print "glyphs.GetScaleFactor()=",glyphs.GetScaleFactor()
        # print "glyphs.GetScaleFactor()=",
        # rwh: should use of this factor depend on the state of the "Scale arrow length" checkbox?
        arrowScalingFactor = Configuration.getSetting("ArrowLength",fieldName) # scaling factor for an arrow (ArrowLength indicates scaling factor not actual length)

        if Configuration.getSetting("FixedArrowColorOn",fieldName):
            glyphs.SetScaleModeToScaleByVector()
            rangeSpan = self.maxMagnitude-self.minMagnitude
            dataScalingFactor = max(abs(self.minMagnitude),abs(self.maxMagnitude))
#            print MODULENAME,"initVectorFieldCellLevelActors():  self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

            if dataScalingFactor==0.0:
                dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
            glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)
            #coloring arrows

#            arrowColor = Configuration.getSetting("ArrowColor",fieldName)
            color = Configuration.getSetting("ArrowColor",fieldName)
            r,g,b = color.red(), color.green(), color.blue()
#            print MODULENAME,"initVectorFieldCellLevelActors():  arrowColor=",arrowColor   # QColor or #ffffff  (hex format)?
#            print MODULENAME,"initVectorFieldCellLevelActors():  arrowColor r,g,b=",r,g,b
#            r, g, b = arrowColor[1:3], arrowColor[3:5], arrowColor[5:]
#            r, g, b = [int(n, 16) for n in (r, g, b)]
#            if len(arrowColor) == 3:
#                print MODULENAME,"initVectorFieldCellLevelActors():  got arrowColor= (r,g,b) triple; got ",arrowColor
##                raise
#                r, g, b = arrowColor
#            else:
#            print MODULENAME,"initVectorFieldCellLevelActors():  try to extract arrowColor.red(), etc"
#            r = arrowColor.red()
#            g = arrowColor.green()
#            b = arrowColor.blue()
#            _actors[0].GetProperty().SetColor(self.toVTKColor(r), self.toVTKColor(g), self.toVTKColor(b))
            _actors[0].GetProperty().SetColor(r, g, b)
        else:
            if Configuration.getSetting("ScaleArrowsOn",fieldName):
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleModeToScaleByVector()

                rangeSpan = self.maxMagnitude-self.minMagnitude
                dataScalingFactor = max(abs(self.minMagnitude),abs(self.maxMagnitude))
#                print "self.minMagnitude=",self.minMagnitude," self.maxMagnitude=",self.maxMagnitude

                if dataScalingFactor==0.0:
                    dataScalingFactor=1.0 # in this case we are plotting 0 vectors and in this case data scaling factor will be set to 1
                glyphs.SetScaleFactor(arrowScalingFactor/dataScalingFactor)

            else:
                glyphs.SetColorModeToColorByVector()
                glyphs.SetScaleFactor(arrowScalingFactor)

        self.glyphs_mapper.SetInputConnection(glyphs.GetOutputPort())
        self.glyphs_mapper.SetLookupTable(self.clut)


        # # # print "range=",range
        # # # print "vectors.GetNumberOfTuples()=",vectors.GetNumberOfTuples()
        # self.glyphsMapper.SetScalarRange(vectors.GetRange(-1)) # this will return the range of magnitudes of all the vectors store int vtkFloatArray
        # self.glyphsMapper.SetScalarRange(range)
        self.glyphs_mapper.SetScalarRange([self.minMagnitude, self.maxMagnitude])

        _actors[0].SetMapper(self.glyphs_mapper)

    # Optimize code?
    def dimOrder(self, plane):
        plane=string.lower(plane)
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
        plane=string.lower(plane)
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

    # ?
    def drawContourLines(self):
        pass

    def wheelEvent(self, ev):
        self.__zoomStep(ev.delta())

    # Overrides the mousePressEvent() method from QVTKRenderWidget
    def mousePressEvent(self,ev):
        if (ev.button() == 1):
            self._Mode = "Pan"
            self._ActiveButton = ev.button()
            #self.PickActor(ev.x(), ev.y())
            #print self.GetPicker()
            #self.showTip(ev.x(), ev.y())

        elif (ev.button() == 2):
            self._Mode = "Zoom"
            self._ActiveButton = ev.button()

        self.UpdateRenderer(ev.x(),ev.y())

    def event(self, ev):
        if ev.type() == QEvent.ToolTip:
            self.showTip(ev)
        return QWidget.event(self, ev)

    def showTip(self, ev):
        # toll tips are not enabled in this release
        return
        import CompuCell
        pt = CompuCell.Point3D()

        self.PickActor(ev.x(), ev.y())
        id = self.GetPicker().GetCellId()
        if id != -1:
            pos = self.GetPicker().GetPickPosition()
            pt.x, pt.y, pt.z = int(pos[0]), int(pos[1]), 0

            if  self.cellField.get(pt) is not None and self.cellField.get(pt).id != 0:
                QToolTip.hideText()
                QToolTip.showText(ev.globalPos(), self.toolTip(self.cellField.get(pt)))

    # Overrides the Zoom() method from QVTKRenderWidget
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


    # def zoomIn(self):
    #     delta = 2*120
    #     self.__zoomStep(delta)
    #
    # def zoomOut(self):
    #     delta = -2*120
    #     self.__zoomStep(delta)
    #
    # def zoomFixed(self, val):
    #     if self.ren:
    #         # renderer = self._CurrentRenderer
    #         camera = self.ren.GetActiveCamera()
    #         self.__curDist = camera.GetDistance()
    #
    #         # To zoom fixed, dolly should be set to initial position
    #         # and then moved to a new specified position!
    #         if (self.__initDist != 0):
    #             # You might need to rewrite the fixed zoom in case if there
    #             # will be flickering
    #             camera.Dolly(self.__curDist/self.__initDist)
    #
    #         camera.Dolly(self.zitems[val])
    #         self.ren.ResetCameraClippingRange()
    #
    #         self.Render()

     # Never used?!  Rf. same fns in MVCDrawView2D.py
#    def takeShot_rwh(self):
#        filter = "PNG files (*.png)"
#        fileName = QFileDialog.getSaveFileName(\
#            self,
#            "Save Screenshot",
#            os.getcwd(),
#            filter
#            )
#
#        # Other way to get the correct file name: fileName.toAscii().data())
#        if fileName is not None and fileName != "":
#            self.takeSimShot(str(fileName))
#
#    # fileName - full file name (e.g. "/home/user/shot.png")
#    def takeSimShot_rwh(self, fileName):
#        # DON'T REMOVE!
#        # Better quality
#        # Passes vtkRenderer. Takes actual screenshot of the region within the widget window
#        # If other application are present within this region it will shoot them also
#
#        renderLarge = vtk.vtkRenderLargeImage()
#        renderLarge.SetInput(self.graphicsFrameWidget.ren)
#        renderLarge.SetMagnification(1)
#
#        # We write out the image which causes the rendering to occur. If you
#        # watch your screen you might see the pieces being rendered right
#        # after one another.
#        writer = vtk.vtkPNGWriter()
#        writer.SetInputConnection(renderLarge.GetOutputPort())
#        # # # print "GOT HERE fileName=",fileName
#        writer.SetFileName(fileName)
#
#        writer.Write()

    def toolTip(self, cellG):
        return "Id:             %s\nType:       %s\nVolume:  %s" % (cellG.id, cellG.type, cellG.volume)

    def configsChanged(self):  # never called?!
        print MODULENAME,'  >>>>>>>>>>   configsChange()    <<<<<<<<<<<<<<<<<<'
        self.populateLookupTable()
        self.setBorderColor()
        # Doesn't work, gives error:
        # vtkScalarBarActor (0x8854218): Need a mapper to render a scalar bar
        #self.showLegend(Configuration.getSetting("LegendEnable"))

#        self.showContours(Configuration.getSetting("ContoursOn",self.currentDrawingParameters.fieldName))
        self.showContours(True)
        self.parentWidget.requestRedraw()

    # this function is used during prototyping. in production code it is replaced by C++ counterpart
#    def fillCellFieldData_old(self,_cellFieldG, plane=None, pos=None):
#        import CompuCell
#
#        pt = CompuCell.Point3D()
#        cell = CompuCell.CellG()
#        fieldDim = _cellFieldG.getDim()
#        if plane is None:
#            self.dim = [fieldDim.x , fieldDim.y , fieldDim.z]
#        else:
#            dimOrder    = self.dimOrder(plane)
#            pointOrder  = self.pointOrder(plane)
#            self.dim        = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
#            # # # print "pointOrder=",pointOrder
#
#        # # # print "FILLCELLFIELDDATA"
#        # # # print "self.dim=",self.dim
#        offset = 0
#        if plane is None:
#            self.cellType = vtk.vtkIntArray()
#            self.cellType.SetName("celltype")
##            self.cellType.SetNumberOfValues((self.dim[2]+1)*(self.dim[1]+1)*(self.dim[0]+1))
#            self.cellType.SetNumberOfValues((self.dim[2])*(self.dim[1]+1)*(self.dim[0]))
#
##            self.cellId = [[[0 for k in range(self.dim[2])] for j in range(self.dim[1]+1)] for i in range(self.dim[0]+1)]
#            self.cellId = [[[0 for k in range(self.dim[2])] for j in range(self.dim[1])] for i in range(self.dim[0])]
#
#            # For some reasons the points x=0 are eaten up (don't know why).
#            # So we just populate empty cellIds.
##            for i in range(self.dim[0]+1):
#            for i in range(self.dim[0]):
#                self.cellType.SetValue(offset, 0)
#                offset += 1
#
#            for k in range(self.dim[2]):
##                for j in range(self.dim[1]+1):
#                for j in range(self.dim[1]):
##                    for i in range(self.dim[0]+1):
#                    for i in range(self.dim[0]):
#                        pt.x = i
#                        pt.y = j
#                        pt.z = k
#                        cell = _cellFieldG.get(pt)
#                        if cell is not None:
#                            type    = int(cell.type)
#                            id      = int(cell.id)
#                        else:
#                            type    = 0
#                            id      = 0
#                        self.cellType.InsertValue(offset, type)
#                        # print "inserting type ",type," offset ",offset
#
#                        offset += 1
#
#                        self.cellId[i][j][k] = id
#        else:
#            self.cellType = vtk.vtkIntArray()
#            self.cellType.SetName("celltype")
##            self.cellType.SetNumberOfValues((self.dim[1]+2)*(self.dim[0]+1))
#            self.cellType.SetNumberOfValues((self.dim[1])*(self.dim[0]))
#
##            self.cellId = [[0 for j in range(self.dim[1]+1)] for i in range(self.dim[0]+1)]
#            self.cellId = [[0 for j in range(self.dim[1])] for i in range(self.dim[0])]
#
#            # For some reasons the points x=0 are eaten up (don't know why).
#            # So we just populate empty cellIds.
##            for i in range(self.dim[0]+1):
#            for i in range(self.dim[0]):
#                self.cellType.SetValue(offset, 0)
#                offset += 1
#
#            # for k in range(self.dim[2]):
##            for j in range(self.dim[1]+1):
##                for i in range(self.dim[0]+1):
#            for j in range(self.dim[1]):
#                for i in range(self.dim[0]):
#                    point = self.planeMapper(pointOrder, (i, j, pos))
#                    pt.x = point[0]
#                    pt.y = point[1]
#                    pt.z = point[2]
#
#                    # pt.x = i
#                    # pt.y = j
#                    # pt.z = k
#                    cell = _cellFieldG.get(pt)
#                    # print "pt=",pt," cell=",cell
#                    if cell is not None:
#                        type    = int(cell.type)
#                        id      = int(cell.id)
#                    else:
#                        type    = 0
#                        id      = 0
#                    self.cellType.InsertValue(offset, type)
#                    # print "inserting type ",type," offset ",offset
#
#                    offset += 1
#
#                    self.cellId[i][j] = id

    #this function is used during prototyping. in production code it is replaced by C++ counterpart
#    def fillConFieldData_old(self,_cellFieldG,_conField, plane=None, pos=None):
#        import CompuCell
#
#        pt = CompuCell.Point3D(0,0,0)
#        cell = CompuCell.CellG()
#        fieldDim = _cellFieldG.getDim()
#
#        dimOrder    = self.dimOrder(plane)
#        pointOrder  = self.pointOrder(plane)
#        self.dim    = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))  # [fieldDim.x, fieldDim.y, fieldDim.z]
#        # # # print "pointOrder=",pointOrder
#
#        # # # print "FILL CONFIELDDATA"
#        # # # print "self.dim=",self.dim
#
#        self.conArray = vtk.vtkDoubleArray()
#        self.conArray.SetName("concentration")
#        self.conArray.SetNumberOfValues((self.dim[1]+2)*(self.dim[0]+1))
#
#        offset=0
#        # For some reasons the points x=0 are eaten up (don't know why).
#        # So we just populate empty cellIds.
#        for i in range(self.dim[0]+1):
#            self.conArray.SetValue(offset, 0.0)
#            offset += 1
#
#
#        maxCon = float(_conField.get(pt)) # concentration at pt=0,0,0
#        minCon = float(_conField.get(pt)) # concentration at pt=0,0,0
#
#        # for k in range(self.dim[2]):
#        for j in range(self.dim[1]+1):
#            for i in range(self.dim[0]+1):
#                point = self.planeMapper(pointOrder, (i, j, pos))
#                pt.x = point[0]
#                pt.y = point[1]
#                pt.z = point[2]
#
#                if i==self.dim[0] or j==self.dim[1]:
#                    con=0.0
#                else:
#                    con = float(_conField.get(pt))
#
#                self.conArray.SetValue(offset, con)
#
#                if maxCon < con:
#                    maxCon = con
#
#                if minCon > con:
#                    minCon = con
#
#                offset += 1
#
#        return (minCon, maxCon, self.dim[0]+1,self.dim[1]+1)


    def initClusterBordersActors2D(self,clusterBordersActor):
#        print MODULENAME,' initClusterBordersActors2D'
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        linesIntAddr=self.extractAddressIntFromVtkObject(lines)

        self.parentWidget.fieldExtractor.fillClusterBorderData2D(pointsIntAddr , linesIntAddr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION>=6:
            self.clusterBorderMapper.SetInputData(borders)
        else:
            self.clusterBorderMapper.SetInput(borders)

        clusterBordersActor.SetMapper(self.clusterBorderMapper)

    def initClusterBordersActors2DHex(self,clusterBordersActor):
#        print MODULENAME,' initClusterBordersActors2DHex'
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        linesIntAddr=self.extractAddressIntFromVtkObject(lines)

        self.parentWidget.fieldExtractor.fillClusterBorderData2DHex(pointsIntAddr , linesIntAddr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

        borders = vtk.vtkPolyData()

        borders.SetPoints(points)
        borders.SetLines(lines)

        if VTK_MAJOR_VERSION>=6:
            self.clusterBorderMapperHex.SetInputData(borders)
        else:
            self.clusterBorderMapperHex.SetInput(borders)


        clusterBordersActor.SetMapper(self.clusterBorderMapperHex)

#--------------------------------------------------------------------------------------------------
    def initCellGlyphsActor2D(self,cellGlyphActor):
#        print MODULENAME,'  initCellGlyphsActor2D'

        from PySteppables import CellList

        #points = vtk.vtkPoints()
        #lines = vtk.vtkCellArray()
        #pointsIntAddr=self.extractAddressIntFromVtkObject(points)
        #linesIntAddr=self.extractAddressIntFromVtkObject(lines)
        #self.parentWidget.fieldExtractor.fillCentroidData2D(pointsIntAddr , linesIntAddr, self.currentDrawingParameters.plane, self.currentDrawingParameters.planePos)

#        self.drawCentroids()   # doing in pure Python

        #centroids = vtk.vtkPolyData()
        #centroids.SetPoints(points)
        #centroids.SetLines(lines)

        #self.centroidMapper.SetInput(centroids)
        #cellGlyphActor.SetMapper(self.centroidMapper)

#    def drawCentroids(self):
#        print MODULENAME,'  initCellGlyphsActor2D:   self.parentWidget.latticeType=',self.parentWidget.latticeType
        fieldDim=self.currentDrawingParameters.bsd.fieldDim
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList=CellList(inventory)
        centroidPoints = vtk.vtkPoints()
        cellTypes = vtk.vtkIntArray()
        cellTypes.SetName("CellTypes")

#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")
        cellScalars = vtk.vtkFloatArray()
        cellScalars.SetName("CellScalars")


        # test for Jim's liver lobule
#        cellTypeScale = vtk.vtkFloatArray()
#        cellTypeScale.SetName("CellTypeScale")

        scaleByVolume = Configuration.getSetting("CellGlyphScaleByVolumeOn")       # todo: make class attrib; update only when changes

#        print MODULENAME,'  initCellGlyphsActor2D: self.offset=',self.offset
        if self.hexFlag:
          print MODULENAME,'  initCellGlyphsActor2D:   ----- doing hexFlag block'
          for cell in cellList:
#            if cell.volume > 0:
#              xmid = float(cell.xCM) / cell.volume
#              ymid = float(cell.yCM) / cell.volume
#              xmid = xmid/1.07457
              xmid = cell.xCOM/1.07457
              ymid = cell.yCOM/1.07457
              centroidPoints.InsertNextPoint(xmid,ymid,0.0)
              cellTypes.InsertNextValue(cell.type)
              if scaleByVolume:
                  cellScalars.InsertNextValue(math.sqrt(cell.volume))   # A(circle) = pi * r^2
              else:
                  cellScalars.InsertNextValue(1.0)      # lame way of doing this
#            else:
#              print 'cell.id, .volume=',cell.id,cell.volume

        else:   # square (non-hex) lattice
          for cell in cellList:

#          if cell.type == 8:  # hep cell
#              xmid = float(cell.xCM) / cell.volume
#              ymid = float(cell.yCM) / cell.volume
#              centroidPoints.InsertNextPoint(xmid,ymid,0.0)
#              cellTypes.InsertNextValue(cell.type)
#              cellTypeScale.InsertNextValue(20.0)
#          elif cell.type == 5:  # sinusoid cell
#              xmid = float(cell.xCM) / cell.volume
#              ymid = float(cell.yCM) / cell.volume
#              centroidPoints.InsertNextPoint(xmid,ymid,0.0)
#              cellTypes.InsertNextValue(cell.type)
#              cellTypeScale.InsertNextValue(10.0)

          #print 'cell.id=',cell.id  # = 2,3,4,...
          #print 'cell.type=',cell.type
          #print 'cell.volume=',cell.volume
#            if cell.volume > 0:
#              xmid = float(cell.xCM) / cell.volume + self.offset
              xmid = cell.xCOM #  + self.offset
              ymid = cell.yCOM # + self.offset
              centroidPoints.InsertNextPoint(xmid,ymid,0.0)
              cellTypes.InsertNextValue(cell.type)
#              cellVolumes.InsertNextValue(cell.volume)

              if scaleByVolume:
                  cellScalars.InsertNextValue(math.sqrt(cell.volume))   # A(circle) = pi * r^2
              else:
                  cellScalars.InsertNextValue(1.0)      # lame way of doing this

#            else:
#              print 'cell.id, .volume=',cell.id,cell.volume



        centroidsPD = vtk.vtkPolyData()
        centroidsPD.SetPoints(centroidPoints)
        centroidsPD.GetPointData().SetScalars(cellTypes)
#        centroidsPD.GetPointData().AddArray(cellVolumes)  # scale glyph size by cell volume
        centroidsPD.GetPointData().AddArray(cellScalars)  # scale by ~radius

        centroidGS = vtk.vtkGlyphSource2D()
        centroidGS.SetGlyphTypeToCircle()
        #centroidGS.SetScale(1)
        #gs.FilledOff()
        #gs.CrossOff()

        centroidGlyph = vtk.vtkGlyph3D()
        if VTK_MAJOR_VERSION>=6:
            centroidGlyph.SetInputData(centroidsPD)
        else:
            centroidGlyph.SetInput(centroidsPD)

        try:
            centroidGlyph.SetSource(centroidGS.GetOutput())
        except AttributeError:
            centroidGlyph.SetSourceData(centroidGS.GetOutput())


#        centroidGlyph.SetScaleFactor( 0.2 )  # rwh: should this lattice size dependent or cell vol or ?
        glyphScale = Configuration.getSetting("CellGlyphScale")
        centroidGlyph.SetScaleFactor( glyphScale )
        #centroidGlyph.SetIndexModeToScalar()
        #centroidGlyph.SetRange(0,2)

        #centroidGlyph.SetScaleModeToDataScalingOff()
        centroidGlyph.SetColorModeToColorByScalar()
        centroidGlyph.SetScaleModeToScaleByScalar()
        centroidGlyph.SetRange(0,self.celltypeLUTMax)

        centroidGlyph.SetInputArrayToProcess(3,0,0,0,"CellTypes")
#        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellVolumes")
        centroidGlyph.SetInputArrayToProcess(0,0,0,0,"CellScalars")

        if VTK_MAJOR_VERSION>=6:
            self.cellGlyphsMapper.SetInputData(centroidGlyph.GetOutput())
        else:
            self.cellGlyphsMapper.SetInput(centroidGlyph.GetOutput())

        self.cellGlyphsMapper.SetScalarRange(0,self.celltypeLUTMax)
        self.cellGlyphsMapper.ScalarVisibilityOn()
        self.cellGlyphsMapper.SetLookupTable(self.celltypeLUT)

        cellGlyphActor.SetMapper(self.cellGlyphsMapper)

#--------------------------------------------------------------------------------------------------
    def initFPPLinksActor2D_orig(self,fppActor):
#        print MODULENAME,'  initFPPLinksActor2D'

        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor2D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):
          print '    fppPlugin is null, returning'
          return

        fieldDim=self.currentDrawingParameters.bsd.fieldDim
        cellField = self.currentDrawingParameters.bsd.sim.getPotts().getCellFieldG()
        inventory = self.currentDrawingParameters.bsd.sim.getPotts().getCellInventory()
        #print 'inventory=',type(inventory)  # = <class 'CompuCell.CellInventory'>
        cellList = CellList(inventory)

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
#        cellTypes = vtk.vtkIntArray()
#        cellTypes.SetName("CellTypes")
#        cellVolumes = vtk.vtkIntArray()
#        cellVolumes.SetName("CellVolumes")

        # figure out which list we need
        for cell in cellList:
            numList1 = sum(1 for _ in FocalPointPlasticityDataList(fppPlugin, cell))
            numList2 = sum(1 for _ in InternalFocalPointPlasticityDataList(fppPlugin, cell))
#            print MODULENAME,' numList1=',numList1
#            print MODULENAME,' numList2=',numList2
            break

        dataList = FocalPointPlasticityDataList
        if numList2 > 0:
            dataList = InternalFocalPointPlasticityDataList

        beginPt = 0
        for cell in cellList:
#          print '\n cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue
#          xmid = float(cell.xCM) / vol
#          ymid = float(cell.yCM) / vol
#          print MODULENAME,'cell.id=',cell.id,'  x,y (begin)=',xmid,ymid
#          points.InsertNextPoint(xmid,ymid,0)
          points.InsertNextPoint(cell.xCOM,cell.yCOM,0)

          endPt = beginPt + 1

#          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
          for fppd in dataList(fppPlugin, cell):

#            print '   nbrId=',fppd.neighborAddress.id
#            vol = fppd.neighborAddress.volume
#            xmid=fppd.neighborAddress.xCOM
#            ymid=fppd.neighborAddress.yCOM
#            print '    x,y (end)=',xmid,ymid
#            points.InsertNextPoint(xmid,ymid,0)
            points.InsertNextPoint(fppd.neighborAddress.xCOM, fppd.neighborAddress.yCOM, 0)

            lines.InsertNextCell(2)  # our line has 2 points
#            print beginPt,' -----> ',endPt
            lines.InsertCellPoint(beginPt)
            lines.InsertCellPoint(endPt)
            endPt += 1
          beginPt = endPt


        FPPLinksPD = vtk.vtkPolyData()
        FPPLinksPD.SetPoints(points)
        FPPLinksPD.SetLines(lines)

        if VTK_MAJOR_VERSION>=6:
            self.FPPLinksMapper.SetInputData(FPPLinksPD)
        else:
            self.FPPLinksMapper.SetInput(FPPLinksPD)



        fppActor.SetMapper(self.FPPLinksMapper)

#--------------------------------------------------------------------------------------------------
    def initFPPLinksActor2D(self, fppActor):
#        print MODULENAME,'  initFPPLinksActor2D'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor2D:  fppPlugin=',fppPlugin
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

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0

        for cell in cellList:
#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
          vol = cell.volume
          if vol < self.eps: continue

          if self.hexFlag:
            xmid0 = cell.xCOM/1.07457
            ymid0 = cell.yCOM/1.07457
          else:
            xmid0 = cell.xCOM # + self.offset
            ymid0 = cell.yCOM # + self.offset
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,0)

          endPt = beginPt + 1
#2345678901234
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
#           for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            if self.hexFlag:
              xmid = fppd.neighborAddress.xCOM/1.07457
              ymid = fppd.neighborAddress.yCOM/1.07457
            else:
              xmid = fppd.neighborAddress.xCOM # + self.offset
              ymid = fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1

                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,0)
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
                points.InsertNextPoint(xmid,ymid,0)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' (internal link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#2345678901234
#          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            if self.hexFlag:
              xmid = fppd.neighborAddress.xCOM/1.07457
              ymid = fppd.neighborAddress.yCOM/1.07457
            else:
              xmid=fppd.neighborAddress.xCOM # + self.offset
              ymid=fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1

                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,0)
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
                points.InsertNextPoint(xmid,ymid,0)
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

        fppActor.SetMapper(self.FPPLinksMapper)

#--------------------------------------------------------------------------------------------------
    def initFPPLinksLabelsActor2D(self, fppActor):
        print MODULENAME,'  initFPPLinksLabelsActor2D'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor2D:  fppPlugin=',fppPlugin
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

        labelStr = vtk.vtkStringArray()
        labelStr.SetName('fppLabels')
        labelsPD = vtk.vtkPolyData()
        labelVerts = vtk.vtkCellArray()

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0

        for cell in cellList:
#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue

          if self.hexFlag:
            xmid0 = cell.xCOM/1.07457
            ymid0 = cell.yCOM/1.07457
          else:
            xmid0 = cell.xCOM # + self.offset
            ymid0 = cell.yCOM # + self.offset
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0
          points.InsertNextPoint(xmid0,ymid0,0)

          endPt = beginPt + 1
#2345678901234
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
#           for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            if self.hexFlag:
              xmid = fppd.neighborAddress.xCOM/1.07457
              ymid = fppd.neighborAddress.yCOM/1.07457
            else:
              xmid = fppd.neighborAddress.xCOM # + self.offset
              ymid = fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1

                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,0)
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
                points.InsertNextPoint(xmid,ymid,0)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' (internal link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                lineNum += 1
                endPt += 1

#2345678901234
#          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            if self.hexFlag:
              xmid = fppd.neighborAddress.xCOM/1.07457
              ymid = fppd.neighborAddress.yCOM/1.07457
            else:
              xmid=fppd.neighborAddress.xCOM # + self.offset
              ymid=fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist  > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    lineNum += 1

                    endPt += 1
                else:   # wraps around in y-direction
#                    print '>>>>>> wraparound Y'
                    xmid0end = xmid0
                    if ydiff < 0:
                      ymid0end = ymid0 + self.stubSize
                    else:
                      ymid0end = ymid0 - self.stubSize
                    points.InsertNextPoint(xmid0end,ymid0end,0)
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
                points.InsertNextPoint(xmid,ymid,0)
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

        fppActor.SetMapper(self.FPPLinksMapper)

#--------------------------------------------------------------------------------------------------
    def initFPPLinksColorActor2D_buggy(self, fppActor):
        print MODULENAME,'  initFPPLinksColorActor2D'
        from PySteppables import CellList, FocalPointPlasticityDataList, InternalFocalPointPlasticityDataList
        import CompuCell

        fppPlugin = CompuCell.getFocalPointPlasticityPlugin()
#        print '    initFPPLinksActor2D:  fppPlugin=',fppPlugin
        if (fppPlugin == 0):  # bogus check
          print '    fppPlugin is null, returning'
          return

        fieldDim = self.currentDrawingParameters.bsd.fieldDim
#        print MODULENAME,'fieldDim, fieldDim.x =',fieldDim,fieldDim.x
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

        beginPt = 0
#        numCells = sum(1 for _ in cellList)
#        print MODULENAME,'  numCell=',numCells
        lineNum = 0

        scalarValMin = 1000.0
        scalarValMax = -scalarValMin

        for cell in cellList:
#          print MODULENAME,'--cell (addr) = ',cell
#          print 'cell.id=',cell.id  # = 2,3,4,...
#          print 'cell.type=',cell.type
#          print 'cell.volume=',cell.volume
#          vol = cell.volume
#          if vol < self.eps: continue

          if self.hexFlag:
            xmid0 = cell.xCOM/1.07457
            ymid0 = cell.yCOM/1.07457
          else:
            xmid0 = cell.xCOM # + self.offset
            ymid0 = cell.yCOM # + self.offset
#          print 'cell.id=',cell.id,'  x,y,z (begin)=',xmid0,ymid0,zmid0

          points.InsertNextPoint(xmid0,ymid0,0)

          endPt = beginPt + 1

#2345678901234
          for fppd in InternalFocalPointPlasticityDataList(fppPlugin, cell):
#2345678901234
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid = fppd.neighborAddress.xCOM # + self.offset
            ymid = fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist > fppd.maxDistance:   # implies we have wraparound (via periodic BCs); should also verify periodic BC
                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2

                    scalarVal = actualDist - fppd.targetDistance    # Abbas's metric
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
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                    scalarVal = actualDist - fppd.targetDistance    # Abbas's metric
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
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
                points.InsertNextPoint(xmid,ymid,0)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' (internal link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)

                lineNum += 1
                endPt += 1

#2345678901234
          for fppd in FocalPointPlasticityDataList(fppPlugin, cell):
#            print '   nbrId=',fppd.neighborAddress.id
#            if beginPt < 10:
#             print 'targetDistance,maxDistance=',fppd.targetDistance,fppd.maxDistance
#targetDistance,maxDistance= 3.0 6.0
#targetDistance,maxDistance= 2.0 4.0
#            vol = fppd.neighborAddress.volume
#            if vol < self.eps: continue
            xmid = fppd.neighborAddress.xCOM # + self.offset
            ymid = fppd.neighborAddress.yCOM # + self.offset
#            print '    x,y,z (end)=',xmid,ymid,zmid
#            points.InsertNextPoint(xmid,ymid,zmid)
            xdiff = xmid-xmid0
            ydiff = ymid-ymid0
            actualDist = math.sqrt((xdiff*xdiff)+(ydiff*ydiff))
#            if beginPt < 10:
#              print beginPt,')----- actualDist, maxDist= ',actualDist, fppd.maxDistance
#            if d2 > fppd.maxDistance*fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
            if actualDist > fppd.maxDistance:   # implies we have wraparound (via periodic BCs)
#                print '>>>>>> wraparound w/ beginPt=',beginPt
                # add dangling "out" line to beginning cell
                if abs(xdiff) > abs(ydiff):   # wraps around in x-direction
#                    print '>>>>>> wraparound X'
                    if xdiff < 0:
                      xmid0end = xmid0 + self.stubSize
                    else:
                      xmid0end = xmid0 - self.stubSize
                    ymid0end = ymid0
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = xdim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                    scalarVal = actualDist - fppd.targetDistance    # Abbas's metric
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
                    points.InsertNextPoint(xmid0end,ymid0end,0)
                    lines.InsertNextCell(2)  # our line has 2 points
                    lines.InsertCellPoint(beginPt)
                    lines.InsertCellPoint(endPt)

                    # coloring the FPP links
                    actualDist = ydim - actualDist   # compute (approximate) real actualDist
#                    targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                    scalarVal = actualDist - fppd.targetDistance    # Abbas's metric
                    if scalarVal < scalarValMin: scalarValMin = scalarVal
                    if scalarVal > scalarValMax: scalarValMax = scalarVal
#                    colorScalars.SetValue(lineNum, scalarVal)
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
                points.InsertNextPoint(xmid,ymid,0)
                lines.InsertNextCell(2)  # our line has 2 points
#                print beginPt,' ----- (external link, no wrap) -----> ',endPt
#                print beginPt,' (external link, no wrap) -----> ',endPt
                lines.InsertCellPoint(beginPt)
                lines.InsertCellPoint(endPt)

                # coloring the FPP links
#                targetDist2 = fppd.targetDistance * fppd.targetDistance   # targetDist^2
                scalarVal = actualDist - fppd.targetDistance    # Abbas prefers this
#                scalarVal = actualDist
                if scalarVal < scalarValMin: scalarValMin = scalarVal
                if scalarVal > scalarValMax: scalarValMax = scalarVal
                colorScalars.InsertNextValue(scalarVal)

                lineNum += 1
                endPt += 1

#2345678901234
#          print 'after external links: beginPt, endPt=',beginPt,endPt
          beginPt = endPt  # update point index

        #-----------------------
        if lineNum == 0:  return
#        print '---------- # links=',lineNum

        # create Blue-Red LUT
        lutBlueRed = vtk.vtkLookupTable()
        lutBlueRed.SetHueRange(0.667,0.0)
        lutBlueRed.Build()

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

#        self.FPPLinksMapper.SetScalarModeToUseCellFieldData()

        fppActor.SetMapper(self.FPPLinksMapper)

#        scalarBar = vtk.vtkScalarBarActor()
#        scalarBar.SetLookupTable(self.lutBlueRed)
#        #scalarBar.SetTitle("Stress")
#        scalarBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
#        #scalarBar.GetPositionCoordinate().SetValue(0.8,0.05)
#        scalarBar.SetOrientationToVertical()
#        scalarBar.SetWidth(0.1)
#        scalarBar.SetHeight(0.9)
#        scalarBar.SetPosition(0.88,0.1)
#        #scalarBar.SetLabelFormat("%-#6.3f")
#        scalarBar.SetLabelFormat("%-#3.1f")
#        scalarBar.GetLabelTextProperty().SetColor(1,1,1)
#        #scalarBar.GetTitleTextProperty().SetColor(1,0,0)
#
##        self.graphicsFrameWidget.ren.AddActor2D(scalarBar)

#--------------------------------------------------------------------------------------------------
    #this function is used during prototyping. in production code it is replaced by C++ counterpart
#    def drawBorders_old(self):
#        # Draw borders
#        points = vtk.vtkPoints()
#        lines = vtk.vtkCellArray()
#
#        k = 0 # dim[2]-1 -- k = 0 for 2D lattice
#        pc = 0 # point counter
#        # Add lines for the very bottom edge
#
#        for i in range(self.dim[0]):
#            for j in range(self.dim[1]):
#                if (i > 0) and (j < self.dim[1]) and (self.cellId[i][j] != self.cellId[i-1][j]):
#                    points.InsertNextPoint(i,j,0)
#                    points.InsertNextPoint(i,j+1,0)
#                    pc+=2
#                    lines.InsertNextCell(2)
#                    lines.InsertCellPoint(pc-2)
#                    lines.InsertCellPoint(pc-1)
#
#                if (j > 0) and (i<self.dim[0]) and (self.cellId[i][j] != self.cellId[i][j-1]):
#                    points.InsertNextPoint(i,j,0)
#                    points.InsertNextPoint(i+1,j,0)
#                    pc+=2
#                    lines.InsertNextCell(2)
#                    lines.InsertCellPoint(pc-2)
#                    lines.InsertCellPoint(pc-1)
#
#                if (i < self.dim[0]) and (j < self.dim[1]) and (self.cellId[i][j] != self.cellId[i+1][j]):
#                    points.InsertNextPoint(i+1,j,0)
#                    points.InsertNextPoint(i+1,j+1,0)
#                    pc+=2
#                    lines.InsertNextCell(2)
#                    lines.InsertCellPoint(pc-2)
#                    lines.InsertCellPoint(pc-1)
#
#                if (j < self.dim[1]) and (i < self.dim[1]) and (self.cellId[i][j] != self.cellId[i][j+1]):
#                    points.InsertNextPoint(i,j+1,0)
#                    points.InsertNextPoint(i+1,j+1,0)
#                    pc+=2
#                    lines.InsertNextCell(2)
#                    lines.InsertCellPoint(pc-2)
#                    lines.InsertCellPoint(pc-1)
#
#        borders = vtk.vtkPolyData()
#
#        borders.SetPoints(points)
#        borders.SetLines(lines)
#
#        self.borderMapper.SetInput(borders)
#        self.borderActor.SetMapper(self.borderMapper)
#        # self.setBorderColor()
#        self.borderActor.GetProperty().SetColor(1,1,1)
#        if not self.currentActors.has_key("BorderActor"):
#            self.currentActors["BorderActor"]=self.borderActor
#            self.ren.AddActor(self.borderActor)
#        else:
#            # will ensure that borders is the last item to draw
#            actorsCollection=self.ren.GetActors()
#            if actorsCollection.GetLastItem()!=self.borderActor:
#                self.ren.RemoveActor(self.borderActor)
#                self.ren.AddActor(self.borderActor)

    def HexCoordXY(self,x,y,z):
        from math import sqrt
        if(z%2):
            if(y%2):
                return [x , sqrt(3.0)/2.0*(y+2.0/3.0), z*sqrt(6.0)/3.0 ]
            else:
                return [ x+0.5 ,  sqrt(3.0)/2.0*(y+2.0/3.0) , z*sqrt(6.0)/3.0]

        else:
            if(y%2):
                return [x , sqrt(3.0)/2.0*y, z*sqrt(6.0)/3.0 ]
            else:
                return [ x+0.5 ,  sqrt(3.0)/2.0*y , z*sqrt(6.0)/3.0]


#    def drawBordersHex(self):
#        self.hexVerts=[]
#        import math
#        sqrt_3_3=math.sqrt(3.0)/3.0
#        self.hexVerts.append([0, sqrt_3_3, 0.0])
#        self.hexVerts.append([0.5 , 0.5*sqrt_3_3, 0.0])
#        self.hexVerts.append([0.5, -0.5*sqrt_3_3, 0.0])
#        self.hexVerts.append([0. , -sqrt_3_3, 0.0])
#        self.hexVerts.append([-0.5 , -0.5*sqrt_3_3, 0.0])
#        self.hexVerts.append([-0.5, 0.5*sqrt_3_3, 0.0])
#
#        # Draw borders
#        points = vtk.vtkPoints()
#        lines = vtk.vtkCellArray()
#
#        k = 0 # dim[2]-1 -- k = 0 for 2D lattice
#        pc = 0 # point counter
#        # Add lines for the very bottom edge
#
#        for i in range(self.dim[0]):
#            for j in range(self.dim[1]):
#                hexPt = self.HexCoordXY(i,j,0)
#                if j%2:
#                    if i-1>=0 and self.cellId[i][j] != self.cellId[i-1][j]:
#                        points.InsertNextPoint(self.hexVerts[4][0]+hexPt[0],self.hexVerts[4][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[5][0]+hexPt[0],self.hexVerts[5][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i-1>=0 and j+1<self.dim[1] and self.cellId[i][j] != self.cellId[i-1][j+1]:
#                        points.InsertNextPoint(self.hexVerts[5][0]+hexPt[0],self.hexVerts[5][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[0][0]+hexPt[0],self.hexVerts[0][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if j+1<self.dim[1] and self.cellId[i][j] != self.cellId[i][j+1]:
#                        points.InsertNextPoint(self.hexVerts[0][0]+hexPt[0],self.hexVerts[0][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[1][0]+hexPt[0],self.hexVerts[1][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i+1<self.dim[0] and self.cellId[i][j] != self.cellId[i+1][j]:
#                        points.InsertNextPoint(self.hexVerts[1][0]+hexPt[0],self.hexVerts[1][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[2][0]+hexPt[0],self.hexVerts[2][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if j-1>=0 and self.cellId[i][j] != self.cellId[i][j-1]:
#                        points.InsertNextPoint(self.hexVerts[2][0]+hexPt[0],self.hexVerts[2][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[3][0]+hexPt[0],self.hexVerts[3][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i-1>=0 and j-1>= 0 and self.cellId[i][j] != self.cellId[i-1][j-1]:
#                        points.InsertNextPoint(self.hexVerts[3][0]+hexPt[0],self.hexVerts[3][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[4][0]+hexPt[0],self.hexVerts[4][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                else:
#                    if i-1>=0 and self.cellId[i][j] != self.cellId[i-1][j]:
#                        points.InsertNextPoint(self.hexVerts[4][0]+hexPt[0],self.hexVerts[4][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[5][0]+hexPt[0],self.hexVerts[5][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if j+1<self.dim[1] and self.cellId[i][j] != self.cellId[i][j+1]:
#                        points.InsertNextPoint(self.hexVerts[5][0]+hexPt[0],self.hexVerts[5][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[0][0]+hexPt[0],self.hexVerts[0][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i+1<self.dim[0] and j+1<self.dim[1] and self.cellId[i][j] != self.cellId[i+1][j+1]:
#                        points.InsertNextPoint(self.hexVerts[0][0]+hexPt[0],self.hexVerts[0][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[1][0]+hexPt[0],self.hexVerts[1][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i+1<self.dim[0] and self.cellId[i][j] != self.cellId[i+1][j]:
#                        points.InsertNextPoint(self.hexVerts[1][0]+hexPt[0],self.hexVerts[1][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[2][0]+hexPt[0],self.hexVerts[2][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if i+1<self.dim[0] and j-1>= 0 and self.cellId[i][j] != self.cellId[i+1][j-1]:
#                        points.InsertNextPoint(self.hexVerts[2][0]+hexPt[0],self.hexVerts[2][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[3][0]+hexPt[0],self.hexVerts[3][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#                    if j-1>=0 and self.cellId[i][j] != self.cellId[i][j-1]:
#                        points.InsertNextPoint(self.hexVerts[3][0]+hexPt[0],self.hexVerts[3][1]+hexPt[1],0)
#                        points.InsertNextPoint(self.hexVerts[4][0]+hexPt[0],self.hexVerts[4][1]+hexPt[1],0)
#                        pc+=2
#                        lines.InsertNextCell(2)
#                        lines.InsertCellPoint(pc-2)
#                        lines.InsertCellPoint(pc-1)
#
#
#
#        borders = vtk.vtkPolyData()
#
#        borders.SetPoints(points)
#        borders.SetLines(lines)
#
#        self.borderMapperHex.SetInput(borders)
#        self.borderActorHex.SetMapper(self.borderMapperHex)
#        # self.setBorderColor()
#        self.borderActorHex.GetProperty().SetColor(1,1,1)
#        if not self.currentActors.has_key("BorderActorHex"):
#            self.currentActors["BorderActorHex"]=self.borderActorHex
#            self.ren.AddActor(self.borderActorHex)
#        else:
#            # wil ensure that borders is the last item to draw
#            actorsCollection=self.ren.GetActors()
#            if actorsCollection.GetLastItem()!=self.borderActorHex:
#                self.ren.RemoveActor(self.borderActorHex)
#                self.ren.AddActor(self.borderActorHex)


    def initCellFieldActorsData(self,_actors):

        dim=[self.dim[0]+1,self.dim[1]+1,self.dim[2]+1]
#        dim = [self.dim[0],self.dim[1],self.dim[2]]  #rwh
#        print MODULENAME,'  initCellFieldActorsData(),   dim=',dim

        uGridConc = vtk.vtkStructuredPoints()
        uGridConc.SetDimensions(dim[0],dim[1],dim[2])

        uGridConc.GetPointData().SetScalars(self.cellType)

        cellsPlane=vtk.vtkImageDataGeometryFilter()
        cellsPlane.SetExtent(0,dim[0],0,dim[1],0,0)
        if VTK_MAJOR_VERSION>=6:
            cellsPlane.SetInputData(uGridConc)
        else:
            cellsPlane.SetInput(uGridConc)

        # concMapper=self.cellsMapper

        self.cellsMapper.SetInputConnection(cellsPlane.GetOutputPort())
        self.cellsMapper.ScalarVisibilityOn()

        self.cellsMapper.SetLookupTable(self.celltypeLUT)  # def'd in parent class
        self.cellsMapper.SetScalarRange(0,self.celltypeLUTMax)

        # self.cellsActor.SetMapper(self.cellsMapper)
        _actors[0].SetMapper(self.cellsMapper)

        imageViewer = vtk.vtkImageViewer2()

        # if not self.currentActors.has_key("CellsActor"):
            # self.currentActors["CellsActor"]=self.cellsActor
            # self.graphicsFrameWidget.ren.AddActor(self.cellsActor)
            # # print "\n\n\n\n added CELLS ACTOR"

        # self.prepareOutlineActor(dim)
        # self.showOutlineActor()

#    def drawCellFieldHex_old(self, sim, fieldType):
#
#        cellField  = sim.getPotts().getCellFieldG()
#
#        # # # print "INSIDE drawCellFieldHex"
#        # # # print "drawing plane ",self.plane," planePos=",self.planePos
#        fieldDim = cellField.getDim()
#        dimOrder    = self.dimOrder(self.plane)
#        self.dim = self.planeMapper(dimOrder, (fieldDim.x, fieldDim.y, fieldDim.z))# [fieldDim.x, fieldDim.y, fieldDim.z]
#
#        self.cellType = vtk.vtkIntArray()
#        self.cellType.SetName("celltype")
#        self.cellTypeIntAddr=self.extractAddressIntFromVtkObject(self.cellType)
#        self.hexPoints = vtk.vtkPoints()
#        # self.hexPoints.SetName("hexpoints")
#        self.hexPointsIntAddr=self.extractAddressIntFromVtkObject(self.hexPoints)
#
#        self.parentWidget.fieldExtractor.fillCellFieldData2DHex_old(self.cellTypeIntAddr,self.hexPointsIntAddr,self.plane, self.planePos)
#        # if self.parentWidget.borderAct.isChecked():
#            # self.drawBorders2DHex()
#            # # self.drawBordersHex()
#        # return
#        # Python function used during prototyping
#        # self.fillCellFieldData(cellField,self.plane, self.planePos)
#
#        hexagonSrc=vtk.vtkRegularPolygonSource()
#        from math import sqrt
#        hexagonSrc.SetNumberOfSides(6)
#        hexagonSrc.SetRadius(sqrt(3)/3.0)
#
#        # creating/display the lattice verts is optional
#        hexPixelsPD = vtk.vtkPolyData()
#        hexPixelsPD.SetPoints(self.hexPoints)
#        hexPixelsPD.GetPointData().SetScalars(self.cellType)
#
#        hexGlyphs = vtk.vtkGlyph3D()
#        hexGlyphs.SetInput(hexPixelsPD)
#        hexGlyphs.SetSource(0,hexagonSrc.GetOutput())
#        hexGlyphs.SetIndexModeToScalar()
#        hexGlyphs.SetScaleModeToDataScalingOff()
#
#        # hexCellsMapper = vtk.vtkPolyDataMapper()
#        self.hexCellsMapper.SetInput(hexGlyphs.GetOutput())
#        self.hexCellsMapper.SetLookupTable(self.celltypeLUT)  # def'd in parent class
#        self.hexCellsMapper.SetScalarRange(0,self.celltypeLUTMax)
#        self.hexCellsMapper.ScalarVisibilityOn()
#
#        self.hexCellsActor.SetMapper(self.hexCellsMapper)
#
#        if self.currentActors.has_key("CellsActor"):
#            self.ren.RemoveActor(self.currentActors["CellsActor"])
#            del self.currentActors["CellsActor"]
#
#        if self.currentActors.has_key("BorderActor"):
#            self.ren.RemoveActor(self.currentActors["BorderActor"])
#            del self.currentActors["BorderActor"]
#
#        if not self.currentActors.has_key("HexCellsActor"):
#            self.currentActors["HexCellsActor"]=self.hexCellsActor
#            self.ren.AddActor(self.hexCellsActor)
#
#        if self.parentWidget.borderAct.isChecked():
#            self.drawBorders2DHex()
#        else:
#            self.hideBorder()
#
#            # self.drawBordersHex()
#
#
#        import math
##        self.prepareOutlineActor([self.dim[0]+1, int(self.dim[1]*math.sqrt(3.0)/2.0)+2, 1])
#        self.prepareOutlineActor([self.dim[0], int(self.dim[1]*math.sqrt(3.0)/2.0), 1])
#        self.showOutlineActor()
#
#
#        # self.repaint()
#        self.Render()