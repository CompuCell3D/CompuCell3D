from .MVCDrawViewBase import MVCDrawViewBase
import vtk
from collections import OrderedDict
from copy import deepcopy
from cc3d.core.GraphicsOffScreen.MetadataHandler import MetadataHandler

MODULENAME = '==== MVCDrawView3D.py:  '


class MVCDrawView3D(MVCDrawViewBase):
    def __init__(self, _drawModel, ren=None):
        MVCDrawViewBase.__init__(self, _drawModel, ren=ren)

        self.initArea()
        self.setParams()
        # self.usedCellTypesList=None
        # self.usedDraw3DFlag=False
        # self.boundingBox = Configuration.getSetting("BoundingBoxOn")
        # self.show3DAxes = Configuration.getSetting("ShowAxes")
        self.warnUserCellBorders = True

    def initArea(self):
        '''
        Sets up the VTK simulation area
        :return:None
        '''
        # Zoom items
        self.zitems = []

        self.actors_dict = {}

        self.cellTypeActors = {}
        self.outlineActor = vtk.vtkActor()
        self.outlineDim = [0, 0, 0]

        self.invisibleCellTypes = {}
        self.typesInvisibleStr = ""
        self.set3DInvisibleTypes()

        self.axesActor = vtk.vtkCubeAxesActor2D()

        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0, 1.0)
        self.clut.SetValueRange(1.0, 1.0)
        self.clut.SetAlphaRange(1.0, 1.0)
        self.clut.SetNumberOfColors(1024)
        self.clut.Build()

        ## Set up the mapper and actor (3D) for concentration field.
        # self.conMapper = vtk.vtkPolyDataMapper()
        self.conActor = vtk.vtkActor()

        self.glyphsActor = vtk.vtkActor()
        # self.glyphsMapper=vtk.vtkPolyDataMapper()

        self.cellGlyphsActor = vtk.vtkActor()
        self.FPPLinksActor = vtk.vtkActor()

        # Weird attributes
        self.typeActors = {}  # vtkActor
        # self.smootherFilters        = {} # vtkSmoothPolyDataFilter
        # self.polyDataNormals        = {} # vtkPolyDataNormals
        # self.typeExtractors         = {} # vtkDiscreteMarchingCubes
        # self.typeExtractorMappers   = {} # vtkPolyDataMapper

        self.actors_dict['cellsActor'] = {}

    def getActors(self, actor_label_list=None):
        """
        returns container with actors
        :param actor_label_list:{list of str} list of actors
        :return: {OrderedDict}
        """

        od = OrderedDict()
        if actor_label_list is None:
            return od
        for actor_label in actor_label_list:
            od[actor_label] = self.actors_dict[actor_label]

        return od

    def prepare_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares cell_field_actors  based on actor_specs specifications
        Scans list of invisible cell types and used cell types and creates those actors that user selected to be visible
        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """
        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        metadata = actor_specs_copy.metadata

        invisible_types = metadata['invisible_types']
        all_types = metadata['all_types']

        # todo - optimize creation of actors when using glyphs - we only need one actor when using glyphs
        # scr_data = drawing_params.screenshot_data
        # if scr_data.cell_glyphs_on:
        #     actor_specs_copy.actors_dict[0] = vtk.vtkActor()
        # else:

        for actorNumber in all_types:
            if not actorNumber in invisible_types:
                actor_specs_copy.actors_dict[actorNumber] = vtk.vtkActor()

        return actor_specs_copy

    def prepare_border_actors(self, actor_specs, drawing_params=None):
        """
        Prepares border actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_border_actors.__name__)

    def prepare_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares concentration field actors
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.conActor = vtk.vtkActor()
        self.contourActor = vtk.vtkActor()
        self.legendActor = vtk.vtkScalarBarActor()
        self.min_max_text_actor = vtk.vtkTextActor()

        actor_specs_copy.actors_dict['concentration_actor'] = self.conActor
        actor_specs_copy.actors_dict['legend_actor'] = self.legendActor
        actor_specs_copy.actors_dict['min_max_text_actor'] = self.min_max_text_actor

        return actor_specs_copy

    def show_concentration_field_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows concentration actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """

        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        if show_flag:
            self.add_actor_to_renderer(actor_label='concentration_actor', actor_obj=self.conActor)

            if mdata.get('LegendEnable', default=False):
                self.add_actor_to_renderer(actor_label='legend_actor', actor_obj=self.legendActor)

            if mdata.get('DisplayMinMaxInfo', default=True):
                self.add_actor_to_renderer(actor_label='min_max_text_actor', actor_obj=self.min_max_text_actor)

        else:
            self.remove_actor_from_renderer(actor_label='concentration_actor', actor_obj=self.conActor)
            self.remove_actor_from_renderer(actor_label='legend_actor', actor_obj=self.legendActor)

    def prepare_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares vector field actors
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        actor_specs_copy.actors_dict['vector_field_actor'] = self.glyphsActor
        actor_specs_copy.actors_dict['min_max_text_actor'] = self.min_max_text_actor

        return actor_specs_copy

    def show_vector_field_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows vector field actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)

        if show_flag:
            self.add_actor_to_renderer(actor_label='vector_field_actor', actor_obj=self.glyphsActor)

            if mdata.get('DisplayMinMaxInfo', default=True):
                self.add_actor_to_renderer(actor_label='min_max_text_actor', actor_obj=self.min_max_text_actor)

        else:
            self.remove_actor_from_renderer(actor_label='vector_field_actor', actor_obj=self.glyphsActor)
            self.remove_actor_from_renderer(actor_label='min_max_text_actor', actor_obj=self.min_max_text_actor)

    def prepare_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        Prepares fpp links actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """
        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        actor_specs_copy.actors_dict['fpp_links_actor'] = self.FPPLinksActor
        return actor_specs_copy

    def show_fpp_links_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows fpp links actors
        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        scene_metadata = drawing_params.screenshot_data.metadata
        mdata = MetadataHandler(mdata=scene_metadata)
        if show_flag:
            self.add_actor_to_renderer(actor_label='fpp_links_actor', actor_obj=self.FPPLinksActor)
        else:
            self.remove_actor_from_renderer(actor_label='fpp_links_actor', actor_obj=self.FPPLinksActor)


    def show_cell_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides cells
        :param show_flag:
        :return:
        """
        invisible_types = actor_specs.metadata['invisible_types']
        all_types = actor_specs.metadata['all_types']

        if show_flag:

            for actor_number in all_types:
                actor_name = "CellType_" + str(actor_number)
                if actor_number not in invisible_types:
                    self.currentActors[actor_name] = actor_specs.actors_dict[actor_number]
                    self.ren.AddActor(self.currentActors[actor_name])

        else:
            for actor_number in all_types:
                actor_name = "CellType_" + str(actor_number)
                if actor_name in self.currentActors:
                    actor_to_remove = self.currentActors[actor_name]
                    del self.currentActors[actor_name]
                    self.ren.RemoveActor(actor_to_remove)

    def prepare_outline_actors(self, actor_specs, drawing_params=None):
        """
        Prepares cell_field_actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.outlineActor = vtk.vtkActor()
        actor_specs_copy.actors_dict['outline_actor'] = self.outlineActor

        return actor_specs_copy

    def show_outline_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides bounding box
        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return::
        """
        if show_flag:
            if "Outline" not in self.currentActors:
                self.currentActors["Outline"] = self.outlineActor
                self.ren.AddActor(self.outlineActor)
            else:
                self.ren.RemoveActor(self.outlineActor)

                self.ren.AddActor(self.outlineActor)
        else:
            if "Outline" in self.currentActors:
                del self.currentActors["Outline"]
                self.ren.RemoveActor(self.outlineActor)

    def prepare_axes_actors(self, actor_specs, drawing_params=None):
        """
        Prepares cell_field_actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.axesActor = vtk.vtkCubeAxesActor2D()
        actor_specs_copy.actors_dict['axes_actor'] = self.axesActor

        return actor_specs_copy

    def show_axes_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides axes box
        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return:
        """
        camera = actor_specs.metadata['camera']
        if show_flag:
            if "Axes" not in self.currentActors:
                self.currentActors["Axes"] = self.axesActor
                # setting camera for the actor is very important to get axes working properly
                self.axesActor.SetCamera(camera)
                self.ren.AddActor(self.axesActor)
            else:
                self.ren.RemoveActor(self.axesActor)
                # setting camera for the actor is very important to get axes working properly
                self.axesActor.SetCamera(camera)
                self.ren.AddActor(self.axesActor)
        else:
            if "Axes" in self.currentActors:
                del self.currentActors["Axes"]
                self.ren.RemoveActor(self.axesActor)

    def getPlane(self):
        return ("3D", 0)

    def setPlotData(self, _plotData):
        self.currentFieldType = _plotData

    def set3DInvisibleTypes(self):
        '''
        Initializes a dictionary self.invisibleCellTypes of invisible cell types - reads settings "Types3DInvisible"
        :return:None
        '''
        from cc3d.CompuCellSetup import persistent_globals
        configuration = persistent_globals.configuration

        self.colorMap = configuration.getSetting("TypeColorMap")

        typesInvisibleStrTmp = str(configuration.getSetting("Types3DInvisible"))
        # print "GOT ",typesInvisibleStrTmp
        if typesInvisibleStrTmp != self.typesInvisibleStr:
            self.typesInvisibleStr = str(configuration.getSetting("Types3DInvisible"))

            typesInvisible = self.typesInvisibleStr.replace(" ", "")

            typesInvisible = typesInvisible.split(",")

            def cell_type_check(cell_type):
                try:
                    cell_type_int = int(cell_type)
                    if cell_type_int >= 0:

                        return True
                    else:
                        return False
                except:
                    False

            typesInvisible = [int(cell_type) for cell_type in typesInvisible if cell_type_check(cell_type)]
            # print "typesInvisibleVec=",typesInvisibleVec
            # turning list into a dictionary
            self.invisibleCellTypes.clear()
            for type in typesInvisible:
                self.invisibleCellTypes[int(type)] = 0
                # print "\t\t\t self.invisibleCellTypes=",self.invisibleCellTypes

    def set_default_camera(self, fieldDim=None):
        '''
        Initializes default camera view for 3D scene
        :param fieldDim:field dimension (Dim3D C++ object)
        :return: None
        '''
        camera = self.ren.GetActiveCamera()

        self.setDim(fieldDim)
        # Should I specify these parameters explicitly?
        # What if I change dimensions in XML file?
        # The parameters should be set based on the configuration parameters!
        # Should it set depending on projection? (e.g. xy, xz, yz)
        distance = self.largestDim(self.dim) * 2  # 200 #273.205 #
        # FIXME: Hardcoded numbers
        camera.SetPosition(self.dim[0] / 2, self.dim[1] / 2, distance)
        camera.SetFocalPoint(self.dim[0] / 2, self.dim[1] / 2, 0)
        camera.SetClippingRange(distance - 1, distance + 1)
        self.__initDist = distance  # camera.GetDistance()

    def setDim(self, fieldDim):
        '''
        assigns field dimensions (x,y,z) to a vector self.dim
        :param fieldDim: field dimension - instance of Dim3D (CC3D ++ object)
        :return: None
        '''
        # self.dim = [fieldDim.x+1 , fieldDim.y+1 , fieldDim.z]
        self.dim = [fieldDim.x, fieldDim.y, fieldDim.z]

