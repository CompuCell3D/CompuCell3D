from .MVCDrawViewBase import MVCDrawViewBase
import vtk
import string
from collections import OrderedDict
from copy import deepcopy
from cc3d.core.GraphicsOffScreen.MetadataHandler import MetadataHandler


MODULENAME = "----MVCDrawView2D.py: "

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()
VTK_MINOR_VERSION = vtk.vtkVersion.GetVTKMinorVersion()


class MVCDrawView2D(MVCDrawViewBase):
    def __init__(self, _drawModel, ren=None):
        MVCDrawViewBase.__init__(self, _drawModel, ren=ren)

        self.initArea()
        self.setParams()

        # self.pixelizedScalarField = Configuration.getSetting("PixelizedScalarField")

    def initArea(self):
        """
        Sets up the VTK simulation area
        :return:None
        """

        self.actors_dict = {}

        self.actorCollection = vtk.vtkActorCollection()
        self.borderActor = vtk.vtkActor()
        self.borderActorHex = vtk.vtkActor()
        self.clusterBorderActor = vtk.vtkActor()
        self.clusterBorderActorHex = vtk.vtkActor()
        self.cellGlyphsActor = vtk.vtkActor()
        self.FPPLinksActor = vtk.vtkActor()  # used for both white and colored links
        self.outlineActor = vtk.vtkActor()
        # self.axesActor = vtk.vtkCubeAxesActor2D()
        self.axesActor = vtk.vtkCubeAxesActor()

        self.outlineDim = [0, 0, 0]

        self.cellsActor = vtk.vtkActor()
        self.cellsActor.GetProperty().SetInterpolationToFlat()  # ensures that pixels are drawn exactly not with interpolations/antialiasing

        self.hexCellsActor = vtk.vtkActor()
        self.hexCellsActor.GetProperty().SetInterpolationToFlat()  # ensures that pixels are drawn exactly not with interpolations/antialiasing

        self.conActor = vtk.vtkActor()
        self.conActor.GetProperty().SetInterpolationToFlat()

        self.con_actor_glyphs = vtk.vtkActor()
        self.con_actor_glyphs.GetProperty().SetInterpolationToFlat()

        self.hexConActor = vtk.vtkActor()
        self.hexConActor.GetProperty().SetInterpolationToFlat()

        self.contourActor = vtk.vtkActor()

        self.glyphsActor = vtk.vtkActor()
        # self.linksActor=vtk.vtkActor()

        # # Concentration lookup table

        self.clut = vtk.vtkLookupTable()
        self.clut.SetHueRange(0.67, 0.0)
        self.clut.SetSaturationRange(1.0, 1.0)
        self.clut.SetValueRange(1.0, 1.0)
        self.clut.SetAlphaRange(1.0, 1.0)
        self.clut.SetNumberOfColors(1024)
        self.clut.Build()

        # Contour lookup table
        # Do I need lookup table? May be just one color?
        self.ctlut = vtk.vtkLookupTable()
        self.ctlut.SetHueRange(0.6, 0.6)
        self.ctlut.SetSaturationRange(0, 1.0)
        self.ctlut.SetValueRange(1.0, 1.0)
        self.ctlut.SetAlphaRange(1.0, 1.0)
        self.ctlut.SetNumberOfColors(1024)
        self.ctlut.Build()

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
            od[actor_label] = getattr(self, actor_label)

        return od

    def prepare_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares vector field actors
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        actor_specs_copy.actors_dict["vector_field_actor"] = self.glyphsActor
        actor_specs_copy.actors_dict["min_max_text_actor"] = self.min_max_text_actor

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
            self.add_actor_to_renderer(actor_label="vector_field_actor", actor_obj=self.glyphsActor)

            if mdata.get("DisplayMinMaxInfo", default=True):
                self.add_actor_to_renderer(actor_label="min_max_text_actor", actor_obj=self.min_max_text_actor)

        else:
            self.remove_actor_from_renderer(actor_label="vector_field_actor", actor_obj=self.glyphsActor)
            self.remove_actor_from_renderer(actor_label="min_max_text_actor", actor_obj=self.min_max_text_actor)

    def prepare_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares concentration field actors
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """
        scr_data = drawing_params.screenshot_data
        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()

        # it is best to create actors on demand rather than reuse "global" actors stored in the class instance
        # this si because one drawing mode may scale actors and another one may not scale actors and this different
        # set of operations on actors leads to strange side effects
        self.conActor = vtk.vtkActor()
        self.contourActor = vtk.vtkActor()
        self.legendActor = vtk.vtkScalarBarActor()
        self.min_max_text_actor = vtk.vtkTextActor()

        actor_specs_copy.actors_dict["concentration_actor"] = self.conActor
        actor_specs_copy.actors_dict["contour_actor"] = self.contourActor
        actor_specs_copy.actors_dict["legend_actor"] = self.legendActor
        actor_specs_copy.actors_dict["min_max_text_actor"] = self.min_max_text_actor

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
            self.add_actor_to_renderer(actor_label="concentration_actor", actor_obj=self.conActor)

            if mdata.get("ContoursOn", default=False):
                self.add_actor_to_renderer(actor_label="contour_actor", actor_obj=self.contourActor)

            if mdata.get("LegendEnable", default=False):
                self.add_actor_to_renderer(actor_label="legend_actor", actor_obj=self.legendActor)

            if mdata.get("DisplayMinMaxInfo", default=True):
                self.add_actor_to_renderer(actor_label="min_max_text_actor", actor_obj=self.min_max_text_actor)

        else:
            self.remove_actor_from_renderer(actor_label="concentration_actor", actor_obj=self.conActor)
            self.remove_actor_from_renderer(actor_label="contour_actor", actor_obj=self.contourActor)
            self.remove_actor_from_renderer(actor_label="legend_actor", actor_obj=self.legendActor)
            self.remove_actor_from_renderer(actor_label="min_max_text_actor", actor_obj=self.min_max_text_actor)

    def prepare_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares cell_field_actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :return: {ActorSpecs}
        """
        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.cellsActor = vtk.vtkActor()
        actor_specs_copy.actors_dict["cellsActor"] = self.cellsActor

        return actor_specs_copy

    def show_cell_borders(self, actor_specs, drawing_params=None, show_flag=True):
        '''
        Shows or hides cell border actor

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        '''
        if show_flag:
            self.add_actor_to_renderer(actor_label="border_actor", actor_obj=self.borderActor)
        else:
            self.remove_actor_from_renderer(actor_label="border_actor", actor_obj=self.borderActor)

    def prepare_border_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Prepares border actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :return: {dict}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.borderActor = vtk.vtkActor()

        actor_specs_copy.actors_dict["border_actor"] = self.borderActor
        return actor_specs_copy

    def prepare_cluster_border_actors(self, actor_specs):
        """
        Prepares cluster border actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :return: {dict}
        """

        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.clusterBorderActor = vtk.vtkActor()
        actor_specs_copy.actors_dict["cluster_border_actor"] = self.clusterBorderActor
        return actor_specs_copy

    def show_cluster_border_actors(self, actor_specs, drawing_params=None, show_flag=True):
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
            self.add_actor_to_renderer(actor_label="cluster_border_actor", actor_obj=self.clusterBorderActor)
        else:
            self.remove_actor_from_renderer(actor_label="cluster_border_actor", actor_obj=self.clusterBorderActor)

    def prepare_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        Prepares fpp links actors  based on actor_specs specifications
        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """
        actor_specs_copy = deepcopy(actor_specs)
        actor_specs_copy.actors_dict = OrderedDict()
        self.FPPLinksActor = vtk.vtkActor()
        actor_specs_copy.actors_dict["fpp_links_actor"] = self.FPPLinksActor
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
            self.add_actor_to_renderer(actor_label="fpp_links_actor", actor_obj=self.FPPLinksActor)
        else:
            self.remove_actor_from_renderer(actor_label="fpp_links_actor", actor_obj=self.FPPLinksActor)

    def show_cell_actors(self, actor_specs, show_flag=True):
        """
        shows/hides cells
        :param show_flag:
        :return:
        """
        if show_flag:
            if "CellsActor" not in self.currentActors:
                self.currentActors["CellsActor"] = self.cellsActor
                self.ren.AddActor(self.cellsActor)
        else:
            if "CellsActor" in self.currentActors:
                del self.currentActors["CellsActor"]
                self.ren.RemoveActor(self.cellsActor)

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
        actor_specs_copy.actors_dict["outline_actor"] = self.outlineActor

        return actor_specs_copy

    def show_outline_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides bounding box
        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return:
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
        self.axesActor = vtk.vtkCubeAxesActor()
        actor_specs_copy.actors_dict["axes_actor"] = self.axesActor
        # actor_specs_copy.actors_dict['axes_actor'] = self.outlineActor

        return actor_specs_copy

    def show_axes_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides axes box
        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return:
        """
        camera = actor_specs.metadata["camera"]
        if show_flag:
            if "Axes" not in self.currentActors:
                # setting camera for the actor is very important to get axes working properly
                self.axesActor.SetCamera(camera)
                self.currentActors["Axes"] = self.axesActor
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

    def setPlane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)

    #        print MODULENAME,"  got this plane ",(self.plane, self.planePos)
    #        print (self.plane, self.planePos)

    def getPlane(self):
        return (self.plane, self.planePos)

    def set_default_camera(self, fieldDim=None):
        """
        Initializes default camera view for 2D scene
        :param fieldDim:field dimension (Dim3D C++ object)
        :return: None
        """

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
        self.ren.ResetCameraClippingRange()
        self.__initDist = distance  # camera.GetDistance()

    def setCamera(self, fieldDim=None):
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
        # self.qvtkWidget.ren.ResetCameraClippingRange()
        self.ren.ResetCameraClippingRange()
        self.__initDist = distance  # camera.GetDistance()
        # self.Render()
        # self.qvtkWidget().repaint()

    def setDim(self, fieldDim):
        self.dim = [fieldDim.x, fieldDim.y, fieldDim.z]


    # Optimize code?
    def dimOrder(self, plane):
        plane = string.lower(plane)
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
