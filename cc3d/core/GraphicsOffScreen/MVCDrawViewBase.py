from weakref import ref
import vtk
import math
from cc3d.core.GraphicsOffScreen.DrawingParameters import DrawingParameters

MODULENAME = '----- MVCDrawViewBase.py: '

XZ_Z_SCALE = math.sqrt(6.0) / 3.0
YZ_Y_SCALE = math.sqrt(3.0) / 2.0
YZ_Z_SCALE = math.sqrt(6.0) / 3.0

VTK_MAJOR_VERSION = vtk.vtkVersion.GetVTKMajorVersion()
VTK_MINOR_VERSION = vtk.vtkVersion.GetVTKMinorVersion()
VTK_BUILD_VERSION = vtk.vtkVersion.GetVTKBuildVersion()


class MVCDrawViewBase:
    def __init__(self, _drawModel, ren=None):
        self.legendActor = vtk.vtkScalarBarActor()
        self.legendActor.SetNumberOfLabels(8)
        (self.minCon, self.maxCon) = (0, 0)
        self.plane = 'XY'
        self.planePos = 0
        self.ren = None

        self.drawModel = _drawModel

        self.currentDrawingFunction = None
        self.currentActors = {}  # dictionary of current actors
        self.drawingFcnName = ""  # holds a string describing name of the drawing fcn . Used to determine if current actors need to be removed before next drawing
        self.drawingFcnHasChanged = True
        self.fieldTypes = None
        self.currentDrawingParameters = DrawingParameters()
        # self.currentFieldType = ("Cell_Field", FIELD_TYPES[0])
        self.currentFieldType = ("Cell_Field", 'CellField')
        self.__initDist = 0  # initial camera distance - used in zoom functions
        self.min_max_text_actor = vtk.vtkTextActor()

        # CUSTOM ACTORS
        self.customActors = {}  # {visName: CustomActorsStorage() }
        self.currentCustomVisName = ''  # stores name of the current custom visualization
        self.currentVisName = ''  # stores name of the current visualization
        self.cameraSettingsDict = {}  # {fieldName:CameraSettings()}
        self.ren = ren

    @property
    def drawModel(self):
        return self._drawModel()

    @drawModel.setter
    def drawModel(self, _i):
        self._drawModel = ref(_i)

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
            self._ren = ref(_i)
        except TypeError:
            self._ren = _i

    def version_identifier(self, major, minor, build):
        return major * 10 ** 6 + minor * 10 ** 3 + build

    def vtk_version_identifier(self):
        return self.version_identifier(VTK_MAJOR_VERSION, VTK_MINOR_VERSION, VTK_BUILD_VERSION)

    def setDrawingFunctionName(self, _fcnName):
        # print "\n\n\n THIS IS _fcnName=",_fcnName," self.drawingFcnName=",self.drawingFcnName

        if self.drawingFcnName != _fcnName:
            self.drawingFcnHasChanged = True
        else:
            self.drawingFcnHasChanged = False
        self.drawingFcnName = _fcnName

    def clear_scene(self):
        """
        removes all actors from the renderer
        :return: None
        """
        for actor in self.currentActors:
            self.ren.RemoveActor(self.currentActors[actor])
        self.currentActors.clear()

    def setFieldTypes(self, _fieldTypes):
        self.fieldTypes = _fieldTypes

    def setPlotData(self, _plotData):
        self.currentFieldType = _plotData

    def resetAllCameras(self):
        self.cameraSettingsDict = {}

    def getActors(self, actor_label_list=None):
        """
        returns container with actors

        :param actor_label_list:{list of str} list of actors
        :return: {OrderedDict} 
        """
        raise NotImplementedError(self.__class__.getActors.__name__)

    def prepare_vector_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares vector field actors

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """

        raise NotImplementedError(self.__class__.prepare_vector_field_actors.__name__)

    def show_vector_field_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows vector field actors

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        raise NotImplementedError(self.__class__.show_vector_field_actors.__name__)

    def prepare_concentration_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares concentration field actors

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {ActorSpecs}
        """

        raise NotImplementedError(self.__class__.prepare_concentration_field_actors.__name__)

    def show_concentration_field_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows concentration actors

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """

        raise NotImplementedError(self.__class__.show_concentration_field_actors.__name__)

    def prepare_cell_field_actors(self, actor_specs, drawing_params=None):
        """
        Prepares cell_field_actors  based on actor_specs specifications

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_cell_field_actors.__name__)

    def show_cell_actors(self,  actor_specs, drawing_params=None, show_flag=True):
        """
        Shows cell_field actors

        :param actor_specs: {ActorSpecs}
        :param show_flag: {bool}
        :return: None
        """
        raise NotImplementedError(self.__class__.show_cell_actors.__name__)

    def prepare_outline_actors(self, actor_specs, drawing_params=None):
        """
        Prepares bounding box actors  based on actor_specs specifications

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_outline_actors.__name__)

    def show_outline_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides bounding box

        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return:
        """
        raise NotImplementedError(self.__class__.show_outline_actors.__name__)

    def prepare_axes_actors(self, actor_specs, drawing_params=None):
        """
        Prepares axes  based on actor_specs specifications

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_axes_actors.__name__)

    def show_axes_actors(self,  actor_specs, drawing_params=None, show_flag=True):
        """
        shows/hides axes box

        :param actor_specs:
        :param drawing_params:
        :param show_flag:
        :return:
        """
        raise NotImplementedError(self.__class__.show_axes_actors.__name__)


    def prepare_cluster_border_actors(self, actor_specs, drawing_params=None):
        """
        Prepares border actors  based on actor_specs specifications

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_cluster_border_actors.__name__)

    def show_cluster_border_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows cluster borders actors

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        raise NotImplementedError(self.__class__.show_cluster_border_actors.__name__)

    def prepare_border_actors(self, actor_specs, drawing_params=None):
        """
        Prepares border actors  based on actor_specs specifications

        :param actor_specs: {ActorSpecs} specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_border_actors.__name__)

    def show_cell_borders(self, actor_specs,drawing_params=None, show_flag=True):
        """
        Shows or hides cell border actor

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """

        raise NotImplementedError(self.__class__.show_cell_borders.__name__)

    def prepare_fpp_links_actors(self, actor_specs, drawing_params=None):
        """
        Prepares fpp links actors  based on actor_specs specifications

        :param actor_specs {ActorSpecs}: specification of actors to create
        :param drawing_params: {DrawingParameters}
        :return: {dict}
        """

        raise NotImplementedError(self.__class__.prepare_fpp_links_actors.__name__)

    def show_fpp_links_actors(self, actor_specs, drawing_params=None, show_flag=True):
        """
        Shows fpp links actors

        :param actor_specs: {ActorSpecs}
        :param drawing_params: {DrawingParameters}
        :param show_flag: {bool}
        :return: None
        """
        raise NotImplementedError(self.__class__.show_fpp_links_actors.__name__)

    def remove_all_actors_from_renderer(self):
        """
        Removes all actors from renderer

        :return:
        """

        for actor_label, actor_obj in list(self.currentActors.items()):
            self.remove_actor_from_renderer(actor_label=actor_label, actor_obj=actor_obj)

    def add_actor_to_renderer(self, actor_label, actor_obj):
        """
        Convenience fcn that adds actor to a renderer and updates
        current actor container

        :param actor_label:  {str} actor label
        :param actor_obj: {vtkActor} instance of the actor
        :return: None
        """
        if actor_label not in self.currentActors:
            self.currentActors[actor_label] = actor_obj
            self.ren.AddActor(actor_obj)

    def remove_actor_from_renderer(self, actor_label, actor_obj):
        """
        Convenience fcn that removes actor from a renderer and updates
        current actor container

        :param actor_label:  {str} actor label
        :param actor_obj: {vtkActor} instance of the actor
        :return: None
        """
        if actor_label in self.currentActors:
            del self.currentActors[actor_label]
            self.ren.RemoveActor(actor_obj)

    def getCamera(self):
        return self.ren.GetActiveCamera()

    def largestDim(self, dim):
        ldim = dim[0]
        for i in range(len(dim)):
            if dim[i] > ldim:
                ldim = dim[i]

        return ldim

    def getSim3DFlag(self):
        zdim = self.currentDrawingParameters.bsd.fieldDim.z
        #        print MODULENAME,'  getSim3DFlag, zdim=',zdim
        if zdim > 1:
            return True
        else:
            return False

    def setParams(self):
        pass

    def set_custom_camera(self, camera_settings):
        """
        adjusts camera based on camera_settings.
        camera settings can be any structure that has the following members:
        clippingRange,focalPoint, viewUp, position

        :param camera_settings: {object} object conforming to above requirements
        :return: None
        """
        camera = self.ren.GetActiveCamera()
        cs = camera_settings
        if cs.clippingRange and cs.focalPoint and cs.position and cs.viewUp:
            camera.SetClippingRange(cs.clippingRange)
            camera.SetFocalPoint(cs.focalPoint)
            camera.SetPosition(cs.position)
            camera.SetViewUp(cs.viewUp)

    def set_default_camera(self, fieldDim=None):
        """
        Initializes default camera view

        :param fieldDim: field dimension (Dim3D C++ object)
        :return: None
        """
        raise NotImplementedError()

    def setStatusBar(self, statusBar):
        self._statusBar = statusBar
