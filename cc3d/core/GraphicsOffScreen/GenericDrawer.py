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

import cc3d.player5.Configuration as Configuration
import vtk
from os.path import splitext
from cc3d.core.enums import *
from copy import deepcopy
from .MVCDrawView2D import MVCDrawView2D
from .MVCDrawModel2D import MVCDrawModel2D
from .MVCDrawView3D import MVCDrawView3D
from .MVCDrawModel3D import MVCDrawModel3D
from .Specs import ActorSpecs
from cc3d.player5.Utilities.utils import extract_address_int_from_vtk_object
from .DrawingParameters import DrawingParameters

MODULENAME = '---- GraphicsFrameWidget.py: '


class GenericDrawer():
    def __init__(self, parent=None, originatingWidget=None):

        self.plane = None
        self.planePos = None
        self.field_extractor = None

        self.ren_2D = vtk.vtkRenderer()
        # self.ren_3D = vtk.vtkRenderer()
        self.interactive_camera_flag = False

        # self.renWin = self.qvtkWidget.GetRenderWindow()
        # self.renWin.AddRenderer(self.ren)

        # MDIFIX
        self.draw_model_2D = MVCDrawModel2D()
        self.draw_view_2D = MVCDrawView2D(self.draw_model_2D)

        self.draw_model_3D = MVCDrawModel3D()
        self.draw_view_3D = MVCDrawView3D(self.draw_model_3D)

        self.draw_view_2D.ren = self.ren_2D
        self.draw_view_3D.ren = self.ren_2D
        # self.draw_view_3D.ren = self.ren_3D

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

        # # placeholder for recently used basic simulation data
        # self.recent_bsd = None
        #
        # # recent screenshot description data
        # self.recent_screenshot_data = None

    def set_pixelized_cartesian_scene(self, flag: bool) -> None:
        """
        Enables pixelized cartesian scene
        :param flag:
        :return:
        """

        self.draw_model_2D.pixelized_cartesian_field = flag

    def configsChanged(self):
        """
        Function called by the GraphicsFrameWidget after configuration dialog has been approved
        We update information from configuration dialog here
        :return: None
        """
        self.draw_model_2D.populate_cell_type_lookup_table()
        self.draw_model_3D.populate_cell_type_lookup_table()
        # if self.recent_screenshot_data is not None and self.recent_bsd is not None:
        #     self.draw(screenshot_data=self.recent_screenshot_data,bsd=self.recent_bsd)

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

    def extract_cell_field_data(self):
        """
        Extracts basic information about cell field
        :return:
        """
        cellType = vtk.vtkIntArray()
        cellType.SetName("celltype")
        cellTypeIntAddr = extract_address_int_from_vtk_object(cellType)
        # Also get the CellId
        cellId = vtk.vtkLongArray()
        cellId.SetName("cellid")
        cellIdIntAddr = extract_address_int_from_vtk_object(cellId)

        usedCellTypesList = self.field_extractor.fillCellFieldData3D(cellTypeIntAddr, cellIdIntAddr)

        ret_val = {
            'cell_type_array': cellType,
            'cell_id_array': cellId,
            'used_cell_types': usedCellTypesList
        }
        return ret_val

    def set_field_extractor(self, field_extractor):

        self.field_extractor = field_extractor
        self.draw_model_2D.field_extractor = field_extractor
        self.draw_model_3D.field_extractor = field_extractor

    def draw_vector_field(self, drawing_params):
        """
        Draws  vector field
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
        :return:None
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

        # actors_dict = view.getActors(actor_label_list=['cellsActor'])
        actor_specs_final = view.prepare_cell_field_actors(actor_specs)
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
        # show_flag = drawing_params.screenshot_data.cell_borders_on
        view.show_cell_borders(actor_specs=actor_specs_final)
        # view.show_cell_borders(show_flag=show_flag)

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
        # type: (DrawingParameters) -> (MVCDrawModelBase,MVCDrawViewBase)
        """
        returns pair of model view objects depending on the dimension label
        :param drawing_params: {Graphics.DrawingParameters instance}
        :return: {tuple} mode, view object tuple
        """
        dimension_label = drawing_params.screenshot_data.spaceDimension

        if dimension_label == '2D':
            return self.draw_model_2D, self.draw_view_2D
        else:
            return self.draw_model_3D, self.draw_view_3D

    def draw(self, screenshot_data, bsd, screenshot_name=None):
        # todo 5 - fix this so that it is called only once per drawing series
        # return
        # self.recent_screenshot_data = screenshot_data
        # self.recent_bsd = bsd

        cell_field_data_dict = self.extract_cell_field_data()
        bsd.cell_types_used = deepcopy(cell_field_data_dict['used_cell_types'])

        drawing_params = DrawingParameters()
        drawing_params.screenshot_data = screenshot_data
        drawing_params.bsd = bsd
        drawing_params.plane = screenshot_data.projection
        drawing_params.planePosition = screenshot_data.projectionPosition
        drawing_params.planePos = screenshot_data.projectionPosition
        drawing_params.fieldName = screenshot_data.plotData[0]  # e.g. plotData = ('Cell_Field','CellField')
        drawing_params.fieldType = screenshot_data.plotData[1]

        # model, view = self.get_model_view(dimension_label=screenshot_data.spaceDimension)
        model, view = self.get_model_view(drawing_params=drawing_params)
        # passes information about cell lattice
        model.set_cell_field_data(cell_field_data_dict=cell_field_data_dict)

        ren = view.ren
        # self.draw_model_2D.setDrawingParametersObject(drawing_params)
        model.setDrawingParametersObject(drawing_params)

        try:
            key = drawing_params.fieldType
            draw_fcn = self.drawing_fcn_dict[key]
        except KeyError:
            if str(key).strip() != '':
                print('Could not find function for {}'.format(key))
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

            # setting camera
            # if screenshot_data.spaceDimension == '3D':
            #     if screenshot_data.clippingRange is not None:
            #         view.set_custom_camera(camera_settings=screenshot_data)
            # else:
            #     view.set_default_camera(drawing_params.bsd.fieldDim)

            # we allow resetting of camera properties only in the non-interactive mode
            # in the interactive mode camera is managed by the GUI
            if not self.interactive_camera_flag:
                if screenshot_data.clippingRange is not None:
                    view.set_custom_camera(camera_settings=screenshot_data)

            # self.output_screenshot(ren=ren, screenshot_fname='D:/CC3D_GIT/CompuCell3D/player5/GraphicsOffScreen/{screenshot_name}.png'.format(
            #     screenshot_name=screenshot_name))

            # renWin = vtk.vtkRenderWindow()
            # renWin.SetOffScreenRendering(1)
            # renWin.AddRenderer(ren)
            # renWin.Render()
            #
            # windowToImageFilter = vtk.vtkWindowToImageFilter()
            # windowToImageFilter.SetInput(renWin)
            # windowToImageFilter.Update()
            #
            # writer = vtk.vtkPNGWriter()
            # writer.SetFileName('D:/CC3D_GIT/CompuCell3D/player5/GraphicsOffScreen/{screenshot_name}.png'.format(
            #     screenshot_name=screenshot_name))
            # writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            # writer.Write()

    def output_screenshot(self, screenshot_fname, file_format='png', screenshot_data=None):
        """
        Saves scene rendered in the renderer to the image
        :param ren: {vtkRenderer} renderer
        :param screenshot_fname: {str} screenshot filename
        :return: None
        """

        ren = self.get_renderer()
        ren_win = vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(1)

        if screenshot_data is not None:
            ren_win.SetSize(screenshot_data.win_width, screenshot_data.win_height)

        ren_win.AddRenderer(ren)
        ren_win.Render()

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(ren_win)
        windowToImageFilter.Update()

        if file_format.lower() == 'png':
            writer = vtk.vtkPNGWriter()
        elif file_format.lower() == 'tiff':
            writer = vtk.vtkTIFFWriter()
            screenshot_fname = splitext(screenshot_fname)[0] + '.tif'
        else:
            writer = vtk.vtkPNGWriter()

        writer.SetFileName(screenshot_fname)

        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    def draw_old(self, screenshot_data, bsd, screenshot_name):
        # drawing_params = DrawingParameters()
        # bsd = BasicSimulationData()
        # drawing_params.bsd = bsd
        #
        # drawing_params.plane = 'XY'
        # drawing_params.planePos = 0
        # drawing_params.fieldName = 'CellField'
        # drawing_params.fieldType = 'CellField'
        # self.drawModel2D.setDrawingParametersObject(drawing_params)

        # self.ren = vtk.vtkRenderer()
        # renWin = vtk.vtkRenderWindow()
        # renWin.SetOffScreenRendering(1)
        # renWin.AddRenderer(self.ren)
        # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # renderWindowInteractor.SetRenderWindow(renderWindow)

        drawing_params = DrawingParameters()
        drawing_params.screenshot_data = screenshot_data
        drawing_params.bsd = bsd
        drawing_params.plane = screenshot_data.projection
        drawing_params.planePosition = screenshot_data.projectionPosition
        drawing_params.planePos = screenshot_data.projectionPosition
        drawing_params.fieldName = screenshot_data.plotData[0]  # e.g. plotData = ('Cell_Field','CellField')
        drawing_params.fieldType = screenshot_data.plotData[0]

        model, view = self.get_model_view(screenshot_data.spaceDimension)

        model.setDrawingParametersObject(drawing_params)
        view.drawCellFieldLocalNew(drawing_params, None)

        # self.draw_model_2D.setDrawingParametersObject(drawing_params)
        #
        # self.draw_view_2D.drawCellFieldLocalNew(drawing_params, None)
        # # self.draw2D.drawCellFieldLocalNew_1(self.ren)

        # coneSource = vtk.vtkConeSource()
        # coneSource.SetResolution(60)
        # coneSource.SetCenter(-2, 0, 0)
        # # Create a mapper and actor
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(coneSource.GetOutputPort())
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        #
        # # Visualize
        # renderer = vtk.vtkRenderer()
        # renWin = vtk.vtkRenderWindow()
        # renWin.SetOffScreenRendering(1)
        # renWin.AddRenderer(renderer)
        # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # renderWindowInteractor.SetRenderWindow(renderWindow)

        # # OK
        # coneSource = vtk.vtkConeSource()
        # coneSource.SetResolution(60)
        # coneSource.SetCenter(-2, 0, 0)
        #
        # # Create a mapper and actor
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(coneSource.GetOutputPort())
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        #
        # # Visualize
        # renderer = vtk.vtkRenderer()
        # renWin = vtk.vtkRenderWindow()
        # renWin.SetOffScreenRendering(1)
        # renWin.AddRenderer(renderer)
        # # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # # renderWindowInteractor.SetRenderWindow(renderWindow)
        #
        # renderer.AddActor(actor)
        # # renderer.SetBackground(.1, .2, .3)  # Background color dark blue
        # # renderer.SetBackground(.3, .2, .1)  # Background color dark red
        # renWin.Render()
        # # OK

        # coneSource = vtk.vtkConeSource()
        # coneSource.SetResolution(60)
        # coneSource.SetCenter(-2, 0, 0)
        #
        # # Create a mapper and actor
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(coneSource.GetOutputPort())
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        #
        # # Visualize
        # # renderer = vtk.vtkRenderer()
        # renderer = self.ren
        # renWin = vtk.vtkRenderWindow()
        # renWin.SetOffScreenRendering(1)
        # renWin.AddRenderer(renderer)
        # # renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        # # renderWindowInteractor.SetRenderWindow(renderWindow)
        #
        # renderer.AddActor(actor)
        # # renderer.SetBackground(.1, .2, .3)  # Background color dark blue
        # # renderer.SetBackground(.3, .2, .1)  # Background color dark red
        # renWin.Render()

        renWin = vtk.vtkRenderWindow()
        renWin.SetOffScreenRendering(1)
        renWin.AddRenderer(self.ren)
        renWin.Render()

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renWin)
        windowToImageFilter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName('D:/CC3D_GIT/CompuCell3D/player5/GraphicsOffScreen/{screenshot_name}.png'.format(
            screenshot_name=screenshot_name))
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    def resetAllCameras(self):
        print('resetAllCameras in GraphicsFrame =', self)

        self.draw_view_2D.resetAllCameras()
        self.draw3D.resetAllCameras()

    def populateLookupTable(self):
        self.draw_model_2D.populateLookupTable()
        self.drawModel3D.populateLookupTable()

    def Render(self):
        color = Configuration.getSetting("WindowColor")
        self.ren.SetBackground(float(color.red()) / 255, float(color.green()) / 255, float(color.blue()) / 255)
        self.qvtkWidget.Render()

    def getCamera(self):
        return self.getActiveCamera()

    def getActiveCamera(self):
        return self.ren.GetActiveCamera()

    def setActiveCamera(self, _camera):
        return self.ren.SetActiveCamera(_camera)

    def getCamera2D(self):
        return self.camera2D

    def setZoomItems(self, _zitems):
        self.draw_view_2D.setZoomItems(_zitems)
        self.draw3D.setZoomItems(_zitems)

    def setPlane(self, plane, pos):
        (self.plane, self.planePos) = (str(plane).upper(), pos)
        # print (self.plane, self.planePos)

    def getPlane(self):
        return (self.plane, self.planePos)

    def setDrawingStyle(self, _style):
        style = _style.upper()
        if style == "2D":
            self.draw3DFlag = False
            # self.currentDrawingObject = self.draw_view_2D
            # self.ren.SetActiveCamera(self.camera2D)
            self.qvtkWidget.setMouseInteractionSchemeTo2D()
            self.draw3D.clearDisplay()
        elif style == "3D":
            self.draw3DFlag = True
            # self.currentDrawingObject = self.draw3D
            # self.ren.SetActiveCamera(self.camera3D)
            self.qvtkWidget.setMouseInteractionSchemeTo3D()
            self.draw_view_2D.clearDisplay()

    # todo 5 - orig code
    # def setDrawingStyle(self, _style):
    #     style = string.upper(_style)
    #     if style == "2D":
    #         self.draw3DFlag = False
    #         self.currentDrawingObject = self.draw_view_2D
    #         self.ren.SetActiveCamera(self.camera2D)
    #         self.qvtkWidget.setMouseInteractionSchemeTo2D()
    #         self.draw3D.clearDisplay()
    #     elif style == "3D":
    #         self.draw3DFlag = True
    #         self.currentDrawingObject = self.draw3D
    #         self.ren.SetActiveCamera(self.camera3D)
    #         self.qvtkWidget.setMouseInteractionSchemeTo3D()
    #         self.draw_view_2D.clearDisplay()

    def getCamera3D(self):
        return self.camera3D

    def getActiveCamera(self):
        return self.ren.GetActiveCamera()

    def getCurrentSceneNameAndType(self):
        # this is usually field name but we can also allow other types of visualizations hence I am calling it getCurrerntSceneName
        sceneName = str(self.fieldComboBox.currentText())
        return sceneName, self.parentWidget.fieldTypes[sceneName]

    def apply3DGraphicsWindowData(self, gwd):

        for p in range(self.projComboBox.count()):

            if str(self.projComboBox.itemText(p)) == '3D':
                # camera = self.getActiveCamera()
                # print 'activeCamera=',activeCamera

                self.projComboBox.setCurrentIndex(p)

                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D
                self.camera3D.SetClippingRange(gwd.cameraClippingRange)
                self.camera3D.SetFocalPoint(gwd.cameraFocalPoint)
                self.camera3D.SetPosition(gwd.cameraPosition)
                self.camera3D.SetViewUp(gwd.cameraViewUp)

                break

    def apply2DGraphicsWindowData(self, gwd):

        for p in range(self.projComboBox.count()):

            if str(self.projComboBox.itemText(p)).lower() == str(gwd.planeName).lower():
                self.projComboBox.setCurrentIndex(p)
                # print 'self.projSpinBox.maximum()= ', self.projSpinBox.maximum()
                # if gwd.planePosition <= self.projSpinBox.maximum():
                self.projSpinBox.setValue(gwd.planePosition)  # automatically invokes the callback (--Changed)

                # notice: there are two cameras one for 2D and one for 3D  here we set camera for 3D
                self.camera2D.SetClippingRange(gwd.cameraClippingRange)
                self.camera2D.SetFocalPoint(gwd.cameraFocalPoint)
                self.camera2D.SetPosition(gwd.cameraPosition)
                self.camera2D.SetViewUp(gwd.cameraViewUp)

    def applyGraphicsWindowData(self, gwd):
        # print 'COMBO BOX CHECK '
        # for i in xrange(self.fieldComboBox.count()):
        # print 'self.fieldComboBox.itemText(i)=',self.fieldComboBox.itemText(i)

        for i in range(self.fieldComboBox.count()):

            if str(self.fieldComboBox.itemText(i)) == gwd.sceneName:

                self.fieldComboBox.setCurrentIndex(i)
                # setting 2D projection or 3D
                if gwd.is3D:
                    self.apply3DGraphicsWindowData(gwd)
                else:
                    self.apply2DGraphicsWindowData(gwd)

                break

        # import time
        # time.sleep(2)

    def _takeShot(self):
        #        print MODULENAME, '  _takeShot():  self.parentWidget.screenshotManager=',self.parentWidget.screenshotManager
        print(MODULENAME, '  _takeShot():  self.renWin.GetSize()=', self.renWin.GetSize())
        camera = self.getActiveCamera()
        # # # camera = self.ren.GetActiveCamera()
        #        print MODULENAME, '  _takeShot():  camera=',camera
        #        clippingRange= camera.GetClippingRange()
        #        focalPoint= camera.GetFocalPoint()
        #        position= camera.GetPosition()
        #        viewUp= camera.GetViewUp()
        #        viewAngle= camera.GetViewAngle()
        #        print MODULENAME,"_takeShot():  Range,FP,Pos,Up,Angle=",clippingRange,focalPoint,position,viewUp,viewAngle

        if self.parentWidget.screenshotManager is not None:
            name = str(self.fieldComboBox.currentText())
            self.parentWidget.fieldTypes[name]
            fieldType = (name, self.parentWidget.fieldTypes[name])
            print(MODULENAME, '  _takeShot():  fieldType=', fieldType)

            #            if self.threeDRB.isChecked():
            if self.draw3DFlag:
                self.parentWidget.screenshotManager.add_3d_screenshot(fieldType[0], fieldType[1], camera)
            else:
                planePositionTupple = self.draw_view_2D.getPlane()
                # print "planePositionTupple=",planePositionTupple
                self.parentWidget.screenshotManager.add_2d_screenshot(fieldType[0], fieldType[1], planePositionTupple[0],
                                                                      planePositionTupple[1], camera)

    def setInitialCrossSection(self, _basicSimulationData):
        #        print MODULENAME, '  setInitialCrossSection'
        fieldDim = _basicSimulationData.fieldDim

        self.updateCrossSection(_basicSimulationData)

        # new (rwh, May 2011)
        self.currentProjection = 'xy'  # rwh

        self.projComboBox.setCurrentIndex(1)  # set to be 'xy' projection by default, regardless of 2D or 3D sim?

        self.projSpinBox.setMinimum(0)
        #        self.projSpinBox.setMaximum(fieldDim.z - 1)
        self.projSpinBox.setMaximum(10000)
        #        if fieldDim.z/2 >= 1: # do this trick to avoid empty vtk widget after stop, play sequence for some 3D simulations
        #            self.projSpinBox.setValue(fieldDim.z/2 + 1)
        self.projSpinBox.setValue(fieldDim.z / 2)  # If you want to set the value from configuration

    #        self.projSpinBox.setWrapping(True)

    def updateCrossSection(self, _basicSimulationData):
        fieldDim = _basicSimulationData.fieldDim
        self.xyMaxPlane = fieldDim.z - 1
        #        self.xyPlane = fieldDim.z/2 + 1
        self.xyPlane = fieldDim.z / 2

        self.xzMaxPlane = fieldDim.y - 1
        self.xzPlane = fieldDim.y / 2

        self.yzMaxPlane = fieldDim.x - 1
        self.yzPlane = fieldDim.x / 2

    def setFieldTypesComboBox(self, _fieldTypes):
        self.fieldTypes = _fieldTypes  # assign field types to be the same as field types in the workspace
        self.draw_view_2D.setFieldTypes(
            self.fieldTypes)  # make sure that field types are the same in graphics widget and in the drawing object
        self.draw3D.setFieldTypes(
            self.fieldTypes)  # make sure that field types are the same in graphics widget and in the drawing object
        # self.draw3D.setFieldTypes(self.fieldTypes)# make sure that field types are the same in graphics widget and in the drawing object

        self.fieldComboBox.clear()
        self.fieldComboBox.addItem("-- Field Type --")
        self.fieldComboBox.addItem("Cell_Field")
        for key in list(self.fieldTypes.keys()):
            if key != "Cell_Field":
                self.fieldComboBox.addItem(key)
        self.fieldComboBox.setCurrentIndex(1)  # setting value of the Combo box to be cellField - default action

        # self.qvtkWidget.resetCamera() # last call triggers fisrt call to draw function so we here reset camera so that all the actors are initially visible
        self.resetCamera()  # last call triggers fisrt call to draw function so we here reset camera so that all the actors are initially visible

    def resetCamera(self):
        '''
        Resets camera to default settings
        :return:None
        '''
        pass
