import sys
import os
import string
import Configuration
import vtk
from enums import *

import sys

from MVCDrawView2D import MVCDrawView2D
from MVCDrawModel2D import MVCDrawModel2D

from MVCDrawView3D import MVCDrawView3D
from MVCDrawModel3D import MVCDrawModel3D
from Specs import ActorSpecs

MODULENAME = '---- GraphicsFrameWidget.py: '

from weakref import ref

from DrawingParameters import DrawingParameters
from BasicSimulationData import BasicSimulationData


class GenericDrawer():
    def __init__(self, parent=None, originatingWidget=None):

        self.plane = None
        self.planePos = None
        self.field_extractor = None

        self.ren_2D = vtk.vtkRenderer()
        self.ren_3D = vtk.vtkRenderer()

        # self.renWin = self.qvtkWidget.GetRenderWindow()
        # self.renWin.AddRenderer(self.ren)

        # MDIFIX
        self.draw_model_2D = MVCDrawModel2D()
        self.draw_view_2D = MVCDrawView2D(self.draw_model_2D)

        self.draw_model_3D = MVCDrawModel3D()
        self.draw_view_3D = MVCDrawView3D(self.draw_model_3D)

        self.draw_view_2D.ren = self.ren_2D
        self.draw_view_3D.ren = self.ren_3D

        #
        # dict {field_type: drawing fcn}
        self.drawing_fcn_dict = {
            ('CellField', 'Cart'): self.draw_cell_field
        }
        #
        # self.drawModel3D = MVCDrawModel3D(self, self.parentWidget)
        # self.draw3D = MVCDrawView3D(self.drawModel3D, self, self.parentWidget)

        # self.camera3D = self.ren.MakeCamera()
        # self.camera2D = self.ren.GetActiveCamera()
        # self.ren.SetActiveCamera(self.camera2D)

        # self.currentDrawingObject = self.draw2D

        # self.draw3DFlag = False
        # self.usedDraw3DFlag = False
        # # self.getattrFcn=self.getattrDraw2D
        self.screenshotWindowFlag = False

    def set_field_extractor(self, field_extractor):

        self.field_extractor = field_extractor
        self.draw_model_2D.field_extractor = field_extractor
        self.draw_model_3D.field_extractor = field_extractor

    def draw_cell_field(self, drawing_params):
        """
        Draws cell field
        :param drawing_params:
        :return:
        """
        model, view = self.get_model_view(drawing_params=drawing_params)

        actors_specs = ActorSpecs()
        actors_specs.actor_label_list = ['cellsActor']
        actors_specs.metadata = {
            'invisible_types':drawing_params.screenshot_data.invisible_types,
            'all_types':list(range(2+1))
        }

        # actors_dict = view.getActors(actor_label_list=['cellsActor'])
        actor_specs_final = view.prepare_cell_field_actors(actors_specs)

        # model.prepare_cells_actors(actors_specs)

        # cell_actors_metadata = model.get_cell_actors_metadata()

        # model.initCellFieldActors((view.cellsActor,))
        # model.init_cell_field_actors(actors=actors_dict.values())

        model.init_cell_field_actors(actor_specs=actor_specs_final, drawing_params=drawing_params)

        # if drawing_params.screenshot_data.spaceDimension == '2D':
        #     model.init_cell_field_actors(actor_specs=actor_specs_final)
        # else:
        #     if drawing_params.screenshot_data.cell_borders_on:
        #         model.init_cell_field_actors(actor_specs=actor_specs_final)
        #     else:
        #         model.init_cell_field_actors_borderless(actor_specs=actor_specs_final)

        view.show_cell_actors(actor_specs=actor_specs_final)
        # view.setCamera(drawing_params.bsd.fieldDim)

        # self.draw_model_2D.initCellFieldActors((self.draw_view_2D.cellsActor,))
        # self.draw_view_2D.show_cells_actor()

        # self.drawModel.initCellFieldActors((self.cellsActor,))
        # self.drawModel2D.initCellFieldActors((self.drawVcellsActor,))

    def draw_cell_borders(self, drawing_params):
        """
        Draws cell borders
        :param drawing_params:
        :return:
        """
        if drawing_params.screenshot_data.spaceDimension == '3D':
            return


        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs.actor_label_list = ['borderActor']
        actor_specs.actors_dict ={
            'borderActor':view.borderActor
        }


        model.init_borders_actors(actor_specs=actor_specs, drawing_params=drawing_params)
        show_flag = drawing_params.screenshot_data.cell_borders_on
        view.show_cell_borders(show_flag=show_flag)

        # self.draw_model_2D.initBordersActors2D((self.draw_view_2D.borderActor,))
        # show_flag = drawing_params.screenshot_data.cell_borders_on
        # self.draw_view_2D.show_cell_borders(show_flag=show_flag)

    def draw_bounding_box(self, drawing_params):
        model, view = self.get_model_view(drawing_params=drawing_params)

        actor_specs = ActorSpecs()
        actor_specs.actor_label_list = ['outlineActor']
        actor_specs.actors_dict ={
            'outlineActor':view.outlineActor
        }

        model.init_outline_actors(actor_specs=actor_specs,drawing_params=drawing_params)
        show_flag = drawing_params.screenshot_data.bounding_box_on
        view.show_bounding_box(show_flag=show_flag)

        # self.draw_model_2D.init_outline_actors((self.draw_view_2D.outlineActor,))
        # show_flag = drawing_params.screenshot_data.bounding_box_on
        # self.draw_view_2D.show_bounding_box(show_flag=show_flag)

    # def draw(self, drawing_params, screenshot_name):

    # def get_model_view(self, dimension_label):
    #     """
    #     returns pair of model view objects depending on the dimension label
    #     :param dimension_label: {str}
    #     :return: {tuple} mode, view object tuple
    #     """
    #
    #     if dimension_label == '2D':
    #         return self.draw_model_2D, self.draw_view_2D
    #     else:
    #         return self.draw_model_3D, self.draw_view_3D

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

    def draw(self, screenshot_data, bsd, screenshot_name):

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
        ren = view.ren
        # self.draw_model_2D.setDrawingParametersObject(drawing_params)
        model.setDrawingParametersObject(drawing_params)

        try:
            key = (drawing_params.fieldType, 'Cart')
            draw_fcn = self.drawing_fcn_dict[key]

        except KeyError:
            print 'Could not find function for {}'.format(key)
            draw_fcn = None

        if draw_fcn is not None:
            draw_fcn(drawing_params=drawing_params)
            # decorations
            if drawing_params.screenshot_data.cell_borders_on:
                self.draw_cell_borders(drawing_params=drawing_params)
            if drawing_params.screenshot_data.bounding_box_on:
                self.draw_bounding_box(drawing_params=drawing_params)

            # setting camera
            if screenshot_data.spaceDimension == '3D':
                if screenshot_data.clippingRange is not None:
                    view.set_custom_camera(camera_settings=screenshot_data)
            else:
                view.set_default_camera(drawing_params.bsd.fieldDim)

            renWin = vtk.vtkRenderWindow()
            renWin.SetOffScreenRendering(1)
            renWin.AddRenderer(ren)
            renWin.Render()

            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(renWin)
            windowToImageFilter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName('D:/CC3D_GIT/CompuCell3D/player5/GraphicsOffScreen/{screenshot_name}.png'.format(
                screenshot_name=screenshot_name))
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
        print 'resetAllCameras in GraphicsFrame =', self

        self.draw_view_2D.resetAllCameras()
        self.draw3D.resetAllCameras()

    # def __getattr__(self, attr):
    #     """Makes the object behave like a DrawBase"""
    #     if not self.draw3DFlag:
    #         if hasattr(self.draw2D, attr):
    #             return getattr(self.draw2D, attr)
    #         else:
    #             raise AttributeError, self.__class__.__name__ + \
    #                                   " has no attribute named " + attr
    #     else:
    #         if hasattr(self.draw3D, attr):
    #             return getattr(self.draw3D, attr)
    #         else:
    #             raise AttributeError, self.__class__.__name__ + \
    #                                   " has no attribute named " + attr

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
        style = string.upper(_style)
        if style == "2D":
            self.draw3DFlag = False
            self.currentDrawingObject = self.draw_view_2D
            self.ren.SetActiveCamera(self.camera2D)
            self.qvtkWidget.setMouseInteractionSchemeTo2D()
            self.draw3D.clearDisplay()
        elif style == "3D":
            self.draw3DFlag = True
            self.currentDrawingObject = self.draw3D
            self.ren.SetActiveCamera(self.camera3D)
            self.qvtkWidget.setMouseInteractionSchemeTo3D()
            self.draw_view_2D.clearDisplay()

    def getCamera3D(self):
        return self.camera3D

    def getActiveCamera(self):
        return self.ren.GetActiveCamera()

    def getCurrentSceneNameAndType(self):
        # this is usually field name but we can also allow other types of visualizations hence I am calling it getCurrerntSceneName
        sceneName = str(self.fieldComboBox.currentText())
        return sceneName, self.parentWidget.fieldTypes[sceneName]

    def apply3DGraphicsWindowData(self, gwd):

        for p in xrange(self.projComboBox.count()):

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

        for p in xrange(self.projComboBox.count()):

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

        for i in xrange(self.fieldComboBox.count()):

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
        print MODULENAME, '  _takeShot():  self.renWin.GetSize()=', self.renWin.GetSize()
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
            print MODULENAME, '  _takeShot():  fieldType=', fieldType

            #            if self.threeDRB.isChecked():
            if self.draw3DFlag:
                self.parentWidget.screenshotManager.add3DScreenshot(fieldType[0], fieldType[1], camera)
            else:
                planePositionTupple = self.draw_view_2D.getPlane()
                # print "planePositionTupple=",planePositionTupple
                self.parentWidget.screenshotManager.add2DScreenshot(fieldType[0], fieldType[1], planePositionTupple[0],
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
        for key in self.fieldTypes.keys():
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
        self.qvtkWidget.resetCamera()
