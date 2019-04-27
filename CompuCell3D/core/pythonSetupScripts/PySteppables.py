"This module contains definitions of basic classes that are used to construct Python based Steppables"

# import sys

# necessary to keep refernces to attribute adder and dictAdder current
global pyAttributeAdder
global dictAdder
from enums import *
import numpy as np
from SteeringParam import SteeringParam
from collections import OrderedDict

pyAttributeAdder = None
dictAdder = None


# steppables

class SimObjectPy:
    def __init__(self): pass

    def init(self, _simulator):
        self.simulator = _simulator

    def extraInit(self, _simulator):
        self.simulator = _simulator


class SteppablePy(SimObjectPy):

    def __init__(self, _frequency=1):
        self.frequency = _frequency
        self.runBeforeMCS = 0
        # def __name__(self):
        # self.name="Steppable"

    def setFrequency(self, _freq):
        self.frequency = _freq

    def start(self): pass

    def step(self, _mcs): pass

    def finish(self): pass

    def cleanup(self): pass


class FieldVisData(object):
    (CELL_LEVEL_SCALAR_FIELD, CELL_LEVEL_VECTOR_FIELD, HISTOGRAM) = range(0, 3)

    def __init__(self, field, field_type, attribute_name, function=None):
        self.field = field
        self.function = function
        self.attribute_name = attribute_name
        self.number_of_bins = 0
        self.cell_type_list = []
        if function is None:
            self.function = lambda x: x

        self.field_type = field_type


class PlotData(object):
    (HISTOGRAM) = range(0, 1)

    def __init__(self, plot_name, plot_type, attribute_name, function=None):
        self.plot_name = plot_name
        self.plot_window = None
        self.function = function
        self.attribute_name = attribute_name
        self.x_axis_title = ''
        self.y_axis_title = ''
        self.x_scale_type = 'linear'
        self.y_scale_type = 'linear'
        self.number_of_bins = 0
        self.color = 'green'
        self.cell_type_list = []

        if function is None:
            self.function = lambda x: x

        self.plot_type = plot_type


# class SteppableBasePy(SteppablePy):
from SBMLSolverHelper import SBMLSolverHelper


class SteppableBasePy(SteppablePy, SBMLSolverHelper):

    (CC3D_FORMAT, TUPLE_FORMAT) = range(0, 2)

    def __init__(self, _simulator, _frequency=1):
        SteppablePy.__init__(self, _frequency)
        SBMLSolverHelper.__init__(self)
        self.simulator = _simulator
        self.potts = _simulator.getPotts()
        self.cellField = self.potts.getCellFieldG()
        self.dim = self.cellField.getDim()
        self.inventory = self.simulator.getPotts().getCellInventory()
        self.clusterInventory = self.inventory.getClusterInventory()
        self.cellList = CellList(self.inventory)
        self.cellListByType = CellListByType(self.inventory)
        self.clusterList = ClusterList(self.inventory)
        self.clusters = Clusters(self.inventory)
        self.mcs = -1

        # this dictionary stores fields and visualization data for automatic tracking of attribute visualization
        self.tracking_field_vis_dict = {}
        self.plot_dict = {}  # {plot_name:plotWindow  - pW object}
        # this dictionary stores plots for automatic tracking of attribute visualization
        self.tracking_plot_dict = {}

        self.boundaryStrategy = self.simulator.getBoundaryStrategy()

        self.en_calculator = self.potts.getEnergyFunctionCalculator()

        self.__modulesToUpdateDict = {}  # keeps modules to update
        # used by clone attributes functions 
        self.clonableAttributeNames = \
            ['lambdaVolume', 'targetVolume', 'targetSurface', 'lambdaSurface', 'targetClusterSurface',
             'lambdaClusterSurface', \
             'type', 'lambdaVecX', 'lambdaVecY', 'lambdaVecZ', 'fluctAmpl']

        import CompuCellSetup
        global pyAttributeAdder
        global dictAdder
        if not pyAttributeAdder or dictAdder:
            pyAttributeAdder, dictAdder = CompuCellSetup.attachDictionaryToCells(self.simulator)

        import CompuCellSetup
        self.typeIdTypeNameDict = CompuCellSetup.ExtractTypeNamesAndIds()
        for typeId in self.typeIdTypeNameDict:
            setattr(self, self.typeIdTypeNameDict[typeId].upper(), typeId)

        # self.steering_param_dict = OrderedDict()

        # initialize plugins so that they are accessible from python
        self.init_plugins()

    def init_plugins(self):
        """
        prepares plugins exposed to Python to be accessible from Steppable
        :return:
        """
        import CompuCell
        pluginManager = CompuCell.getPluginManagerAsBPM()
        stepManager = CompuCell.getSteppableManagerAsBPM()  # have to use explicit cast to BasicPluginManager to get steppable manager to work

        # VolumeTrackerPlugin
        self.volumeTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("VolumeTracker"):
            import CompuCell
            self.volumeTrackerPlugin = CompuCell.getVolumeTrackerPlugin()
            self.cellField.volumeTrackerPlugin = self.volumeTrackerPlugin  # used in setitem function in swigg CELLFIELDEXTEDER macro CompuCell.i

        # CenterOfMassPlugin
        self.centerOfMassPlugin = None
        if self.simulator.pluginManager.isLoaded("CenterOfMass"):
            import CompuCell
            self.centerOfMassPlugin = CompuCell.getCenterOfMassPlugin()

        # NeighborTrackerPlugin
        self.neighborTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("NeighborTracker"):
            import CompuCell
            self.neighborTrackerPlugin = CompuCell.getNeighborTrackerPlugin()

        # FocalPointPlasticity
        self.focalPointPlasticityPlugin = None
        if self.simulator.pluginManager.isLoaded("FocalPointPlasticity"):
            import CompuCell
            self.focalPointPlasticityPlugin = CompuCell.getFocalPointPlasticityPlugin()

        # Chemotaxis
        self.chemotaxisPlugin = None
        if self.simulator.pluginManager.isLoaded("Chemotaxis"):
            import CompuCell
            self.chemotaxisPlugin = CompuCell.getChemotaxisPlugin()

        # BoundaryPixelTrackerPlugin
        self.boundaryPixelTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("BoundaryPixelTracker"):
            import CompuCell
            self.boundaryPixelTrackerPlugin = CompuCell.getBoundaryPixelTrackerPlugin()

        # PixelTrackerPlugin
        self.pixelTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("PixelTracker"):
            import CompuCell
            self.pixelTrackerPlugin = CompuCell.getPixelTrackerPlugin()

        # ElasticityTrackerPlugin
        self.elasticityTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("ElasticityTracker"):
            import CompuCell
            self.elasticityTrackerPlugin = CompuCell.getElasticityTrackerPlugin()

        # PlasticityTrackerPlugin
        self.plasticityTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("PlasticityTracker"):
            import CompuCell
            self.plasticityTrackerPlugin = CompuCell.getPlasticityTrackerPlugin()

        # ConnectivityLocalFlexPlugin
        self.connectivityLocalFlexPlugin = None
        if self.simulator.pluginManager.isLoaded("ConnectivityLocalFlex"):
            import CompuCell
            self.connectivityLocalFlexPlugin = CompuCell.getConnectivityLocalFlexPlugin()

        # ConnectivityLocalFlexPlugin
        self.connectivityGlobalPlugin = None
        if self.simulator.pluginManager.isLoaded("ConnectivityGlobal"):
            import CompuCell
            self.connectivityGlobalPlugin = CompuCell.getConnectivityGlobalPlugin()

            # #LengthConstraintLocalFlexPlugin
            # self.lengthConstraintLocalFlexPlugin=None
            # if self.simulator.pluginManager.isLoaded("LengthConstraintLocalFlex"):
            # import CompuCell            
            # self.lengthConstraintLocalFlexPlugin=CompuCell.getLengthConstraintLocalFlexPlugin()

        # LengthConstraintPlugin
        self.lengthConstraintPlugin = None
        if self.simulator.pluginManager.isLoaded("LengthConstraint"):
            import CompuCell
            self.lengthConstraintPlugin = CompuCell.getLengthConstraintPlugin()
            self.lengthConstraintLocalFlexPlugin = self.lengthConstraintPlugin  # kept for compatibility reasons

        # ContactLocalFlexPlugin
        self.contactLocalFlexPlugin = None
        if self.simulator.pluginManager.isLoaded("ContactLocalFlex"):
            import CompuCell
            self.contactLocalFlexPlugin = CompuCell.getContactLocalFlexPlugin()

        # ContactLocalProductPlugin
        self.contactLocalProductPlugin = None
        if self.simulator.pluginManager.isLoaded("ContactLocalProduct"):
            import CompuCell
            self.contactLocalProductPlugin = CompuCell.getContactLocalProductPlugin()

        # ContactMultiCadPlugin
        self.contactMultiCadPlugin = None
        if self.simulator.pluginManager.isLoaded("ContactMultiCad"):
            import CompuCell
            self.contactMultiCadPlugin = CompuCell.getContactMultiCadPlugin()

        # ContactOrientationPlugin
        self.contactOrientationPlugin = None
        if self.simulator.pluginManager.isLoaded("ContactOrientation"):
            import CompuCell
            self.contactOrientationPlugin = CompuCell.getContactOrientationPlugin()

        # AdhesionFlexPlugin
        self.adhesionFlexPlugin = None
        if self.simulator.pluginManager.isLoaded("AdhesionFlex"):
            import CompuCell
            self.adhesionFlexPlugin = CompuCell.getAdhesionFlexPlugin()

        # CellOrientationPlugin
        self.cellOrientationPlugin = None
        if self.simulator.pluginManager.isLoaded("CellOrientation"):
            import CompuCell
            self.cellOrientationPlugin = CompuCell.getCellOrientationPlugin()

        # PolarizationVectorPlugin
        self.polarizationVectorPlugin = None
        if self.simulator.pluginManager.isLoaded("PolarizationVector"):
            import CompuCell
            self.polarizationVectorPlugin = CompuCell.getPolarizationVectorPlugin()

        # MomentOfInertiaPlugin
        self.momentOfInertiaPlugin = None
        if self.simulator.pluginManager.isLoaded("MomentOfInertia"):
            import CompuCell
            self.momentOfInertiaPlugin = CompuCell.getMomentOfInertiaPlugin()

        # SecretionPlugin
        self.secretionPlugin = None
        if self.simulator.pluginManager.isLoaded("Secretion"):
            import CompuCell
            self.secretionPlugin = CompuCell.getSecretionPlugin()

        # ClusterSurfacePlugin
        self.clusterSurfacePlugin = None
        if self.simulator.pluginManager.isLoaded("ClusterSurface"):
            import CompuCell
            self.clusterSurfacePlugin = CompuCell.getClusterSurfacePlugin()

        # ClusterSurfaceTrackerPlugin
        self.clusterSurfaceTrackerPlugin = None
        if self.simulator.pluginManager.isLoaded("ClusterSurfaceTracker"):
            import CompuCell
            self.clusterSurfaceTrackerPlugin = CompuCell.getClusterSurfaceTrackerPlugin()

        # polarization23Plugin
        self.polarization23Plugin = None
        if self.simulator.pluginManager.isLoaded("Polarization23"):
            import CompuCell
            self.polarization23Plugin = CompuCell.getPolarization23Plugin()

        # cellTypeMonitorPlugin
        self.cellTypeMonitorPlugin = None
        if self.simulator.pluginManager.isLoaded("CellTypeMonitor"):
            import CompuCell
            self.cellTypeMonitorPlugin = CompuCell.getCellTypeMonitorPlugin()

        # boundaryMonitorPlugin
        self.boundaryMonitorPlugin = None
        if self.simulator.pluginManager.isLoaded("BoundaryMonitor"):
            import CompuCell
            self.boundaryMonitorPlugin = CompuCell.getBoundaryMonitorPlugin()

        self.curvatureCalculatorPlugin = None
        if self.simulator.pluginManager.isLoaded("CurvatureCalculator"):
            import CompuCell
            self.curvatureCalculatorPlugin = CompuCell.getCurvatureCalculatorPlugin()



        # CleaverMeshDumper
        self.cleaverMeshDumper = None
        if stepManager.isLoaded("CleaverMeshDumper"):
            import CompuCell
            self.cleaverMeshDumper = CompuCell.getCleaverMeshDumper()


    def add_steering_panel(self):
        pass

    def process_steering_panel_data(self):
        pass

    def process_steering_panel_data_wrapper(self):
        """
        Calls process_steering_panel_data if and only if there are dirty
        parameters in the steering panel model
        :return: None
        """
        if self.steering_param_dirty():
            self.process_steering_panel_data()

        # NOTE: resetting of the dirty flag for the steering
        # panel model is done in the SteppableRegistry's "step" function


    def add_steering_param(self,name, val, min_val=None, max_val=None, decimal_precision=3, enum=None,widget_name=None):
        if self.mcs >=0:
            raise RuntimeError ('Steering Parameters Can only be added in "__init__" or "start" function of the steppable')

        import CompuCellSetup
        if name in CompuCellSetup.steering_param_dict.keys():
            raise RuntimeError('Steering parameter named {} has already been defined. Please use different parameter name'.format(name))

        CompuCellSetup.steering_param_dict[name]=SteeringParam(name=name, val=val, min_val=min_val, max_val=max_val,
                                                       decimal_precision=decimal_precision,
                                                       enum=enum,widget_name=widget_name)


    def get_steering_param(self,name):
        import CompuCellSetup
        try:
            return CompuCellSetup.steering_param_dict[name].val
        except KeyError:
            raise RuntimeError('Could not find steering_parameter named {}'.format(name))

    def steering_param_dirty(self, name=None):
        """
        Checks if a given steering parameter is dirty or if name is None if any of the parameters are dirty

        :param name:{str} name of the parameter
        True gets returned (False otherwise)
        :return:{bool} dirty flag
        """
        import CompuCellSetup
        if name is not None:
            return self.get_steering_param(name=name).dirty_flag
        else:
            for p_name, steering_param in CompuCellSetup.steering_param_dict.items():
                if steering_param.dirty_flag:
                    return True
            return False

    def set_steering_param_dirty(self, name=None, flag=True):
        """
        Sets dirty flag for given steering parameter or if name is None all parameters
        have their dirty flag set to a given boolean value

        :param name:{str} name of the parameter
        :param flag:{bool} dirty_flag
        :return:None
        """
        import CompuCellSetup
        if name is not None:
            self.get_steering_param(name=name).dirty_flag = flag
        else:
            for p_name, steering_param in CompuCellSetup.steering_param_dict.items():
                steering_param.dirty_flag = flag

    def addNewPlotWindow(self, _title, _xAxisTitle, _yAxisTitle, _xScaleType='linear', _yScaleType='linear',_grid=True,_config_options=None):

        import CompuCellSetup
        if _title in self.plot_dict.keys():
            raise RuntimeError('PLOT WINDOW: ' + _title + ' already exists. Please choose a different name')

        pW = CompuCellSetup.addNewPlotWindow(_title, _xAxisTitle, _yAxisTitle, _xScaleType, _yScaleType,_grid,_config_options=_config_options)
        self.plot_dict[_title] = pW

        return pW

    def update_all_plots_windows(self):
        # tracking visualization part

        for plot_window_name, plot_window in self.plot_dict.iteritems():
            plot_window.showAllPlots()
            plot_window.showAllHistPlots()
            plot_window.showAllBarCurvePlots()

    def track_cell_level_scalar_attribute(self, field_name, attribute_name, function=None, cell_type_list=[]):

        field_vis_data = FieldVisData(field=self.createScalarFieldCellLevelPy(field_name),
                                      field_type=FieldVisData.CELL_LEVEL_SCALAR_FIELD, attribute_name=attribute_name,
                                      function=function)
        field_vis_data.cell_type_list = cell_type_list

        self.tracking_field_vis_dict[field_name] = field_vis_data

    def track_cell_level_vector_attribute(self, field_name, attribute_name, function=None, cell_type_list=[]):

        field_vis_data = FieldVisData(field=self.createVectorFieldCellLevelPy(field_name),
                                      field_type=FieldVisData.CELL_LEVEL_VECTOR_FIELD, attribute_name=attribute_name,
                                      function=function)
        field_vis_data.cell_type_list = cell_type_list

        self.tracking_field_vis_dict[field_name] = field_vis_data

    def histogram_scalar_attribute(self, histogram_name, attribute_name, number_of_bins, function=None,
                                   cell_type_list=[], x_axis_title='', y_axis_title='', color='green',
                                   x_scale_type='linear', y_scale_type='linear'):

        tpd = PlotData(plot_name=histogram_name, plot_type=PlotData.HISTOGRAM, attribute_name=attribute_name,
                       function=function)
        tpd.number_of_bins = number_of_bins

        tpd.x_scale_type = x_scale_type
        tpd.y_scale_type = y_scale_type

        tpd.x_axis_title = x_axis_title
        tpd.y_axis_title = y_axis_title
        if x_axis_title == '':
            tpd.x_axis_title = histogram_name
        if y_axis_title == '':
            tpd.y_axis_title = 'Value'

        tpd.color = color

        tpd.cell_type_list = cell_type_list

        self.tracking_plot_dict[histogram_name] = tpd
        # print 'self.tracking_plot_dict=', self.tracking_plot_dict

    def initialize_tracking_plots(self):

        for plot_name, tracking_plot_data in self.tracking_plot_dict.iteritems():
            tpd = tracking_plot_data
            pW = self.addNewPlotWindow(_title=tpd.plot_name, _xAxisTitle=tpd.x_axis_title, _yAxisTitle=tpd.y_axis_title,
                                       _xScaleType=tpd.x_scale_type, _yScaleType=tpd.y_scale_type)
            pW.addHistogramPlot(_plotName=tpd.plot_name, _color=tpd.color)
            tpd.plot_window = pW

    def fetch_attribute(self, cell, attrib_name):
        try:
            return cell.dict[attrib_name]
        except KeyError:
            try:
                return getattr(cell, attrib_name)
            except AttributeError:
                raise KeyError('Could not locate attribute: ' + attrib_name + ' in a cell object')

    def update_tracking_histogram(self, tracking_plot_data):

        tpd = tracking_plot_data

        val_list = []
        plot_window = tpd.plot_window

        if tpd.cell_type_list == []:
            selective_cell_list = self.cellList
        else:
            # unpacking lsit to positional arguments - using * operator
            selective_cell_list = self.cellListByType(*tpd.cell_type_list)
        try:
            for cell in selective_cell_list:

                try:
                    attrib = self.fetch_attribute(cell, tpd.attribute_name)
                except KeyError:
                    continue
                val_list.append(tpd.function(attrib))

        except:
            raise RuntimeError(
                'Automatic Attribute Tracking :wrong type of cell attribute, missing attribute or wrong tracking function is used by track_cell_level functions')

        if len(val_list):
            plot_window.addHistogram(plot_name=tpd.plot_name, value_array=val_list, number_of_bins=tpd.number_of_bins)

    def update_tracking_plot(self):

        for plot_name, tracking_plot_data in self.tracking_plot_dict.iteritems():
            if tracking_plot_data.plot_type == PlotData.HISTOGRAM:
                self.update_tracking_histogram(tracking_plot_data)

    def update_tracking_fields(self):
        # tracking visualization part
        for field_name, field_vis_data in self.tracking_field_vis_dict.iteritems():

            field_vis_data.field.clear()

            if field_vis_data.cell_type_list == []:
                selective_cell_list = self.cellList
            else:
                # unpacking lsit to positional arguments - using * operator
                selective_cell_list = self.cellListByType(*tpd.cell_type_list)

            try:
                for cell in selective_cell_list:
                    try:
                        # attrib = cell.dict[field_vis_data.attribute_name]
                        attrib = self.fetch_attribute(cell, field_vis_data.attribute_name)

                    except KeyError:
                        continue
                    field_vis_data.field[cell] = field_vis_data.function(attrib)
            except:
                raise RuntimeError(
                    'Automatic Attribute Tracking :wrong type of cell attribute, missing attribute or wrong tracking function is used by track_cell_level functions')

    def initialize_automatic_tasks(self):
        self.initialize_tracking_plots()

    def perform_automatic_tasks(self):
        self.update_tracking_fields()
        self.update_tracking_plot()
        self.update_all_plots_windows()

    def areCellsDifferent(self, _cell1, _cell2):
        import CompuCell
        return CompuCell.areCellsDifferent(_cell1, _cell2)

    def getDictionaryAttribute(self, _cell):
        # access/modification of a dictionary attached to cell - make sure to decalare in main script that you will use such attribute
        import CompuCell
        return CompuCell.getPyAttrib(_cell)

    def changeNumberOfWorkNodes(self, _numberOfWorkNodes):

        import CompuCell
        numberOfWorkNodesEv = CompuCell.CC3DEventChangeNumberOfWorkNodes()
        numberOfWorkNodesEv.oldNumberOfNodes = 1
        numberOfWorkNodesEv.newNumberOfNodes = _numberOfWorkNodes
        self.simulator.postEvent(numberOfWorkNodesEv)

    def resizeAndShiftLattice(self, _newSize, _shiftVec=(0, 0, 0)):

        print 'PYSTEPPABLES INSIDE resizeAndShiftLattice'
        if self.potts.getBoundaryXName().lower() == 'periodic' \
                or self.potts.getBoundaryYName().lower() == 'periodic' \
                or self.potts.getBoundaryZName().lower() == 'periodic':
            raise EnvironmentError('Cannot resize lattice with Periodic Boundary Conditions')

        import CompuCell

        newSize = map(int, _newSize)  # converting new size to integers
        shiftVec = map(int, _shiftVec)  # converting shift vec to integers

        okFlag = self.volumeTrackerPlugin.checkIfOKToResize(CompuCell.Dim3D(newSize[0], newSize[1], newSize[2]),
                                                            CompuCell.Dim3D(shiftVec[0], shiftVec[1], shiftVec[2]))
        print 'okFlag=', okFlag
        if not okFlag:
            print 'WARNING: Lattice Resize Denied  - the proposed lattice resizing/shift would lead to disappearance of cells.'
            return

        oldGeometryDimensionality = 2
        if self.dim.x > 1 and self.dim.y > 1 and self.dim.z > 1:
            oldGeometryDimensionality = 3

        newGeometryDimensionality = 2
        if newSize[0] > 1 and newSize[1] > 1 and newSize[2] > 1:
            newGeometryDimensionality = 3

        if newGeometryDimensionality != oldGeometryDimensionality:
            raise EnvironmentError(
                'Changing dimmensionality of simulation from 2D to 3D is not supported. It also makes little sense as 2D and 3D simulations have different mathematical properties. Please see CPM literature for more details.')

        self.potts.resizeCellField(CompuCell.Dim3D(newSize[0], newSize[1], newSize[2]),
                                   CompuCell.Dim3D(shiftVec[0], shiftVec[1], shiftVec[2]))
        #         if sum(shiftVec)==0: # there is no shift in cell field
        #             return


        # posting CC3DEventLatticeResize so that participating modules can react
        resizeEv = CompuCell.CC3DEventLatticeResize()
        resizeEv.oldDim = self.dim
        resizeEv.newDim = CompuCell.Dim3D(newSize[0], newSize[1], newSize[2])
        resizeEv.shiftVec = CompuCell.Dim3D(shiftVec[0], shiftVec[1], shiftVec[2])

        self.simulator.postEvent(resizeEv)

        self.__init__(self.simulator, self.frequency)
        import CompuCellSetup

        # with new cell field and possibly other fields  we have to reinitialize steppables
        for steppable in CompuCellSetup.globalSteppableRegistry.allSteppables():
            if steppable != self:
                steppable.__init__(steppable.simulator, steppable.frequency)

    def everyPixelWithSteps(self, step_x, step_y, step_z):
        for x in xrange(0, self.dim.x, step_x):
            for y in xrange(0, self.dim.y, step_y):
                for z in xrange(0, self.dim.z, step_z):
                    yield x, y, z

    def everyPixel(self, step_x=1, step_y=1, step_z=1):
        if step_x == 1 and step_y == 1 and step_z == 1:
            import itertools
            return itertools.product(xrange(self.dim.x), xrange(self.dim.y), xrange(self.dim.z))
        else:
            return self.everyPixelWithSteps(step_x, step_y, step_z)

    def attemptFetchingCellById(self, _id):
        return self.inventory.attemptFetchingCellById(_id)

    def getCellByIds(self, _id, _clusterId):
        return self.inventory.getCellByIds(_id, _clusterId)

    def getClusterCells(self, _clusterId):
        return ClusterCellList(self.inventory.getClusterInventory().getClusterCells(_clusterId))
        # may work on some systems
        # return self.inventory.getClusterInventory().getClusterCells(_clusterId)       

    def reassignClusterId(self, _cell, _clusterId):
        oldClusterId = _cell.clusterId
        newClusterId = _clusterId
        self.inventory.reassignClusterId(_cell, newClusterId)
        if self.clusterSurfaceTrackerPlugin:
            self.clusterSurfaceTrackerPlugin.updateClusterSurface(oldClusterId)
            self.clusterSurfaceTrackerPlugin.updateClusterSurface(newClusterId)

    def getCellNeighbors(self, _cell):
        if not self.neighborTrackerPlugin:
            raise AttributeError('Could not find NeighborTrackerPlugin')

        return CellNeighborListAuto(self.neighborTrackerPlugin, _cell)

    def getCellNeighborDataList(self, _cell):
        if not self.neighborTrackerPlugin:
            raise AttributeError('Could not find NeighborTrackerPlugin')

        return CellNeighborListFlex(self.neighborTrackerPlugin, _cell)

        # return self.getCellNeighbors(_cell)

    def getFocalPointPlasticityDataList(self, _cell):
        if self.focalPointPlasticityPlugin:
            return FocalPointPlasticityDataList(self.focalPointPlasticityPlugin, _cell)

        return None

    def getInternalFocalPointPlasticityDataList(self, _cell):
        if self.focalPointPlasticityPlugin:
            return InternalFocalPointPlasticityDataList(self.focalPointPlasticityPlugin, _cell)

        return None

    def getAnchorFocalPointPlasticityDataList(self, _cell):
        if self.focalPointPlasticityPlugin:
            return AnchorFocalPointPlasticityDataList(self.focalPointPlasticityPlugin, _cell)

        return None

    def getCellBoundaryPixelList(self, _cell, _neighborOrder=-1):
        if self.boundaryPixelTrackerPlugin:
            return CellBoundaryPixelList(self.boundaryPixelTrackerPlugin, _cell, _neighborOrder)

        return None

    def getCopyOfCellBoundaryPixels(self, _cell, _format=CC3D_FORMAT):
        try:
            if _format == SteppableBasePy.CC3D_FORMAT:
                import CompuCell
                return [CompuCell.Point3D(boundaryPixelTrackerData.pixel) for boundaryPixelTrackerData in
                        self.getCellBoundaryPixelList(_cell)]
            else:
                return [(boundaryPixelTrackerData.pixel.x, boundaryPixelTrackerData.pixel.y,
                         boundaryPixelTrackerData.pixel.z) for boundaryPixelTrackerData in
                        self.getCellBoundaryPixelList(_cell)]
        except:
            raise AttributeError('Could not find BoundaryPixelTracker Plugin')

    def getCellPixelList(self, _cell):
        if self.pixelTrackerPlugin:
            return CellPixelList(self.pixelTrackerPlugin, _cell)

        return None

    def getCopyOfCellPixels(self, _cell, _format=CC3D_FORMAT):

        try:
            if _format == SteppableBasePy.CC3D_FORMAT:
                import CompuCell
                return [CompuCell.Point3D(pixelTrackerData.pixel) for pixelTrackerData in self.getCellPixelList(_cell)]
            else:
                return [(pixelTrackerData.pixel.x, pixelTrackerData.pixel.y, pixelTrackerData.pixel.z) for
                        pixelTrackerData in self.getCellPixelList(_cell)]
        except:
            raise AttributeError('Could not find PixelTracker Plugin')

    def getElasticityDataList(self, _cell):
        if self.elasticityTrackerPlugin:
            return ElasticityDataList(self.elasticityTrackerPlugin, _cell)

    def getPlasticityDataList(self, _cell):
        if self.plasticityTrackerPlugin:
            return PlasticityDataList(self.plasticityTrackerPlugin, _cell)

        return None

    def getFieldSecretor(self, _fieldName):

        if self.secretionPlugin:
            return self.secretionPlugin.getFieldSecretor(_fieldName)

        raise RuntimeError(
            "Please define Secretion Plugin in the XML before requesting secretor object from Python script."
            'Secreion Plugin can be defined by including <Plugin Name="Secretion"/> in the XML')

    '''    
        We have to call volumeTracker. setp function manually when tryuign to delete cell. This function is called only from potts loop whil Python steppables are run outside this loop.
    '''

    def cleanDeadCells(self):
        if self.volumeTrackerPlugin:
            self.volumeTrackerPlugin.step()

    def clean_cell_field(self,reset_inventory=True):

        self.potts.clean_cell_field(reset_inventory)


    def deleteCell(self, cell):
        import CompuCell
        #         pixelsToDelete=self.getCopyOfCellPixels(cell,SteppableBasePy.CC3D_FORMAT) # returns list of Point3D
        pixelsToDelete = self.getCopyOfCellPixels(cell, SteppableBasePy.TUPLE_FORMAT)  # returns list of tuples
        self.mediumCell = CompuCell.getMediumCell()
        for pixel in pixelsToDelete:
            self.cellField[pixel[0], pixel[1], pixel[2]] = self.mediumCell

    def createNewCell(self, type, pt, xSize, ySize, zSize=1):
        import CompuCell
        ptTmp = CompuCell.Point3D()

        if isinstance(pt, list) or isinstance(pt, tuple):
            ptTmp.x = pt[0]
            ptTmp.y = pt[1]
            ptTmp.z = pt[2]
        else:
            ptTmp = pt

        if not self.checkIfInTheLattice(ptTmp):
            return None
        cell = self.potts.createCell()
        cell.type = type
        self.cellField[ptTmp.x:ptTmp.x + xSize - 1, ptTmp.y:ptTmp.y + ySize - 1, ptTmp.z:ptTmp.z + zSize - 1] = cell
        return cell

    def newCell(self, type=0):
        cell = self.potts.createCell()
        cell.type = type
        return cell

    def moveCell(self, cell, shiftVector):
        import CompuCell
        # we have to make two list of pixels :
        pixelsToDelete = []  # used to hold pixels to delete
        pixelsToMove = []  # used to hold pixels to move

        shiftVec = CompuCell.Point3D()
        if isinstance(shiftVector, list) or isinstance(shiftVector, tuple):
            shiftVec.x = shiftVector[0]
            shiftVec.y = shiftVector[1]
            shiftVec.z = shiftVector[2]
        else:
            shiftVec = shiftVector
        # If we try to reassign pixels in the loop where we iterate over pixel data we will corrupt the container so in the loop below all we will do is to populate the two list mentioned above
        pixelList = self.getCellPixelList(cell)
        pt = CompuCell.Point3D()

        for pixelTrackerData in pixelList:
            pt.x = pixelTrackerData.pixel.x + shiftVec.x
            pt.y = pixelTrackerData.pixel.y + shiftVec.y
            pt.z = pixelTrackerData.pixel.z + shiftVec.z
            # here we are making a copy of the cell                 
            pixelsToDelete.append(CompuCell.Point3D(pixelTrackerData.pixel))

            if self.checkIfInTheLattice(pt):
                pixelsToMove.append(CompuCell.Point3D(pt))
                # self.cellField.set(pt,cell)

        # Now we will move cell
        for pixel in pixelsToMove:
            self.cellField[pixel.x, pixel.y, pixel.z] = cell

        # Now we will delete old pixels    
        self.mediumCell = CompuCell.getMediumCell()
        for pixel in pixelsToDelete:
            self.cellField[pixel.x, pixel.y, pixel.z] = self.mediumCell

    def checkIfInTheLattice(self, _pt):
        if _pt.x >= 0 and _pt.x < self.dim.x and _pt.y >= 0 and _pt.y < self.dim.y and _pt.z >= 0 and _pt.z < self.dim.z:
            return True
        return False

    def buildWall(self, type):
        cell = None
        if type == 0:  # medium:
            import CompuCell
            cell = CompuCell.getMediumCell()
        else:
            cell = self.potts.createCell()
            cell.type = type

        indexOf1 = -1
        dimLocal = [self.dim.x, self.dim.y, self.dim.z]

        for idx in range(len(dimLocal)):

            if dimLocal[idx] == 1:
                indexOf1 = idx
                break

        # this could be recoded in a more general way but the code would be longer than the most naive verios presented here
        if indexOf1 >= 0:  # 2D case
            if indexOf1 == 2:  # xy plane simulation
                self.cellField[0:self.dim.x, 0, 0] = cell
                self.cellField[0:self.dim.x, self.dim.y - 1:self.dim.y, 0] = cell
                self.cellField[0, 0:self.dim.y, 0] = cell
                self.cellField[self.dim.x - 1:self.dim.x, 0:self.dim.y, 0:0] = cell
            elif indexOf1 == 0:  # yz simulation
                self.cellField[0, 0:self.dim.y, 0] = cell
                self.cellField[0, 0:self.dim.y, self.dim.z - 1:self.dim.z] = cell
                self.cellField[0, 0, 0:self.dim.z] = cell
                self.cellField[0, self.dim.y - 1:self.dim.y, 0:self.dim.z] = cell

            elif indexOf1 == 1:  # xz simulation
                self.cellField[0:self.dim.x, 0, 0] = cell
                self.cellField[0:self.dim.x, 0, self.dim.z - 1:self.dim.z] = cell
                self.cellField[0, 0, 0:self.dim.z] = cell
                self.cellField[self.dim.x - 1:self.dim.x, 0, 0:self.dim.z] = cell
        else:  # 3D case
            # wall 1 (front)
            self.cellField[0:self.dim.x, 0:self.dim.y, 0] = cell
            # wall 2 (rear)
            self.cellField[0:self.dim.x, 0:self.dim.y, self.dim.z - 1] = cell
            # wall 3 (bottom)
            self.cellField[0:self.dim.x, 0, 0:self.dim.z] = cell
            # wall 4 (top)
            self.cellField[0:self.dim.x, self.dim.y - 1, 0:self.dim.z] = cell
            # wall 5 (left)
            self.cellField[0, 0:self.dim.y, 0:self.dim.z] = cell
            # wall 6 (right)
            self.cellField[self.dim.x - 1, 0:self.dim.y, 0:self.dim.z] = cell

    def destroyWall(self):
        self.buildWall(0)  # build wall of Medium

    def getPixelNeighborsBasedOnDistance(self, _pixel, _maxDistance=1.1):
        import CompuCell
        boundaryStrategy = CompuCell.BoundaryStrategy.getInstance()
        maxNeighborIndex = boundaryStrategy.getMaxNeighborIndexFromDepth(_maxDistance)

        for i in xrange(maxNeighborIndex + 1):
            pixelNeighbor = boundaryStrategy.getNeighborDirect(_pixel, i)
            if pixelNeighbor.distance:  # neighbor is valid
                yield pixelNeighbor

    def getPixelNeighborsBasedOnNeighborOrder(self, _pixel, _neighborOrder=1):
        import CompuCell
        boundaryStrategy = CompuCell.BoundaryStrategy.getInstance()
        maxNeighborIndex = boundaryStrategy.getMaxNeighborIndexFromNeighborOrder(_neighborOrder)

        for i in xrange(maxNeighborIndex + 1):
            pixelNeighbor = boundaryStrategy.getNeighborDirect(_pixel, i)
            if pixelNeighbor.distance:  # neighbor is valid
                yield pixelNeighbor

    def invariantDistanceVectorInteger(self, _from=[0, 0, 0], _to=[0, 0, 0]):
        '''
        
        This function will calculate distance vector with integer coordinates between two Point3D points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place 
        @_from - list/tuple of 3 integers
        @_to  -list/tuple of 3 integers
        @ return value - numpy float array    
        '''
        import CompuCell
        import numpy
        distVec = CompuCell.distanceVectorInvariant(_to, _from, self.dim)
        return numpy.array([float(distVec.x), float(distVec.y), float(distVec.z)])

    def distanceVector(self, _from, _to):
        '''
        This function will calculate distance vector between  two points - (_to-_from)  
        This is most straightforward implementation and will ignore periodic boundary conditions if such are present
        @_from - list/tuple of 3 numbers
        @_to  -list/tuple of 3 numbers        
        @ return value - numpy float array
        '''
        import numpy
        return numpy.array([float(_to[0] - _from[0]), float(_to[1] - _from[1]), float(_to[2] - _from[2])])

    def invariantDistanceVector(self, _to, _from):
        '''
        This function will calculate distance vector with integer coordinates between two Coordinates3D<double> points
        and make sure that the absolute values of the vector are smaller than 1/2 of the corresponding lattice dimension
        this way we simulate 'invariance' of distance assuming that periodic boundary conditions are in place            
        @_from - list/tuple/numpy array of 3 floating point numbers
        @_from - list/tuple/numpy array of 3 floating point numbers
        @ return value - numpy float array
        '''
        import CompuCell
        import numpy

        distVec = CompuCell.distanceVectorCoordinatesInvariant(_to, _from, self.dim)
        return numpy.array([distVec.x, distVec.y, distVec.z])

    def distance(self, _from, _to):
        '''
        Distance between two points. Assumes non-periodic boundary conditions
        @return value - floating point number 
        '''
        return self.vectorNorm(self.distanceVector(_from, _to))

    def invariantDistance(self, _from, _to):
        '''
        Distance between two points. Assumes periodic boundary conditions 
        - or simply makes sure that no component of distance vector 
        is greater than 1/2 corresponding dimension
        @return value - floating point number 
        '''

        return self.vectorNorm(self.invariantDistanceVector(_from, _to))

    def vectorNorm(self, _vec):
        import numpy
        return numpy.linalg.norm(_vec)

    def distanceVectorBetweenCells(self, _cell_from, _cell_to):
        '''
        This function will calculate distance vector between  COM's of cells  assuming non-periodic boundary conditions
        @ return value - numpy float array
        '''
        return self.distanceVector([_cell_to.xCOM, _cell_to.yCOM, _cell_to.zCOM],
                                   [_cell_from.xCOM, _cell_from.yCOM, _cell_from.zCOM])

    def invariantDistanceVectorBetweenCells(self, _cell_from, _cell_to):
        '''
        This function will calculate distance vector between  COM's of cells  assuming periodic boundary conditions
        - or simply makes sure that no component of distance vector 
        is greater than 1/2 corresponding dimension        
        @ return value - numpy float array
        '''
        return self.invariantDistanceVector([_cell_to.xCOM, _cell_to.yCOM, _cell_to.zCOM],
                                            [_cell_from.xCOM, _cell_from.yCOM, _cell_from.zCOM])

    def distanceBetweenCells(self, _cell_from, _cell_to):
        '''
        Distance between COM's between cells. Assumes non-periodic boundary conditions        
        @return value - floating point number 
        '''

        return self.vectorNorm(self.distanceVectorBetweenCells(_cell_from, _cell_to))

    def invariantDistanceBetweenCells(self, _cell_from, _cell_to):
        '''
        Distance between COM's of two cells. Assumes periodic boundary conditions 
        - or simply makes sure that no component of distance vector 
        is greater than 1/2 corresponding dimension
        @return value - floating point number 
        '''
        return self.vectorNorm(self.invariantDistanceVectorBetweenCells(_cell_from, _cell_to))

    def point3DToNumpy(self, _pt):
        '''
        This function converts CompuCell.Point3D into floating point numpy array(vector) of size 3
        '''
        import numpy
        return numpy.array([float(_pt.x), float(_pt.y), float(_pt.z)])

    def numpyToPoint3D(self, _array):
        import CompuCell
        pt = CompuCell.Point3D()
        pt.x = _array[0]
        pt.y = _array[1]
        pt.z = _array[2]
        return pt

    def cartesian2Hex(self, _in):
        '''
        this transformation takes coordinates of a point on a cartesian lattice and returns hex coordinates 
        It is coded as HexCoord fcn in BoundaryStrategy.cpp. 
        NOTE: there is c++ implementation of this function which is much faster and 
        Argument: _in is either a tuple or a list or array with 3 elements or Point3D object        
        returns Coordinates3D<double>
        '''
        return self.boundaryStrategy.HexCoord(_in)

    def hex2Cartesian(self, _in):
        '''
        this transformation takes coordinates of a point on a hex lattice and returns integer coordinates of cartesian pixel that is nearest given point on hex lattice 
        It is the inverse transformation of the one coded in HexCoord in BoundaryStrategy.cpp (see Hex2Cartesian). 
       
        Argument: _in is either a tuple or a list or array with 3 elements or Coordinates3D<double> object        
        returns Point3D
        '''
        return self.boundaryStrategy.Hex2Cartesian(_in)

    def hex2CartesianPython(self, _in):
        '''
        this transformation takes coordinates of a point on ahex lattice and returns integer coordinates of cartesian pixel that is nearest given point on hex lattice 
        It is the inverse transformation of the one coded in HexCoord in BoundaryStrategy.cpp. 
        NOTE: there is c+= implementation of this function which is much faster and 
        Argument: _in is either a tuple or a list or array with 3 elements
        '''
        z_segments = int(round(_in[2] / (sqrt(6.0) / 3.0)))

        if (z_segments % 3) == 1:
            y_segments = int(round(_in[1] / (sqrt(3.0) / 2.0) - 2.0 / 6.0))

            if y_segments % 2:

                return Point3D(int(round(_in[0] - 0.5)), y_segments, z_segments)

            else:

                return Point3D(int(round(_in[0])), y_segments, z_segments)

        elif (z_segments % 3) == 2:

            y_segments = int(round(_in[1] / (sqrt(3.0) / 2.0) + 2.0 / 6.0))

            if y_segments % 2:
                return Point3D(int(round(_in[0] - 0.5)), y_segments, z_segments)
            else:
                return Point3D(int(round(_in[0])), y_segments, z_segments)

        else:
            y_segments = int(round(_in[1] / (sqrt(3.0) / 2.0)))

            if y_segments % 2:
                return Point3D(int(round(_in[0])), y_segments, z_segments)
            else:
                return Point3D(int(round(_in[0] - 0.5)), y_segments, z_segments)

    def getSteppableListByClassName(self, _className):
        '''
        This function returns a list of registered steppables with class name given  by _className
        '''
        import CompuCellSetup
        return CompuCellSetup.globalSteppableRegistry.getSteppablesByClassName(_className)

    def getSteppableByClassName(self, _className):
        '''
        This function returns a registered steppables with class name given  by _className.
        It will be the first registered steppable for this given class name. In almost all the cases we will have one steppable with a given class name
        '''
        import CompuCellSetup
        try:
            return CompuCellSetup.globalSteppableRegistry.getSteppablesByClassName(_className)[0]
        except IndexError, e:
            return None

    def openFileInSimulationOutputDirectory(self, _filePath, _mode="r"):
        import CompuCellSetup
        return CompuCellSetup.openFileInSimulationOutputDirectory(_filePath, _mode)

    def stopSimulation(self):
        import CompuCellSetup
        CompuCellSetup.stopSimulation()

    def setMaxMCS(self, maxMCS):
        self.simulator.setNumSteps(maxMCS)

    def createScalarFieldPy(self, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createScalarFieldPy(self.dim, _fieldName)

    def createScalarFieldCellLevelPy(self, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createScalarFieldCellLevelPy(_fieldName)

    def createVectorFieldPy(self, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldPy(self.dim, _fieldName)

    def createVectorFieldCellLevelPy(self, _fieldName):
        import CompuCellSetup
        return CompuCellSetup.createVectorFieldCellLevelPy(_fieldName)

    def getClusterCells(self, _clusterId):
        return self.inventory.getClusterCells(_clusterId)

    def getConcentrationField(self, _fieldName):
        import CompuCell
        return CompuCell.getConcentrationField(self.simulator, _fieldName)

        # def addNewPlotWindow(self, _title='',_xAxisTitle='',_yAxisTitle='',_xScaleType='linear',_yScaleType='linear'):
        # import CompuCellSetup
        # return CompuCellSetup.addNewPlotWindow(_title,_xAxisTitle,_yAxisTitle,_xScaleType,_yScaleType)

    def getXMLElement(self, *args):
        element = None

        if not len(args):
            return None

        if type(args[0]) is not list:  # it is CC3DXMLElement
            element = args[0]
        else:
            element, moduleRoot = self.getXMLElementAndModuleRoot(*args)

        return element if element else None

    def getXMLElementValue(self, *args):

        element = self.getXMLElement(*args)
        return element.getText() if element else None

    def registerXMLElementUpdate(self, *args):
        '''this function registers core module XML Element from wchich XML subelement has been fetched.It returns XML subelement 
        '''
        # element,coreElement=None,None
        # info=sys.version_info
        # if info[0]>=2 and info[1]>5:
        #     element,coreElement=self.getXMLElementAndModuleRoot(*args,returnModuleRoot=True)  # does not work in python 2.5 - syntax error  
        # else:    
        element, coreElement = self.getXMLElementAndModuleRoot(args, returnModuleRoot=True)

        coreNameComposite = coreElement.getName()
        if coreElement.findAttribute('Name'):
            coreNameComposite += coreElement.getAttribute('Name')
        elif coreElement.findAttribute('Type'):
            coreNameComposite += coreElement.getAttribute('Type')

        if element:

            # now will register which modules were modified we will use this information when we call update function    
            currentMCS = self.simulator.getStep()
            try:
                moduleDict = self.__modulesToUpdateDict[currentMCS]
                try:
                    moduleDict[coreNameComposite]
                except LookupError:
                    moduleDict['NumberOfModules'] += 1
                    moduleDict[coreNameComposite] = [coreElement, moduleDict['NumberOfModules']]
                    # # # print 'moduleDict[NumberOfModules]=',moduleDict['NumberOfModules']

            except LookupError:
                self.__modulesToUpdateDict[currentMCS] = {coreNameComposite: [coreElement, 0], 'NumberOfModules': 0}

        return element

    def setXMLElementValue(self, value, *args):

        element = self.registerXMLElementUpdate(*args)
        if element:
            element.updateElementValue(str(value))

    def getXMLAttributeValue(self, attr, *args):
        element = self.getXMLElement(*args)
        if element is not None:
            if element.findAttribute(attr):
                return element.getAttribute(attr)
            else:
                raise LookupError('Could not find attribute ' + attr + ' in ' + args)
        else:
            return None

    def setXMLAttributeValue(self, attr, value, *args):
        element = self.registerXMLElementUpdate(*args)
        if element:
            if element.findAttribute(attr):
                from XMLUtils import dictionaryToMapStrStr as d2mss
                element.updateElementAttributes(d2mss({attr: value}))

    def updateXML(self):
        currentMCS = self.simulator.getStep()
        try:
            # trying to get dictionary of  modules for which XML has been modified during current step
            moduleDict = self.__modulesToUpdateDict[currentMCS]
        except LookupError:
            # if such dictionary does not exist we clean self.__modulesToUpdateDict deleteing whatever was stored before
            self.__modulesToUpdateDict = {}
            return

        try:
            numberOfModules = moduleDict['NumberOfModules']
            del moduleDict['NumberOfModules']
        except LookupError:
            pass

        # [1][1] refers to number denoting the order in which module was added
        # [1][1] refers to added element with order number being [1][1]            
        list_of_tuples = sorted(moduleDict.iteritems(), key=lambda x: x[1][1])

        # # # print 'list_of_tuples=',list_of_tuples    
        for elem_tuple in list_of_tuples:
            self.simulator.updateCC3DModule(elem_tuple[1][0])

    def getXMLElementAndModuleRoot(self, *args, **kwds):
        ''' This fcn fetches xml element value and returns it as text. Potts, Plugin and steppable are special names and roots of these elements are fetched using simulator
            The implementation of this plugin may be simplified. Current implementation is least invasive and requires no changes apart from modifying PySteppables.
            This Function greatly simplifies access to XML data - one line  easily replaces  many lines of code
        '''
        import types

        if isinstance(args[0],
                      types.TupleType):  # depending on Python version we might need to pass "extra-tupple-wrapped" positional arguments especially in situation when variable list arguments are mixed with keyword arguments during function call
            args = args[0]

        if not isinstance(args[0], types.ListType):  # it is CC3DXMLElement
            return args[0]

        from  itertools import izip
        from XMLUtils import dictionaryToMapStrStr as d2mss
        coreModuleElement = None
        tmpElement = None
        for arg in args:
            if type(arg) is list:
                if arg[0] == 'Potts':
                    coreModuleElement = self.simulator.getCC3DModuleData('Potts')
                    tmpElement = coreModuleElement
                elif arg[0] == 'Plugin':
                    counter = 0
                    for attrName in arg:
                        if attrName == 'Name':
                            pluginName = arg[counter + 1]
                            coreModuleElement = self.simulator.getCC3DModuleData('Plugin', pluginName)
                            tmpElement = coreModuleElement
                            break
                        counter += 1

                elif arg[0] == 'Steppable':
                    counter = 0
                    for attrName in arg:
                        if attrName == 'Type':
                            steppableName = arg[counter + 1]
                            coreModuleElement = self.simulator.getCC3DModuleData('Steppable', steppableName)
                            tmpElement = coreModuleElement
                            break
                        counter += 1
                else:
                    # print 'XML FETCH=',arg
                    attrDict = None
                    if len(arg) >= 3:
                        attrDict = {}
                        for tuple in izip(arg[1::2], arg[2::2]):
                            if attrDict.has_key(tuple[0]):
                                raise LookupError('Duplicate attribute name in the access path ' + str(args))
                            else:
                                attrDict[tuple[0]] = tuple[1]
                        attrDict = d2mss(attrDict)
                        # attrDict=d2mss(dict((tuple[0],tuple[1]) for tuple in izip(arg[1::2],arg[2::2])))

                    if coreModuleElement is not None:
                        elemName = arg[0]
                        tmpElement = tmpElement.getFirstElement(arg[0],
                                                                attrDict) if attrDict is not None else tmpElement.getFirstElement(
                            arg[0])

        if tmpElement is None:
            raise LookupError('Could not find element With the following access path', args)

        if 'returnModuleRoot' in kwds.keys():
            return tmpElement, coreModuleElement

        return tmpElement, None

    def cloneAttributes(self, sourceCell, targetCell, no_clone_key_dict_list=[]):
        # clone "C++" attributes
        from copy import deepcopy
        for attrName in self.clonableAttributeNames:
            setattr(targetCell, attrName, getattr(sourceCell, attrName))

        # clone dictionary
        for key, val in sourceCell.dict.iteritems():

            if key in no_clone_key_dict_list:
                continue

            elif key == 'SBMLSolver':
                self.copySBMLs(_fromCell=sourceCell, _toCell=targetCell)

            elif key == 'Bionetwork':
                import bionetAPI
                bionetAPI.copyBionetworkFromParent(sourceCell, targetCell)
            else:
                # copying the rest of dictionary entries    
                targetCell.dict[key] = deepcopy(sourceCell.dict[key])

        # now copy data associated with plugins
        # AdhesionFlex
        if self.adhesionFlexPlugin:
            sourceAdhesionVector = self.adhesionFlexPlugin.getAdhesionMoleculeDensityVector(sourceCell)
            self.adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(targetCell, sourceAdhesionVector)

        # PolarizationVector
        if self.polarizationVectorPlugin:
            sourcePolarizationVector = self.polarizationVectorPlugin.getPolarizationVector(sourceCell)
            self.polarizationVectorPlugin.setPolarizationVector(targetCell, sourcePolarizationVector[0],
                                                                sourcePolarizationVector[1],
                                                                sourcePolarizationVector[2])

        # polarization23Plugin
        if self.polarization23Plugin:
            polVec = self.polarization23Plugin.getPolarizationVector(sourceCell)
            self.polarization23Plugin.setPolarizationVector(targetCell, polVec)
            polMark = self.polarization23Plugin.getPolarizationMarkers(sourceCell)
            self.polarization23Plugin.setPolarizationMarkers(targetCell, polMark[0], polMark[1])
            l = self.polarization23Plugin.getLambdaPolarization(sourceCell)
            self.polarization23Plugin.setLambdaPolarization(targetCell, l)

        # CellOrientationPlugin
        if self.cellOrientationPlugin:
            l = self.cellOrientationPlugin.getLambdaCellOrientation(sourceCell)
            self.cellOrientationPlugin.setLambdaCellOrientation(targetCell, l)

        # ContactOrientationPlugin
        if self.contactOrientationPlugin:
            oVec = self.contactOrientationPlugin.getOriantationVector(sourceCell)
            self.contactOrientationPlugin.setOriantationVector(targetCell, oVec.x, oVec.y, oVec.z)
            self.contactOrientationPlugin.setAlpha(targetCell, self.contactOrientationPlugin.getAlpha(sourceCell))

        # ContactLocalProductPlugin
        if self.contactLocalProductPlugin:
            cVec = self.contactLocalProductPlugin.getCadherinConcentrationVec(sourceCell)
            self.contactLocalProductPlugin.setCadherinConcentrationVec(targetCell, cVec)

        # LengthConstraintPlugin
        if self.lengthConstraintPlugin:
            l = self.lengthConstraintPlugin.getLambdaLength(sourceCell)
            tl = self.lengthConstraintPlugin.getTargetLength(sourceCell)
            mtl = self.lengthConstraintPlugin.getMinorTargetLength(sourceCell)
            self.lengthConstraintPlugin.setLengthConstraintData(targetCell, l, tl, mtl)

        # ConnectivityGlobalPlugin
        if self.connectivityGlobalPlugin:
            cs = self.connectivityGlobalPlugin.getConnectivityStrength(sourceCell)
            self.connectivityGlobalPlugin.setConnectivityStrength(targetCell, cs)

        # ConnectivityLocalFlexPlugin
        if self.connectivityLocalFlexPlugin:
            cs = self.connectivityLocalFlexPlugin.getConnectivityStrength(sourceCell)
            self.connectivityLocalFlexPlugin.setConnectivityStrength(targetCell, cs)

        # Chemotaxis
        if self.chemotaxisPlugin:
            fieldNames = self.chemotaxisPlugin.getFieldNamesWithChemotaxisData(sourceCell)

            for fieldName in fieldNames:
                source_chd = self.chemotaxisPlugin.getChemotaxisData(sourceCell, fieldName)
                target_chd = self.chemotaxisPlugin.addChemotaxisData(targetCell, fieldName)

                target_chd.setLambda(source_chd.getLambda())
                target_chd.saturationCoef = source_chd.saturationCoef
                target_chd.setChemotaxisFormulaByName(source_chd.formulaName)
                target_chd.assignChemotactTowardsVectorTypes(source_chd.getChemotactTowardsVectorTypes())

                # FocalPointPLasticityPlugin - this plugin has to be handled manually - there is no good way to figure out which links shuold be copied from parent to daughter cell

    def get_accepted_pixel_copy_mask(self):
        num_en_calcs = int(self.en_calculator.get_number_energy_fcn_calculations())
        return self.en_calculator.get_current_mcs_accepted_mask_npy_array(num_en_calcs).astype(np.bool)

    def get_attempted_pixel_copy_prob_array(self):
        num_en_calcs = int(self.en_calculator.get_number_energy_fcn_calculations())
        return self.en_calculator.get_current_mcs_prob_npy_array(num_en_calcs)

    def get_attempted_pixel_copy_points(self):
        num_en_calcs = int(self.en_calculator.get_number_energy_fcn_calculations())
        return np.reshape(self.en_calculator.get_current_mcs_flip_attempt_points_npy_array(3 * num_en_calcs), (-1, 3))


class RunBeforeMCSSteppableBasePy(SteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        SteppableBasePy.__init__(self, _simulator, _frequency)
        self.runBeforeMCS = 1


class SecretionBasePy(SteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        SteppableBasePy.__init__(self, _simulator, _frequency)
        self.runBeforeMCS = 1


class DolfinSolverSteppable(RunBeforeMCSSteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        RunBeforeMCSSteppableBasePy.__init__(self, _simulator, _frequency)

        self.fieldDict = {}  # {fieldName:field}
        self.diffusableVector = None
        # self.createFields(['NEWFIELD_EXTRA','NEWFIELD']) # typically each class which inherits DolfinSolverSteppable will call createFields function

    def createFields(self, _fieldNameList, _registerAllFields=True):
        import CompuCell

        dimWithBorders = CompuCell.Dim3D()
        dimWithBorders.x = self.dim.x + 2
        dimWithBorders.y = self.dim.y + 2
        dimWithBorders.z = self.dim.z + 2

        self.diffusableVector = CompuCell.DiffusableVectorFloat()
        self.diffusableVector.allocateDiffusableFieldVector(len(_fieldNameList), dimWithBorders)
        nameVector = CompuCell.vectorstdstring()

        for fieldName in _fieldNameList:
            nameVector.push_back(fieldName)
        self.diffusableVector.setConcentrationFieldNameVector(nameVector)

        for fieldName in _fieldNameList:
            field = self.diffusableVector.getConcentrationField(fieldName)
            self.fieldDict[fieldName] = field

        if _registerAllFields:
            self.registerFields(_fieldNameList)

    def registerFields(self, _fieldNameList):
        for fieldName in _fieldNameList:
            try:
                field = self.fieldDict[fieldName]
                self.simulator.registerConcentrationField(fieldName, field)
            except LookupError:
                print 'DolfinSolverSteppable: COULD NOT FIND field=', fieldName

    def getStepFunctionExpressionFlex(self, _cellTypeToValueMap={}, _cellIdsToValueMap={},
                                      _defaultValue=0.0):  # this will create scalar expression
        import dolfin
        import dolfinCC3D
        # expressions defined in C++ have to be instantiated using quite complex class composition (uses metaclasses - see e.g. /usr/lib/python2.7/dist-packages/dolfin/functions/expression.py)
        # the way dolfin does it - it compiles C++ code including a class which inherits from  Expression. This class is then wrapped in Python using SWIG
        # however we cannot use this class in UFL expressions because it does not have required interface to be a part of the UFL (aka weak form) expression
        # there is a special functionwhich takes as an argument user-defined C++ class which inherits from Expression (wrapped in Python) and constructs a class which has the required interface (and of course includes all the function from the user defined C++ class)    

        # this is how create_compiled_expression_class looks like
        #         def create_compiled_expression_class(cpp_base):
        #             # Check the cpp_base
        #             assert(isinstance(cpp_base, (types.ClassType, type)))

        #             def __init__(self, cppcode, element=None, cell=None, \
        #                          degree=None, **kwargs):


        StepFunctionExpressionFlexClass = dolfin.functions.expression.create_compiled_expression_class(
            dolfinCC3D.StepFunctionExpressionFlex)  # this statement creates Class type (class template - see documentation on metaclasses)

        # notice that the constructor of  StepFunctionExpressionFlexClass has signatore of the __init__ function defined inside create_compiled_expression_class
        # we use it to construct the class and all remaining initialization will be done on the instance of this class
        stepFunctionExpressionFlexClassInstance = StepFunctionExpressionFlexClass('', None, None, None)
        stepFunctionExpressionFlexClassInstance.setCellField(self.cellField)
        stepFunctionExpressionFlexClassInstance.setStepFunctionValues(_cellTypeToValueMap, _cellIdsToValueMap,
                                                                      _defaultValue)

        return stepFunctionExpressionFlexClassInstance


import time


class SteppableRegistry(SteppablePy):
    def __init__(self):
        self.steppableList = []
        self.runBeforeMCSSteppableList = []
        self.steppableDict = {}  # {steppableClassName:[steppable inst0,steppable inst1,...]}
        from collections import defaultdict
        self.profiler_dict = defaultdict(lambda: defaultdict(float))  # {steppable_class_name:{object_hash:runtime}}

    def get_profiler_report(self):

        profiler_report = []

        for steppable_name, steppable_obj_dict in self.profiler_dict.iteritems():
            for steppable_obj_hash, run_time in steppable_obj_dict.iteritems():
                profiler_report.append([steppable_name, str(steppable_obj_hash), run_time])

        return profiler_report

    # def get_profiler_report(self):
    #     profiler_report = ''
    #
    #     from CompuCellSetup import convert_time_interval_to_hmsm
    #
    #     for steppable_name, steppable_obj_dict in self.profiler_dict.iteritems():
    #         for steppable_obj_hash, run_time in steppable_obj_dict.iteritems():
    #             profiler_report += steppable_name + '    ' +str(steppable_obj_hash) + ':    '\
    #                                + convert_time_interval_to_hmsm(run_time)
    #
    #
    #     return profiler_report


    def allSteppables(self):
        for steppable in self.steppableList:
            yield steppable
        for steppable in self.runBeforeMCSSteppableList:
            yield steppable

    def registerSteppable(self, _steppable):
        try:
            if _steppable.runBeforeMCS:
                self.runBeforeMCSSteppableList.append(_steppable)
            else:
                self.steppableList.append(_steppable)
        except AttributeError:
            self.steppableList.append(_steppable)

        # storing steppable in the dictionary
        try:
            self.steppableDict[_steppable.__class__.__name__].append(_steppable)
        except LookupError, e:
            self.steppableDict[_steppable.__class__.__name__] = [_steppable]

    def getSteppablesByClassName(self, _className):
        try:
            return self.steppableDict[_className]
        except LookupError, e:
            return []

    def init(self, _simulator):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.init(_simulator)

        for steppable in self.steppableList:
            steppable.init(_simulator)

    def extraInit(self, _simulator):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.extraInit(_simulator)

        for steppable in self.steppableList:
            steppable.extraInit(_simulator)


    def restart_steering_panel(self):
        """
        Function used during restart only to bring up the steering panel
        :return: None
        """
        # for steppable in self.runBeforeMCSSteppableList:
        #     # handling steering panel
        #     steppable.add_steering_panel()
        #
        # for steppable in self.steppableList:
        #     # handling steering panel
        #     steppable.add_steering_panel()
        try:
            import CompuCellSetup
            if len(CompuCellSetup.steering_param_dict.keys()):
                CompuCellSetup.addSteeringPanel(CompuCellSetup.steering_param_dict.values())
        except:
            print 'Could not create steering panel'

    def start(self):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.start()
            # handling steering panel
            steppable.add_steering_panel()
            if hasattr(steppable, 'initialize_automatic_tasks'):
                steppable.initialize_automatic_tasks()

            if hasattr(steppable, 'perform_automatic_tasks'):
                steppable.perform_automatic_tasks()

        for steppable in self.steppableList:
            steppable.start()
            # handling steering panel
            steppable.add_steering_panel()
            if hasattr(steppable, 'initialize_automatic_tasks'):
                steppable.initialize_automatic_tasks()
            if hasattr(steppable, 'perform_automatic_tasks'):
                steppable.perform_automatic_tasks()


        # handling steering panel
        try:
            import CompuCellSetup
            if len(CompuCellSetup.steering_param_dict.keys()):
                CompuCellSetup.addSteeringPanel(CompuCellSetup.steering_param_dict.values())
        except:
            print 'Could not create steering panel'

    def step(self, _mcs):

        for steppable in self.steppableList:

            if not _mcs % steppable.frequency:  # this executes given steppable every "frequency" Monte Carlo Steps

                try:
                    steppable.mcs = _mcs
                except AttributeError, e:
                    pass

                begin = time.time()

                steppable.step(_mcs)
                if hasattr(steppable, 'perform_automatic_tasks'):
                    steppable.perform_automatic_tasks()

                end = time.time()

                self.profiler_dict[steppable.__class__.__name__][hex(id(steppable))] += (end - begin) * 1000

                steppable.process_steering_panel_data_wrapper()


        # we reset dirty flag (indicates that a parameter was changed)
        # of the steering parameters via the call to the SteppableBasePy "set_steering_param_dirty"
        # function. At the moment we do not implement fine control over which parameters are dirty. For now if
        # any of the parameters is dirty we will rerun all the code that handles steering. this means
        # that we will process every single steering parameters as if it was dirty (i.e. changed recently)

        steppable_list_aggregate = self.steppableList + self.runBeforeMCSSteppableList
        # setting steering parameter dirty flag to False - this code runs at the end of the steppables call
        if len(steppable_list_aggregate):
            steppable_list_aggregate[0].set_steering_param_dirty(flag=False)

    def stepRunBeforeMCSSteppables(self, _mcs):

        for steppable in self.runBeforeMCSSteppableList:
            if not _mcs % steppable.frequency:  # this executes given steppable every "frequency" Monte Carlo Steps
                begin = time.time()

                steppable.step(_mcs)

                end = time.time()

                self.profiler_dict[steppable.__class__.__name__][hex(id(steppable))] += (end - begin) * 1000
                steppable.process_steering_panel_data_wrapper()

    def finish(self):
        for steppable in self.runBeforeMCSSteppableList:
            steppable.finish()

        for steppable in self.steppableList:
            steppable.finish()

    def cleanup(self):
        print 'inside cleanup'
        for steppable in self.runBeforeMCSSteppableList:
            steppable.cleanup()

        for steppable in self.steppableList:
            steppable.cleanup()

# IMPORTANT: It is best to always provide hand-written iterators for STL containers even though swig generates them for you.
# with multiple swig modules  those autogenerated iterators will work on one platform and crash on another ones so best solution is to write iterators yourself

# this is used to iterate more easily over cells

class CellList:
    def __init__(self, _inventory):
        self.inventory = _inventory

    def __iter__(self):
        return CellListIterator(self)

    def __len__(self):
        return int(self.inventory.getSize())


class CellListIterator:
    def __init__(self, _cellList):
        import CompuCell
        self.inventory = _cellList.inventory
        self.invItr = CompuCell.STLPyIteratorCINV()
        self.invItr.initialize(self.inventory.getContainer())
        self.invItr.setToBegin()

    def next(self):
        if not self.invItr.isEnd():
            self.cell = self.invItr.getCurrentRef()
            self.invItr.next()
            return self.cell
        else:
            raise StopIteration

    def __iter__(self):
        return self


# iterating ofver inventory of cells of a given type
class CellListByType:
    def __init__(self, _inventory, *args):
        import CompuCell
        self.inventory = _inventory

        self.types = CompuCell.vectorint()

        self.inventoryByType = CompuCell.mapLongCellGPtr()

        self.initTypeVec(args)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType, self.types)

    def __iter__(self):
        return CellListByTypeIterator(self)

    def __call__(self, *args):
        self.initTypeVec(args)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType, self.types)

        return self

    def __len__(self):
        return int(self.inventoryByType.size())

    def initTypeVec(self, _typeList):

        self.types.clear()
        if len(_typeList) <= 0:
            self.types.push_back(1)  # type 1
        else:
            for type in _typeList:
                self.types.push_back(type)

    def initializeWithType(self, _type):
        self.types.clear()
        self.types.push_back(_type)
        self.inventory.initCellInventoryByMultiType(self.inventoryByType, self.types)

    def refresh(self):
        self.inventory.initCellInventoryByMultiType(self.inventoryByType, self.types)


class CellListByTypeIterator:
    def __init__(self, _cellListByType):
        import CompuCell
        self.inventoryByType = _cellListByType.inventoryByType

        self.invItr = CompuCell.mapLongCellGPtrPyItr()
        self.invItr.initialize(self.inventoryByType)
        self.invItr.setToBegin()

    def next(self):
        if not self.invItr.isEnd():
            self.cell = self.invItr.getCurrentRef()
            # print 'self.idCellPair=',self.idCellPair
            # print 'dir(self.idCellPair)=',dir(self.idCellPair)
            self.invItr.next()
            return self.cell
        #
        else:
            raise StopIteration

    def __iter__(self):
        return self

        # this is used to iterate more easily over clusters


class ClusterList:
    def __init__(self, _inventory):
        self.inventory = _inventory.getClusterInventory().getContainer()

    def __iter__(self):
        return ClusterListIterator(self)

    def __len__(self):
        return int(self.inventory.size())


class ClusterListIterator:
    def __init__(self, _cellList):
        import CompuCell
        self.inventory = _cellList.inventory

        self.invItr = CompuCell.compartmentinventoryPtrPyItr()
        self.invItr.initialize(self.inventory)
        self.invItr.setToBegin()

        self.compartmentList = None

    def next(self):

        if not self.invItr.isEnd():
            self.compartmentList = self.invItr.getCurrentRef()
            # print 'self.idCellPair=',self.idCellPair
            # print 'dir(self.idCellPair)=',dir(self.idCellPair)
            self.invItr.next()
            return self.compartmentList
        #
        else:
            raise StopIteration


# this is used to iterate more easily over clusters and avoid strange looking Python syntax

class Clusters:
    def __init__(self, _inventory):
        self.inventory = _inventory.getClusterInventory().getContainer()

    def __iter__(self):
        return ClustersIterator(self)

    def __len__(self):
        return int(self.inventory.size())


class ClustersIterator:
    def __init__(self, _cellList):
        import CompuCell
        self.inventory = _cellList.inventory

        self.invItr = CompuCell.compartmentinventoryPtrPyItr()
        self.invItr.initialize(self.inventory)
        self.invItr.setToBegin()

        self.compartmentList = None

    def next(self):

        if not self.invItr.isEnd():
            self.compartmentList = self.invItr.getCurrentRef()
            # print 'self.idCellPair=',self.idCellPair
            # print 'dir(self.idCellPair)=',dir(self.idCellPair)
            self.invItr.next()
            return CompartmentList(self.compartmentList)
        #
        else:
            raise StopIteration


# this is used to iterate more easily over list of compartments , notice regular map iteration will work too but this is more abstracted out and will work with other containers too

class CompartmentList:
    def __init__(self, _inventory):
        import CompuCell
        self.inventory = _inventory

    def __iter__(self):
        return CompartmentListIterator(self)

    def __len__(self):
        return int(self.inventory.size())

    def clusterId(self):
        return self.__iter__().next().clusterId


class CompartmentListIterator:
    def __init__(self, _cellList):
        import CompuCell
        self.inventory = _cellList.inventory

        self.invItr = CompuCell.mapLongCellGPtrPyItr()
        self.invItr.initialize(self.inventory)
        self.invItr.setToBegin()

    def next(self):
        if not self.invItr.isEnd():
            self.cell = self.invItr.getCurrentRef()
            # print 'self.idCellPair=',self.idCellPair
            # print 'dir(self.idCellPair)=',dir(self.idCellPair)
            self.invItr.next()
            return self.cell
        #
        else:
            raise StopIteration

    def __iter__(self):
        return self

    # this is wrapper for std::vector<CellG*>


class ClusterCellList:
    def __init__(self, _inventory):
        self.inventory = _inventory

    def __iter__(self):
        return ClusterCellListIterator(self)

    def __len__(self):
        return int(self.inventory.size())


class ClusterCellListIterator:
    def __init__(self, _cellList):
        import CompuCell
        self.inventory = _cellList.inventory
        # print "dir(self.inventory)=",dir(self.inventory)        
        self.currentIdx = 0
        self.cell = None
        # self.invItr.initialize(self.inventory.getContainer())
        # self.invItr.setToBegin()

    def next(self):
        # if self.invItr !=  self.inventory.end():
        if self.currentIdx < self.inventory.size():
            # print "self.invItr=",dir(self.invItr)
            # print "self.invItr.next()=",self.invItr.next()
            # self.compartmentList = self.invItr.next()



            self.cell = self.inventory[self.currentIdx]
            self.currentIdx += 1

            return self.cell
        else:
            raise StopIteration

    def __iter__(self):
        return self

    # legacy code  - do not modify


class CellNeighborList:
    def __init__(self, _neighborTrackerAccessor, _cell):
        self.neighborTrackerAccessor = _neighborTrackerAccessor
        self.cell = _cell

    def __iter__(self):
        return CellNeighborIterator(self)


class CellNeighborIterator:
    def __init__(self, _cellNeighborList):
        import CompuCell
        self.neighborTrackerAccessor = _cellNeighborList.neighborTrackerAccessor
        self.cell = _cellNeighborList.cell
        self.nsdItr = CompuCell.nsdSetPyItr()
        self.nTracker = self.neighborTrackerAccessor.get(self.cell.extraAttribPtr)
        self.nsdItr.initialize(self.nTracker.cellNeighbors)
        self.nsdItr.setToBegin()

    def next(self):
        if not self.nsdItr.isEnd():
            self.neighborCell = self.nsdItr.getCurrentRef().neighborAddress
            self.nsdItr.next()
            return self.neighborCell
        else:
            raise StopIteration

    def __iter__(self):
        return self


class CellNeighborListAuto:
    def __init__(self, _neighborPlugin, _cell):
        self.neighborPlugin = _neighborPlugin
        self.neighborTrackerAccessor = self.neighborPlugin.getNeighborTrackerAccessorPtr()
        self.cell = _cell

    def __iter__(self):
        return CellNeighborIteratorAuto(self)


class CellNeighborIteratorAuto:
    def __init__(self, _cellNeighborList):
        import CompuCell
        self.neighborTrackerAccessor = _cellNeighborList.neighborTrackerAccessor
        self.cell = _cellNeighborList.cell
        self.nsdItr = CompuCell.nsdSetPyItr()
        self.nTracker = self.neighborTrackerAccessor.get(self.cell.extraAttribPtr)
        self.nsdItr.initialize(self.nTracker.cellNeighbors)
        self.nsdItr.setToBegin()

    def next(self):
        if not self.nsdItr.isEnd():
            self.neighborCell = self.nsdItr.getCurrentRef().neighborAddress
            self.currentNsdItr = self.nsdItr.current
            self.currentNeighborSurfaceData = self.nsdItr.getCurrentRef()
            self.nsdItr.next()
            return self.currentNeighborSurfaceData
        else:
            raise StopIteration

    def __iter__(self):
        return self


# end of legacy code  - do not modify              

class CellNeighborListFlex:
    def __init__(self, _neighborPlugin, _cell):
        self.neighborPlugin = _neighborPlugin
        self.neighborTrackerAccessor = self.neighborPlugin.getNeighborTrackerAccessorPtr()
        self.cell = _cell

    def __len__(self):
        neighborTracker = self.neighborTrackerAccessor.get(self.cell.extraAttribPtr)
        return int(neighborTracker.cellNeighbors.size())

    def __getitem__(self, idx):
        if idx > self.__len__() - 1: raise IndexError(
            "Out of bounds index: CellNeighborListAuto index = %s is out of bounds" % str(idx))
        for counter, data in enumerate(self.__iter__()):
            if idx == counter: return data

    def commonSurfaceAreaWithCellTypes(self, cell_type_list):
        area = 0
        for neighbor, commonSurfaceArea in self.__iter__():
            cell_type = 0 if not neighbor else neighbor.type
            if cell_type in cell_type_list:
                area += commonSurfaceArea
        return area

    def commonSurfaceAreaByType(self):
        from collections import defaultdict
        area_dict = defaultdict(int)
        for neighbor, commonSurfaceArea in self.__iter__():
            cell_type = 0 if not neighbor else neighbor.type
            area_dict[cell_type] += commonSurfaceArea
        return area_dict

    def neighborCountByType(self):
        from collections import defaultdict
        neighbor_counter_dict = defaultdict(int)

        for neighbor, commonSurfaceArea in self.__iter__():
            cell_type = 0 if not neighbor else neighbor.type
            neighbor_counter_dict[cell_type] += 1
        return neighbor_counter_dict

    def __iter__(self):
        return CellNeighborIteratorFlex(self)


class CellNeighborIteratorFlex:
    def __init__(self, _cellNeighborList):
        import CompuCell
        self.neighborTrackerAccessor = _cellNeighborList.neighborTrackerAccessor
        self.cell = _cellNeighborList.cell
        self.nsdItr = CompuCell.nsdSetPyItr()
        self.nTracker = self.neighborTrackerAccessor.get(self.cell.extraAttribPtr)
        self.nsdItr.initialize(self.nTracker.cellNeighbors)
        self.nsdItr.setToBegin()

    def next(self):
        if not self.nsdItr.isEnd():
            self.neighborCell = self.nsdItr.getCurrentRef().neighborAddress
            self.currentNsdItr = self.nsdItr.current
            self.currentNeighborSurfaceData = self.nsdItr.getCurrentRef()
            self.nsdItr.next()
            # return self.currentNeighborSurfaceData.neighborAddress,self.currentNeighborSurfaceData.commonSurfaceArea
            return self.currentNeighborSurfaceData.neighborAddress, self.currentNeighborSurfaceData.commonSurfaceArea
        else:
            raise StopIteration

    def __iter__(self):
        return self


class CellBoundaryPixelList:
    def __init__(self, _boundaryPixelTrackerPlugin, _cell, _neighborOrder=-1):
        self.neighborOrder = _neighborOrder
        self.boundaryPixelTrackerPlugin = _boundaryPixelTrackerPlugin
        self.boundaryPixelTrackerAccessor = self.boundaryPixelTrackerPlugin.getBoundaryPixelTrackerAccessorPtr()
        self.cell = _cell

    def __iter__(self):
        return CellBoundaryPixelIterator(self, self.neighborOrder)

    def numberOfPixels(self):
        return self.boundaryPixelTrackerAccessor.get(self.cell.extraAttribPtr).pixelSet.size()


class CellBoundaryPixelIterator:
    def __init__(self, _cellPixelList, _neighborOrder=-1):
        import CompuCell
        self.boundaryPixelTrackerAccessor = _cellPixelList.boundaryPixelTrackerAccessor
        self.boundaryPixelTrackerPlugin = _cellPixelList.boundaryPixelTrackerPlugin
        self.cell = _cellPixelList.cell
        self.boundaryPixelItr = CompuCell.boundaryPixelSetPyItr()
        self.boundaryPixelTracker = self.boundaryPixelTrackerAccessor.get(self.cell.extraAttribPtr)
        # print '_neighborOrder=',_neighborOrder
        if _neighborOrder <= 0:
            self.pixelSet = self.boundaryPixelTracker.pixelSet
        else:
            self.pixelSet = self.boundaryPixelTrackerPlugin.getPixelSetForNeighborOrderPtr(self.cell, _neighborOrder)
            if not self.pixelSet:
                raise LookupError(
                    'LookupError: CellBoundaryPixelIterator could not locate pixel set for neighbor order = %s. Make sure your BoundaryPixelTracker plugin definition requests tracking of neighbor order =%s boundary' % (
                    _neighborOrder, _neighborOrder))

        # self.boundaryPixelItr.initialize(self.boundaryPixelTracker.pixelSet)
        self.boundaryPixelItr.initialize(self.pixelSet)
        self.boundaryPixelItr.setToBegin()

    def next(self):
        if not self.boundaryPixelItr.isEnd():
            #             self.neighborCell = self.nsdItr.getCurrentRef().neighborAddress
            #             self.currentNsdItr = self.nsdItr.current
            self.currentBoundaryPixelTrackerData = self.boundaryPixelItr.getCurrentRef()
            self.boundaryPixelItr.next()
            return self.boundaryPixelTrackerPlugin.getBoundaryPixelTrackerData(self.currentBoundaryPixelTrackerData)
        #             return self.currentPixelTrackerData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class CellPixelList:
    def __init__(self, _pixelTrackerPlugin, _cell):
        self.pixelTrackerPlugin = _pixelTrackerPlugin
        self.pixelTrackerAccessor = self.pixelTrackerPlugin.getPixelTrackerAccessorPtr()
        self.cell = _cell

    def __iter__(self):
        return CellPixelIterator(self)

    def numberOfPixels(self):
        return self.pixelTrackerAccessor.get(self.cell.extraAttribPtr).pixelSet.size()


class CellPixelIterator:
    def __init__(self, _cellPixelList):
        import CompuCell
        self.pixelTrackerAccessor = _cellPixelList.pixelTrackerAccessor
        self.pixelTrackerPlugin = _cellPixelList.pixelTrackerPlugin
        self.cell = _cellPixelList.cell
        self.pixelItr = CompuCell.pixelSetPyItr()
        self.pixelTracker = self.pixelTrackerAccessor.get(self.cell.extraAttribPtr)
        self.pixelItr.initialize(self.pixelTracker.pixelSet)
        self.pixelItr.setToBegin()

    def next(self):
        if not self.pixelItr.isEnd():
            #             self.neighborCell = self.nsdItr.getCurrentRef().neighborAddress
            #             self.currentNsdItr = self.nsdItr.current
            self.currentPixelTrackerData = self.pixelItr.getCurrentRef()
            self.pixelItr.next()

            return self.pixelTrackerPlugin.getPixelTrackerData(self.currentPixelTrackerData)
            # return self.currentPixelTrackerData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class ElasticityDataList:
    def __init__(self, _elasticityTrackerPlugin, _cell):
        self.elasticityTrackerPlugin = _elasticityTrackerPlugin
        self.elasticityTrackerAccessor = self.elasticityTrackerPlugin.getElasticityTrackerAccessorPtr()
        self.cell = _cell

    def __iter__(self):
        return ElasticityDataIterator(self)


class ElasticityDataIterator:
    def __init__(self, _elasticityDataList):
        import CompuCell
        self.elasticityTrackerAccessor = _elasticityDataList.elasticityTrackerAccessor
        self.cell = _elasticityDataList.cell
        self.elasticityTrackerPlugin = _elasticityDataList.elasticityTrackerPlugin
        self.elasticityTracker = self.elasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        self.elasticityDataSetItr = CompuCell.elasticitySetPyItr()
        self.elasticityDataSetItr.initialize(self.elasticityTracker.elasticityNeighbors)
        self.elasticityDataSetItr.setToBegin()

    def next(self):
        if not self.elasticityDataSetItr.isEnd():
            self.currentElasticityDataSetItr = self.elasticityDataSetItr.current
            self.elasticityData = self.elasticityDataSetItr.getCurrentRef()
            self.elasticityDataSetItr.next()
            return self.elasticityTrackerPlugin.getElasticityTrackerData(self.elasticityData)
        #             return self.elasticityData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class PlasticityDataList:
    def __init__(self, _plasticityTrackerPlugin, _cell):
        self.plasticityTrackerPlugin = _plasticityTrackerPlugin
        self.plasticityTrackerAccessor = self.plasticityTrackerPlugin.getPlasticityTrackerAccessorPtr()
        self.cell = _cell

    def __iter__(self):
        return PlasticityDataIterator(self)


class PlasticityDataIterator:
    def __init__(self, _plasticityDataList):
        import CompuCell
        self.plasticityTrackerAccessor = _plasticityDataList.plasticityTrackerAccessor
        self.cell = _plasticityDataList.cell
        self.plasticityTrackerPlugin = _plasticityDataList.plasticityTrackerPlugin
        self.plasticityTracker = self.plasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        self.plasticityDataSetItr = CompuCell.plasticitySetPyItr()
        self.plasticityDataSetItr.initialize(self.plasticityTracker.plasticityNeighbors)
        self.plasticityDataSetItr.setToBegin()

    def next(self):
        if not self.plasticityDataSetItr.isEnd():
            self.currentPlasticityDataSetItr = self.plasticityDataSetItr.current
            self.plasticityData = self.plasticityDataSetItr.getCurrentRef()
            self.plasticityDataSetItr.next()
            return self.plasticityTrackerPlugin.getPlasticityTrackerData(self.plasticityData)
        #             return self.plasticityData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class FocalPointPlasticityDataList:
    def __init__(self, _focalPointPlasticityPlugin, _cell):
        self.focalPointPlasticityPlugin = _focalPointPlasticityPlugin
        self.focalPointPlasticityTrackerAccessor = self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerAccessorPtr()
        self.cell = _cell

    def __len__(self):
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        return int(self.focalPointPlasticityTracker.focalPointPlasticityNeighbors.size())

    def __getitem__(self, idx):
        if idx > self.__len__() - 1: raise IndexError(
            "Out of bounds index: FocalPointPlasticityDataList index = %s is out of bounds" % str(idx))
        for counter, data in enumerate(self.__iter__()):
            if idx == counter: return data

    def __iter__(self):
        return FocalPointPlasticityDataIterator(self)


class FocalPointPlasticityDataIterator:
    def __init__(self, _focalPointPlasticityDataList):
        import CompuCell
        self.focalPointPlasticityTrackerAccessor = _focalPointPlasticityDataList.focalPointPlasticityTrackerAccessor
        self.cell = _focalPointPlasticityDataList.cell
        self.focalPointPlasticityPlugin = _focalPointPlasticityDataList.focalPointPlasticityPlugin
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        self.focalPointPlasticityDataSetItr = CompuCell.focalPointPlasticitySetPyItr()
        self.focalPointPlasticityDataSetItr.initialize(self.focalPointPlasticityTracker.focalPointPlasticityNeighbors)
        self.focalPointPlasticityDataSetItr.setToBegin()

    def next(self):
        if not self.focalPointPlasticityDataSetItr.isEnd():
            self.currentFocalPointPlasticityDataSetItr = self.focalPointPlasticityDataSetItr.current
            self.focalPointPlasticityData = self.focalPointPlasticityDataSetItr.getCurrentRef()
            self.focalPointPlasticityDataSetItr.next()
            return self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerData(self.focalPointPlasticityData)
        #             return self.plasticityData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class InternalFocalPointPlasticityDataList:
    def __init__(self, _focalPointPlasticityPlugin, _cell):
        self.focalPointPlasticityPlugin = _focalPointPlasticityPlugin
        self.focalPointPlasticityTrackerAccessor = self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerAccessorPtr()
        self.cell = _cell

    def __getitem__(self, idx):
        if idx > self.__len__() - 1: raise IndexError(
            "Out of bounds index: InternalFocalPointPlasticityDataList index = %s is out of bounds" % str(idx))
        for counter, data in enumerate(self.__iter__()):
            if idx == counter: return data

    def __len__(self):
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        return int(self.focalPointPlasticityTracker.internalFocalPointPlasticityNeighbors.size())

    def __iter__(self):
        return InternalFocalPointPlasticityDataIterator(self)


class InternalFocalPointPlasticityDataIterator:
    def __init__(self, _focalPointPlasticityDataList):
        import CompuCell
        self.focalPointPlasticityTrackerAccessor = _focalPointPlasticityDataList.focalPointPlasticityTrackerAccessor
        self.cell = _focalPointPlasticityDataList.cell
        self.focalPointPlasticityPlugin = _focalPointPlasticityDataList.focalPointPlasticityPlugin
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        self.focalPointPlasticityDataSetItr = CompuCell.focalPointPlasticitySetPyItr()
        self.focalPointPlasticityDataSetItr.initialize(
            self.focalPointPlasticityTracker.internalFocalPointPlasticityNeighbors)
        self.focalPointPlasticityDataSetItr.setToBegin()

    def next(self):
        if not self.focalPointPlasticityDataSetItr.isEnd():
            self.currentFocalPointPlasticityDataSetItr = self.focalPointPlasticityDataSetItr.current
            self.focalPointPlasticityData = self.focalPointPlasticityDataSetItr.getCurrentRef()
            self.focalPointPlasticityDataSetItr.next()
            return self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerData(self.focalPointPlasticityData)
        #             return self.plasticityData
        else:
            raise StopIteration

    def __iter__(self):
        return self


class AnchorFocalPointPlasticityDataList:
    def __init__(self, _focalPointPlasticityPlugin, _cell):
        self.focalPointPlasticityPlugin = _focalPointPlasticityPlugin
        self.focalPointPlasticityTrackerAccessor = self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerAccessorPtr()
        self.cell = _cell

    def __getitem__(self, idx):
        if idx > self.__len__() - 1: raise IndexError(
            "Out of bounds index: AnchorFocalPointPlasticityDataList index = %s is out of bounds" % str(idx))
        for counter, data in enumerate(self.__iter__()):
            if idx == counter: return data

    def __len__(self):
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        return int(self.focalPointPlasticityTracker.anchors.size())

    def __iter__(self):
        return AnchorFocalPointPlasticityDataIterator(self)


class AnchorFocalPointPlasticityDataIterator:
    def __init__(self, _focalPointPlasticityDataList):
        import CompuCell
        self.focalPointPlasticityTrackerAccessor = _focalPointPlasticityDataList.focalPointPlasticityTrackerAccessor
        self.cell = _focalPointPlasticityDataList.cell
        self.focalPointPlasticityPlugin = _focalPointPlasticityDataList.focalPointPlasticityPlugin
        self.focalPointPlasticityTracker = self.focalPointPlasticityTrackerAccessor.get(self.cell.extraAttribPtr)
        self.focalPointPlasticityDataSetItr = CompuCell.focalPointPlasticitySetPyItr()
        self.focalPointPlasticityDataSetItr.initialize(self.focalPointPlasticityTracker.anchors)
        self.focalPointPlasticityDataSetItr.setToBegin()

    def next(self):
        if not self.focalPointPlasticityDataSetItr.isEnd():
            self.currentFocalPointPlasticityDataSetItr = self.focalPointPlasticityDataSetItr.current
            self.focalPointPlasticityData = self.focalPointPlasticityDataSetItr.getCurrentRef()
            self.focalPointPlasticityDataSetItr.next()
            return self.focalPointPlasticityPlugin.getFocalPointPlasticityTrackerData(self.focalPointPlasticityData)
        #             return self.plasticityData
        else:
            raise StopIteration

    def __iter__(self):
        return self

    # forEachCellInInventory function takes as arguments inventory of cells and a function that will operate on a single cell


# It will run singleCellOperation on each cell from cell inventory
def forEachCellInInventory(inventory, singleCellOperation):
    import CompuCell
    invItr = CompuCell.STLPyIteratorCINV()
    invItr.initialize(inventory.getContainer())
    invItr.setToBegin()
    cell = invItr.getCurrentRef()
    while (not invItr.isEnd()):
        cell = invItr.getCurrentRef()
        singleCellOperation(cell)
        invItr.next()
