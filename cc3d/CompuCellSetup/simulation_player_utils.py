from cc3d import CompuCellSetup
import numpy as np
from cc3d.core.enums import *



class PlotWindowDummy(object):
    '''
    This class serves as a dummy object that is used when viewManager is None
    It "emulates" plotWindow API but does nothing except ensures that
    in the simulation that does not use CC3D Player (ciew manager in such simulation is set to None)
    any call to plotWindow object gets captured/intercepted by __getattr__ function which returns a dummy method that accepts any
    type of arguments and has ampty body. This way any calls to plots defined in the simulation that runs in the console are gently
    ignored and simulation runs just fine. Most importantly it does not require plots to be commented out int he command line applications
    '''

    def __init__(self):
        pass

    def __getattr__(self, attr):
        return self.dummyFcn

    def dummyFcn(self, *args, **kwds):
        pass


def add_new_plot_window(title='', xAxisTitle='', yAxisTitle='', xScaleType='linear', yScaleType='linear', grid=True, config_options=None):
    """
    Adds new plot window to the player
    :param title:
    :param xAxisTitle:
    :param yAxisTitle:
    :param xScaleType:
    :param yScaleType:
    :param grid:
    :param config_options:
    :return:
    """

    view_manager = CompuCellSetup.persistent_globals.view_manager

    if not view_manager:
        pwd = PlotWindowDummy()
        return pwd

    try:
        pW = view_manager.plotManager.getNewPlotWindow()
    except:
        full_options_dict = {
            'title':title,
            'xAxisTitle':xAxisTitle,
            'yAxisTitle':yAxisTitle,
            'xScaleType':xScaleType,
            'yScaleType':yScaleType,
            'grid':grid
        }
        if config_options:
            full_options_dict.update(config_options)

        pW = view_manager.plotManager.getNewPlotWindow(full_options_dict)


    if not pW:
        raise AttributeError(
            'Missing plot modules. Windows/OSX Users: Make sure you have numpy installed. '
            'For instructions please visit www.compucell3d.org/Downloads. '
            'Linux Users: Make sure you have numpy and PyQwt installed. '
            'Please consult your linux distributioun manual pages on how to best install those packages')

    return pW

# def create_scalar_field_py(dim ,fieldName):
def create_scalar_field_py(field_name):


    field_name = field_name.replace(" ", "_")
    persistent_globals = CompuCellSetup.persistent_globals
    simthread = persistent_globals.simthread
    field_registry = persistent_globals.field_registry

    # field_adapter = field_registry.schedule_field_creation(field_name, SCALAR_FIELD_NPY)
    field_adapter = field_registry.create_field(field_name, SCALAR_FIELD_NPY)


    return field_adapter



    # todo 5 - fix it to handle CML runs
    # if persistent_globals.simthread == "CML":
    #
    #     fieldNP = np.zeros(shape=(_dim.x, _dim.y, _dim.z), dtype=np.float32)
    #     ndarrayAdapter = cmlFieldHandler.fieldStorage.createFloatFieldPy(_dim, _fieldName)
    #     ndarrayAdapter.initFromNumpy(
    #         fieldNP)  # initializing  numpyAdapter using numpy array (copy dims and data ptr)
    #     fieldRegistry.addNewField(ndarrayAdapter, _fieldName, SCALAR_FIELD)
    #     fieldRegistry.addNewField(fieldNP, _fieldName + '_npy', SCALAR_FIELD_NPY)
    #     return fieldNP
    # else:

    fieldNP = np.zeros(shape=(dim.x, dim.y, dim.z), dtype=np.float32)
    ndarrayAdapter = simthread.callingWidget.fieldStorage.createFloatFieldPy(dim, field_name)
    # initializing  numpyAdapter using numpy array (copy dims and data ptr)
    ndarrayAdapter.initFromNumpy(fieldNP)
    field_registry.addNewField(ndarrayAdapter, field_name, SCALAR_FIELD)
    field_registry.addNewField(fieldNP, field_name + '_npy', SCALAR_FIELD_NPY)
    return fieldNP
