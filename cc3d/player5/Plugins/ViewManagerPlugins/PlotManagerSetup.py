plots_available = False
try:
    import numpy
    import pyqtgraph
    from .PlotManager import PlotManager
    plots_available = True

except ImportError as e:
    plots_available = False
    print( 'Could not create PlotManager. '
           'Resorting to PLotManagerBase - plots will not work properly. Here is the exception:')
    print(e)

class PlotManagerBase:
    def __init__(self, _viewManager=None, _plotSupportFlag=False):
        self.vm = _viewManager
        self.plotsSupported = plots_available

    def initSignalAndSlots(self):
        pass

    def getPlotWindow(self):
        pass

    def reset(self):
        pass

    def getNewPlotWindow(self):
        pass

    def processRequestForNewPlotWindow(self, _mutex):
        pass

    def restore_plots_layout(self):
        pass

    def getPlotWindowsLayoutDict(self):
        return {}


# called from SimpleTabView
def createPlotManager(_viewManager=None):
    """

    @param _viewManager:instance of viewManager
    @param preferred_manager_type: not used in currnt API in the future versions
    it will be used to instantiate preferred plotting backend
    @return: instance of PlotManagerBase (inherited)
    """
    if plots_available:
        return PlotManager(_viewManager, True)
    else:
        return PlotManagerBase(_viewManager, False)


