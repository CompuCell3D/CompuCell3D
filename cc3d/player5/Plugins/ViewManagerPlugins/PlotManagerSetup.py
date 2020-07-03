from .PlotManagerBase import PlotManagerBase

plots_available = False
try:
    import numpy
    import pyqtgraph
    from .PlotManager import PlotManager

    plots_available = True

except ImportError as e:
    plots_available = False
    print('Could not create PlotManager. '
          'Resorting to PLotManagerBase - plots will not work properly. Here is the exception:')
    print(e)


# called from SimpleTabView
def create_plot_manager(view_manager=None):
    """

    :param view_manager: instance of viewManager
    :return:
    """
    if plots_available:
        return PlotManager(view_manager, True)
    else:
        return PlotManagerBase(view_manager, False)
