from cc3d import CompuCellSetup
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


def add_new_plot_window(_title='', _xAxisTitle='', _yAxisTitle='', _xScaleType='linear', _yScaleType='linear', _grid=True, _config_options=None):

    view_manager = CompuCellSetup.persistent_globals.view_manager

    if not view_manager:
        pwd = PlotWindowDummy()
        return pwd

    # qt_version = 4
    try:
        pW = view_manager.plotManager.getNewPlotWindow()
    except:
        full_options_dict = {
            'title':_title,
            'xAxisTitle':_xAxisTitle,
            'yAxisTitle':_yAxisTitle,
            'xScaleType':_xScaleType,
            'yScaleType':_yScaleType,
            'grid':_grid
        }
        if _config_options:
            full_options_dict.update(_config_options)

        pW = view_manager.plotManager.getNewPlotWindow(full_options_dict)

        # pW = viewManager.plotManager.getNewPlotWindow({
        #     'title':_title,
        #     'xAxisTitle':_xAxisTitle,
        #     'yAxisTitle':_yAxisTitle,
        #     'xScaleType':_xScaleType,
        #     'yScaleType':_yScaleType,
        #     'grid':_grid
        # }
        # )
        qt_version = 5

    if not pW:
        raise AttributeError(
            'Missing plot modules. Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads. Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages')

    # # setting up default plot window parameters/look
    # if qt_version==4:
    #     # Plot Title - properties
    #     pW.setTitle(_title)
    #     pW.setTitleSize(12)
    #     pW.setTitleColor("Green")
    #
    #     # plot background
    #     pW.setPlotBackgroundColor("white")
    #     # properties of x axis
    #     pW.setXAxisTitle(_xAxisTitle)
    #     if _xScaleType == 'log':
    #         pW.setXAxisLogScale()
    #     pW.setXAxisTitleSize(10)
    #     pW.setXAxisTitleColor("blue")
    #
    #     # properties of y axis
    #     pW.setYAxisTitle(_yAxisTitle)
    #     if _xScaleType == 'log':
    #         pW.setYAxisLogScale()
    #     pW.setYAxisTitleSize(10)
    #     pW.setYAxisTitleColor("red")
    #
    #     pW.addGrid()
    #     # adding automatically generated legend
    #     # default possition is at the bottom of the plot but here we put it at the top
    #     pW.addAutoLegend("top")
    #
    #     # restoring plot window - have to decide whether to keep it or rely on viewManager.plotManager restore_plots_layout function
    #     #     viewManager.plotManager.restoreSingleWindow(pW)

    return pW
