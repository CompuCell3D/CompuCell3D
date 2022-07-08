from cc3d import CompuCellSetup
from cc3d.core.ExtraFieldAdapter import ExtraFieldAdapter


class PlotWindowDummy(object):
    """
    This class serves as a dummy object that is used when viewManager is None
    It "emulates" plotWindow API but does nothing except ensures that
    in the simulation that does not use CC3D Player (ciew manager in such simulation is set to None)
    any call to plotWindow object gets captured/intercepted by __getattr__ function which returns a dummy
    method that accepts any
    type of arguments and has ampty body. This way any calls to plots defined in the simulation that runs in
    the console are gently
    ignored and simulation runs just fine. Most importantly it does not require plots to be commented out in
    the command line applications
    """

    def __init__(self):
        pass

    def __getattr__(self, attr):
        return self.dummyFcn

    def dummyFcn(self, *args, **kwds):
        pass


def add_new_plot_window(title='', xAxisTitle='', yAxisTitle='', xScaleType='linear', yScaleType='linear', grid=True,
                        config_options=None):
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
        plot_window = view_manager.plotManager.get_new_plot_window()
    except:
        full_options_dict = {
            'title': title,
            'xAxisTitle': xAxisTitle,
            'yAxisTitle': yAxisTitle,
            'xScaleType': xScaleType,
            'yScaleType': yScaleType,
            'grid': grid
        }
        if config_options:
            full_options_dict.update(config_options)

        plot_window = view_manager.plotManager.get_new_plot_window(full_options_dict)

    if not plot_window:
        raise AttributeError(
            'Missing plot modules. Windows/OSX Users: Make sure you have numpy installed. '
            'For instructions please visit www.compucell3d.org/Downloads. '
            'Linux Users: Make sure you have numpy and PyQwt installed. '
            'Please consult your linux distribution manual pages on how to best install those packages')

    return plot_window


def add_new_message_window(title=""):
    """
    opens up new popup window if simulation is executed using player
    """

    view_manager = CompuCellSetup.persistent_globals.view_manager

    if not view_manager:
        # we are using PlotWindowDummy as an object that "absorbs all calls" but does nothing - when we detect we
        # are not using player
        pwd = PlotWindowDummy()
        return pwd
    specs = dict(title=title)
    popup_window = view_manager.popup_window_manager.get_new_popup_window(specs=specs)

    return popup_window


def create_extra_field(field_name: str, field_type: int) -> ExtraFieldAdapter:
    """
    Creates field adapter. On initialization it may or may not have functional reference to the actual field
    When field is initialized from constructor only adapter is returned, however fields
    initialized later in the simulation (start or step function ) will have functional field reference inside
    :param field_name:
    :param field_type:
    :return:
    """
    field_name = field_name.replace(" ", "_")
    persistent_globals = CompuCellSetup.persistent_globals
    field_registry = persistent_globals.field_registry

    field_adapter = field_registry.create_field(field_name, field_type)

    return field_adapter
