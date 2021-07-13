from weakref import ref


class PlotManagerBase:
    def __init__(self, view_manager=None, plot_support_flag=False):
        self.vm = view_manager
        self.plotsSupported = plot_support_flag

    @property
    def vm(self):
        try:
            o = self._vm()
        except TypeError:
            o = self._vm
        return o

    @vm.setter
    def vm(self, _i):
        try:
            self._vm = ref(_i)
        except TypeError:
            self._vm = _i

    def init_signal_and_slots(self):
        pass

    def get_plot_window(self):
        pass

    def reset(self):
        pass

    def get_new_plot_window(self):
        pass

    def process_request_for_new_plot_window(self, _mutex, obj):
        pass

    def restore_plots_layout(self):
        pass

    def get_plot_windows_layout_dict(self):
        return {}
