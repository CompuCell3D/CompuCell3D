from cc3d.core.PySteppables import *


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.clear_flag = False

    def start(self):
        self.add_steering_param(name='vol', val=25, min_val=0, max_val=100, widget_name='slider')
        self.add_steering_param(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,
                                widget_name='slider')
        self.add_steering_param(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,
                                widget_name='slider')

        self.clear_flag = False

    def step(self, mcs):
        print('lam_vol=', self.get_steering_param('lam_vol'))
