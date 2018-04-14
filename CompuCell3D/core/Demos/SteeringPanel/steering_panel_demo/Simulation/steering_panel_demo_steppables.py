from PySteppables import *
import CompuCell
import CompuCellSetup
from SteeringParam import SteeringParam
import sys


class VolumeSteeringSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def add_steering_panel(self):
        self.add_steering_param(name='target_vol', val=25, min_val=0, max_val=100, widget_name='slider')
        self.add_steering_param(name='lambda_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider')
        self.add_steering_param(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,widget_name='slider')

    def process_steering_panel_data(self):
        print "VolumeSteeringSteppable: WILL ADJUST PARAMETERS BECAUSE AT LEAST ONE STEERING PARAMETER HAS BEEN CHANGED"
        print 'all dirty flag=', self.steering_param_dirty()
        target_vol = self.get_steering_param('target_vol')
        lambda_vol = self.get_steering_param('lambda_vol')

        for cell in self.cellList:

            cell.targetVolume = target_vol
            cell.lambdaVolume = lambda_vol


    def start(self):

        for cell in self.cellList:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

    def step(self, mcs):

        print 'lambda_vol=',self.get_steering_param('lambda_vol')



class SurfaceSteeringSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def add_steering_panel(self):
        self.add_steering_param(name='target_surface', val=20, min_val=0, max_val=100, widget_name='slider')
        self.add_steering_param(name='lambda_surface', val=0.2, min_val=0, max_val=10.0, decimal_precision=2,
                                widget_name='slider')

    def process_steering_panel_data(self):
        print "SurfaceSteeringSteppable: WILL ADJUST PARAMETERS BECAUSE AT LEAST ONE STEERING PARAMETER HAS BEEN CHANGED"
        print 'all dirty flag=', self.steering_param_dirty()
        target_surf = self.get_steering_param('target_surface')
        lambda_surf = self.get_steering_param('lambda_surface')

        for cell in self.cellList:

            cell.targetSurface = target_surf
            cell.lambdaSurface = lambda_surf


    def start(self):
        for cell in self.cellList:
            cell.targetSurface = 20
            cell.lambdaSurface = 0.2



    def step(self, mcs):

        # print 'lam_vol=',self.item_data[1].val
        print 'lambda_surface=', self.get_steering_param('lambda_surface')
