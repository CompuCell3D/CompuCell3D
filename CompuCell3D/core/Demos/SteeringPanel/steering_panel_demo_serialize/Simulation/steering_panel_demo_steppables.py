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

        print CompuCellSetup.steering_param_dict
        out_fname = 'd:\CC3DProjects\steering_panel.json'
        CompuCellSetup.serialize_steering_panel(fname=out_fname)
        # CompuCellSetup.deserialize_steering_panel(fname=out_fname)
        # for n, v in CompuCellSetup.steering_param_dict.items():
        #     print 'n='
        #     print 'v =',v



class SurfaceSteeringSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    
    
    def add_steering_panel(self):
        self.add_steering_param(name='MY_PARAM_SLIDER', val=20, min_val=0, max_val=100,
                                decimal_precision=2, widget_name='slider')
        self.add_steering_param(name='MY_PARAM_COMBO', val=20, enum=[10,20,30,40,50,60,70,80,90,100],
                                    widget_name='combobox')
    
    def process_steering_panel_data(self):
        print 'processing steering panel updates'
        print 'all dirty flag=', self.steering_param_dirty()
        param_slider = self.get_steering_param('MY_PARAM_SLIDER')
        param_combo = self.get_steering_param('MY_PARAM_COMBO')
        print 'updated MY_PARAM_SLIDER=',param_slider
        print 'updated MY_PARAM_COMBO=', param_combo
    
    
    



    def start(self):
        for cell in self.cellList:
            cell.targetSurface = 20
            cell.lambdaSurface = 0.2



    def step(self, mcs):

        # print 'lam_vol=',self.item_data[1].val
        print 'MY_PARAM_SLIDER=', self.get_steering_param('MY_PARAM_SLIDER')
