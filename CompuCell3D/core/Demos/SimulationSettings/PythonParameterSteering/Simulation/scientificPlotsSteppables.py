from PySteppables import *
import CompuCell
import CompuCellSetup
from SteeringParam import SteeringParam
import sys


class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)

    def start(self):


        self.add_steering_param(name='vol', val=25, min_val=0, max_val=100, widget_name='slider')
        self.add_steering_param(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider')
        self.add_steering_param(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,widget_name='slider')
        

        self.clearFlag = False

    def step(self, mcs):

        # print 'lam_vol=',self.item_data[1].val
        print 'lam_vol=',self.get_steering_param('lam_vol')
        # return
        # if not self.pW:
            # print "To get scientific plots working you need extra packages installed:"
            # print "Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads"
            # print "Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages"
            # return
            # # self.pW.addDataPoint("MCS",mcs,mcs)

        # # self.pW.addDataPoint("MCS1",mcs,-2*mcs)
        # # this is totall non optimized code. It is for illustrative purposes only. 
        # meanSurface = 0.0
        # meanVolume = 0.0
        # numberOfCells = 0
        # for cell in self.cellList:
            # meanVolume += cell.volume
            # meanSurface += cell.surface
            # numberOfCells += 1
        # meanVolume /= float(numberOfCells)
        # meanSurface /= float(numberOfCells)

        # if mcs%100 == 0:  # update plot every 100 mcs
            # self.pW.addDataPoint("MVol", mcs, meanVolume)
            # self.pW.addDataPoint("MSur", mcs, meanSurface)


