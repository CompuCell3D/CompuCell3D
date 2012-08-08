from PySteppables import *
import CompuCell
import sys


from PlayerPython import *
from math import *

class ExtraFieldVisualizationSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.cellFieldG=self.simulator.getPotts().getCellFieldG()
        self.dim=self.cellFieldG.getDim()

    def setScalarField(self,_field):
        self.scalarField=_field

    def start(self):pass

    def step(self,mcs):
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    if (not mcs%20):
                        value=x*y
                        fillScalarValue(self.scalarField,x,y,z,value)
                    else:
                        value=sin(x*y)
                        fillScalarValue(self.scalarField,x,y,z,value)

