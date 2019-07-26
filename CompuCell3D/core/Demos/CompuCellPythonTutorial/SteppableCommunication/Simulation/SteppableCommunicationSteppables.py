from cc3d.core.PySteppables import *
import shared_variables


class CommunicationSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        print('shared_variables.shared_parameter=', shared_variables.shared_parameter)


class ExtraSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        shared_variables.shared_parameter = 25

    def step(self, mcs):
        shared_variables.shared_parameter += 1
        print("ExtraSteppable: This function is called every 1 MCS")
