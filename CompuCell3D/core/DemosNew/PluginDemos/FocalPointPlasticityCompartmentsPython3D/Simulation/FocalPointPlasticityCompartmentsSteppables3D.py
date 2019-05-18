from cc3d.core.PySteppables import *


class FocalPointPlasticityCompartmentsParams(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            print("CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
            if mcs < 100:
                for fppd in self.get_internal_focal_point_plasticity_data_list(cell):
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)

            elif mcs > 700 and mcs < 800:  # will remove fpp constraint for all the cells
                for fppd in self.get_internal_focal_point_plasticity_data_list(cell):
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)
                    # IMPORTANT: although you can access and manipulate focal point plasticity data directly
                    # it is better to do it via setFocalPointPlasticityParameters
                    # IMPORTANT: this way you ensure that data you change is changed in both cell1 and cell2 .
                    # Otherwise if you do direct manipulation , make sure you change parameters in cell1 and
                    # its focal point plasticity neighbor
                    # self.focalPointPlasticityPlugin.setInternalFocalPointPlasticityParameters(cell1,cell2,lambda,targetDistance,maxDistance) 
                    self.focalPointPlasticityPlugin.setInternalFocalPointPlasticityParameters(cell,
                                                                                              fppd.neighborAddress, 0.0,
                                                                                              0.0, 0.0)
