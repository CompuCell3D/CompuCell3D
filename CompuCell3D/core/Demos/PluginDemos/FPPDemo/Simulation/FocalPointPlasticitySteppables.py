from cc3d.core.PySteppables import *


class FocalPointPlasticityParams(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        for cell in self.cell_list:
            print("CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
            if mcs < 100:
                for fppd in self.get_focal_point_plasticity_data_list(cell):
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)
            elif 200 < mcs < 300:
                # setting plasticity constraints to 0
                for fppd in self.get_focal_point_plasticity_data_list(cell):
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)
                    # IMPORTANT: although you can access and manipulate focal point plasticity data directly
                    # it is better to do it via setFocalPointPlasticityParameters
                    # IMPORTANT: this way you ensure that data you change is changed in both cell1 and cell2 .
                    # Otherwise if you do direct manipulation , make sure you change parameters
                    # in cell1 and its focal point plasticity neighbor
                    # self.focalPointPlasticityPlugin.setFocalPointPlasticityParameters(
                    # cell1,cell2,lambda,targetDistance,maxDistance)
                    self.focalPointPlasticityPlugin.setFocalPointPlasticityParameters(cell, fppd.neighborAddress, 0.0,
                                                                                      0.0, 0.0)
