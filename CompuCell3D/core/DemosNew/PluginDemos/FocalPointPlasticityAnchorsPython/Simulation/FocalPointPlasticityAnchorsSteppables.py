from cc3d.core.PySteppables import *
from collections import defaultdict
from collections import namedtuple

# For convenience we define a named tuple that will store anchor parameters
AnchorData = namedtuple('AnchorData', 'lambda_ target_length max_length x y z')


class FocalPointPlasticityAnchorSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.anchor_dict = defaultdict(dict)

    def start(self):

        for cell in self.cell_list:
            if cell.id == 5:
                # it is a good idea to store anchor id because any modification to anchor parameters/ deletion
                # of an anchor require the anchor id
                anchor_data = AnchorData(lambda_=20.0, target_length=30.0, max_length=1000.0, x=1, y=1, z=0)
                anchor_id = self.focalPointPlasticityPlugin.createAnchor(cell,
                                                                         anchor_data.lambda_,
                                                                         anchor_data.target_length,
                                                                         anchor_data.max_length,
                                                                         anchor_data.x,
                                                                         anchor_data.y,
                                                                         anchor_data.z)

                self.anchor_dict[cell.id][anchor_id] = anchor_data

    def step(self, mcs):

        for cell in self.cell_list:

            if mcs < 100:
                for fppd in self.get_focal_point_plasticity_data_list(cell):
                    print("CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)

                for anchor_fppd in self.get_anchor_focal_point_plasticity_data_list(cell):
                    print("ANCHORED CELL ID=", cell.id, " CELL TYPE=", cell.type, " volume=", cell.volume)
                    print("lambda=", anchor_fppd.lambdaDistance,
                          " targetDistance=", anchor_fppd.targetDistance,
                          "anchor_x=", anchor_fppd.anchorPoint[0],
                          "anchor_y=", anchor_fppd.anchorPoint[1],
                          "anchor_z=", anchor_fppd.anchorPoint[2])



            elif mcs > 200 and mcs < 300:
                # setting plasticity constraints to 0
                for fppd in self.get_focal_point_plasticity_data_list(cell):
                    print("fppd.neighborId", fppd.neighborAddress.id, " lambda=", fppd.lambdaDistance,
                          " targetDistance=", fppd.targetDistance)

                    # IMPORTANT: although you can access and manipulate focal point plasticity data directly
                    # it is better to do it via setFocalPointPlasticityParameters
                    # IMPORTANT: this way you ensure that data you change is changed in both cell1 and cell2 .
                    # Otherwise if you do direct manipulation , make sure you change parameters in cell1 and
                    # its focal point plasticity neighbor
                    self.focalPointPlasticityPlugin.setFocalPointPlasticityParameters(
                        cell, fppd.neighborAddress, 0.0, 0.0, 0.0)

            elif mcs == 400:
                anchor_list = self.get_anchor_focal_point_plasticity_data_list(cell)
                if len(anchor_list):
                    for anchor_fppd in anchor_list:
                        self.focalPointPlasticityPlugin.setAnchorParameters(
                            cell, anchor_fppd.anchorId, anchor_fppd.lambdaDistance,
                            anchor_fppd.targetDistance / 2.0, anchor_fppd.maxDistance, anchor_fppd.anchorPoint[0],
                            anchor_fppd.anchorPoint[1], anchor_fppd.anchorPoint[2]
                        )

                elif mcs == 600:
                    anchor_list = self.get_anchor_focal_point_plasticity_data_list(cell)
                    if len(anchor_list):
                        for anchor_fppd in anchor_list:
                            self.focalPointPlasticityPlugin.deleteAnchor(cell, anchor_fppd.anchorId)
