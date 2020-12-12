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

