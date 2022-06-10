"""
Curvature Plugin Demo

Adjust the slider to observe differences in enforcing straightness in a line of cells subjected to a force.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import VolumePlugin, CellTypePlugin, ContactPlugin, \
    FocalPointPlasticityPlugin, CurvaturePlugin, ExternalPotentialPlugin, PIFInitializer, \
    ConnectivityGlobalPlugin
from cc3d.core.PySteppables import *

# Declare cell type names
cell_types = ["Top", "Center", "Bottom"]

# Specify Potts with basic simulation specs
spec_potts = PottsCore()
spec_potts.dim_x, spec_potts.dim_y = 100, 100
spec_potts.steps = 100000
spec_potts.neighbor_order = 2
CompuCellSetup.register_specs(spec_potts)

# Specify cell types
specs_cell_type = CellTypePlugin(*cell_types)
specs_cell_type.frozen_set("Top", True)
CompuCellSetup.register_specs(specs_cell_type)

# Apply a volume constraint
spec_volume = VolumePlugin()
for x in cell_types:
    spec_volume.param_new(x, target_volume=25, lambda_volume=2.0)
CompuCellSetup.register_specs(spec_volume)

# Apply basic adhesion modeling
specs_contact = ContactPlugin(neighbor_order=3)
specs_contact.param_new(type_1="Top", type_2="Center", energy=20)
specs_contact.param_new(type_1="Top", type_2="Bottom", energy=100)
specs_contact.param_new(type_1="Center", type_2="Center", energy=20)
specs_contact.param_new(type_1="Center", type_2="Bottom", energy=20)
for x in cell_types:
    specs_contact.param_new(type_1="Medium", type_2=x, energy=10)
CompuCellSetup.register_specs(specs_contact)

# FocalPointPlasticity
fpp_dict = {"lambda_fpp": 500,
            "activation_energy": -50,
            "target_distance": 5.0,
            "max_distance": 20.0,
            "internal": True}
specs_fpp = FocalPointPlasticityPlugin()
specs_fpp.params_new("Top", "Center", max_junctions=1, **fpp_dict)
specs_fpp.params_new("Center", "Center", max_junctions=2, **fpp_dict)
specs_fpp.params_new("Bottom", "Center", max_junctions=1, **fpp_dict)
CompuCellSetup.register_specs(specs_fpp)

# Curvature
specs_curvature = CurvaturePlugin()
for x in cell_types:
    specs_curvature.params_internal_new(x, "Center", 2000, -50)
specs_curvature.params_type_new("Top", 1, 1)
specs_curvature.params_type_new("Center", 2, 1)
specs_curvature.params_type_new("Bottom", 1, 1)
CompuCellSetup.register_specs(specs_curvature)

# ExternalPotential
CompuCellSetup.register_specs(ExternalPotentialPlugin())

# Apply a connectivity constraint
CompuCellSetup.register_specs(ConnectivityGlobalPlugin(*cell_types))

# Initialize a configuration from file
CompuCellSetup.register_specs(PIFInitializer(pif_name="Simulation/curvature.piff"))

# Define a steppable for providing instructions on creating and handling a steering panel when running in Player


class CurvatureDemoSteppable(SteppableBasePy):

    def start(self):
        """
        Applies a potential to all center cells
        """
        for cell in self.cell_list_by_type(self.cell_type.Center):
            cell.lambdaVecX = - 2.0

    def step(self, mcs):
        """
        Constrains the bottom cell to a vertical axis along the COM of the top cell
        """
        target_x = None
        for cell in self.cell_list_by_type(self.cell_type.Top):
            target_x = int(cell.xCOM)
            break
        diff_force = 10
        for cell in self.cell_list_by_type(self.cell_type.Bottom):
            cell.lambdaVecX = diff_force * (cell.xCOM - target_x)

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in Curvature plugin
        for changing parameters on-the-fly by the user during simulation execution.
        """
        self.add_steering_param(name="lambda_curvature",
                                val=2000,
                                min_val=1000,
                                max_val=10000,
                                decimal_precision=0,
                                widget_name="slider")

    def process_steering_panel_data(self):
        """
        Updates Curvature plugin data based on changes in the steering panel made by the user
        """
        lambda_curve = self.get_steering_param("lambda_curvature")
        # CurvaturePlugin
        specs_curvature: CurvaturePlugin = self.specs.curvature
        for t1, t2 in specs_curvature.internal_params:
            specs_curvature.internal_param[t1][t2].lambda_curve = lambda_curve
        specs_curvature.steer()



CompuCellSetup.register_steppable(steppable=CurvatureDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
