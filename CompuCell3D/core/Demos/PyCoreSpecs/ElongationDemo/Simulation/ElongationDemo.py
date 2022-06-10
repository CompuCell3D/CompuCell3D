"""
Adapated from
    Merks, Roeland MH, et al.
    "Cell elongation is key to in silico replication of in vitro vasculogenesis and subsequent remodeling."
    Developmental biology 289.1 (2006): 44-54.

    Merks, Roeland MH, et al.
    "Contact-inhibited chemotaxis in de novo and sprouting blood-vessel growth."
    PLoS Comput Biol 4.9 (2008): e1000163.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
Use the sliders to adjust chemotaxis and elongation during angiogensis
"""

from cc3d import CompuCellSetup
from cc3d.core.PySteppables import *
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin, ChemotaxisPlugin
from cc3d.core.PyCoreSpecs import LengthConstraintPlugin, ConnectivityGlobalPlugin
from cc3d.core.PyCoreSpecs import DiffusionSolverFE
from cc3d.core.PyCoreSpecs import UniformInitializer

# Declare simulation details
dim_x = 200
dim_y = 200
site_len = 2E-6  # Length of a voxel side
step_len = 30.0  # Period of a simulation step

# Declare secretion rate
secr_rate = 1.8E-4 * step_len

# Specify metadata with multithreading
CompuCellSetup.register_specs(Metadata(num_processors=4))

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=dim_x, dim_y=dim_y, steps=100000))

# Specify cell types
cell_type_specs = CellTypePlugin("T1", "Wall")
cell_type_specs.frozen_set("Wall", True)
CompuCellSetup.register_specs(cell_type_specs)

# Apply a volume constraint
volume_specs = VolumePlugin()
volume_specs.param_new("T1", target_volume=50, lambda_volume=2)
CompuCellSetup.register_specs(volume_specs)

# Apply basic adhesion modeling
contact_specs = ContactPlugin(neighbor_order=2)
contact_specs.param_new(type_1="Medium", type_2="T1", energy=10)
contact_specs.param_new(type_1="T1", type_2="T1", energy=20)
contact_specs.param_new(type_1="T1", type_2="Wall", energy=50)
CompuCellSetup.register_specs(contact_specs)

# Apply an initial configuration
unif_init_specs = UniformInitializer()
unif_init_specs.region_new(pt_min=(7 + 5, 7 + 5, 0), pt_max=(dim_x - 7 - 5, dim_y - 7 - 5, 1),
                           gap=10, width=7, cell_types=["T1"])
CompuCellSetup.register_specs(unif_init_specs)

# Apply a PDE field named "F1" solved by DiffusionSolverFE
diff_solver_specs = DiffusionSolverFE()
f1 = diff_solver_specs.field_new("F1")
f1.diff_data.diff_global = 1E-13 / (site_len * site_len) * step_len
f1.diff_data.decay_types["Medium"] = secr_rate
f1.secretion_data_new("T1", secr_rate)
f1.bcs.x_min_type = "Periodic"
f1.bcs.y_min_type = "Periodic"
CompuCellSetup.register_specs(diff_solver_specs)

# Apply chemotaxis
chemo_specs = ChemotaxisPlugin()
cs = chemo_specs.param_new(field_name="F1", solver_name="DiffusionSolverFE")
cs.params_new("T1", lambda_chemo=10, towards="Medium")
CompuCellSetup.register_specs(chemo_specs)

# Apply a length constraint
len_specs = LengthConstraintPlugin()
len_specs.params_new("T1", lambda_length=0, target_length=0)
CompuCellSetup.register_specs(len_specs)

# Apply a connectivity constraint
connect_specs = ConnectivityGlobalPlugin(fast=True)
connect_specs.cell_type_append("T1")
CompuCellSetup.register_specs(connect_specs)

# Define a steppable for providing instructions on creating and handling a steering panel when running in Player


class ElongationDemoSteppable(SteppableBasePy):

    def start(self):
        """
        Builds a wall at the beginning of simulation
        """
        self.build_wall(self.cell_type.Wall)

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in Elongation and Chemotaxis plugins
        for changing parameters on-the-fly by the user during simulation execution.
        """
        self.add_steering_param(name="target_length", val=0.0, min_val=0.0, max_val=40.0, decimal_precision=0,
                                widget_name="slider")
        self.add_steering_param(name="lambda_length", val=0.0, min_val=0.0, max_val=1.0, decimal_precision=2,
                                widget_name="slider")
        self.add_steering_param(name="lambda_chemo", val=0.0, min_val=0.0, max_val=1E3, decimal_precision=0,
                                widget_name="slider")

    def process_steering_panel_data(self):
        """
        Updates Elongation and Chemotaxis plugin data based on changes in the steering panel made by the user
        """
        specs_len: LengthConstraintPlugin = self.specs.length_constraint
        specs_chemo: ChemotaxisPlugin = self.specs.chemotaxis
        specs_len["T1"].target_length = self.get_steering_param("target_length")
        specs_len["T1"].lambda_length = self.get_steering_param("lambda_length")
        specs_chemo["F1"]["T1"].lambda_chemo = self.get_steering_param("lambda_chemo")
        specs_len.steer()
        specs_chemo.steer()


CompuCellSetup.register_steppable(steppable=ElongationDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
