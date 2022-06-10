"""
Chemotaxis Plugin Demo

The boundary conditions of two soluble signals vary in time. Meanwhile, the sense of chemotactic migration
(*e.g.*, whether attractive or repulsive) switches in cells at regular intervals.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin, ChemotaxisPlugin
from cc3d.core.PyCoreSpecs import DiffusionSolverFE, ReactionDiffusionSolverFE
from cc3d.core.PyCoreSpecs import BlobInitializer
from cc3d.core.PySteppables import *

# Declare cell types
cell_types = ["T1", "T2", "T3", "T4"]

# Declare chemotaxis parameter
lambda_chemotaxis = 5E1

# Specify metadata with multithreading
CompuCellSetup.register_specs(Metadata(num_processors=4))

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=100,
                                        dim_y=100,
                                        steps=10000,
                                        neighbor_order=2))

# Specify cell types
CompuCellSetup.register_specs(CellTypePlugin(*cell_types))

# Apply a volume constraint
volume_specs = VolumePlugin()
for ct in cell_types:
    volume_specs.param_new(ct, target_volume=25, lambda_volume=4)
CompuCellSetup.register_specs(volume_specs)

# Apply basic adhesion modeling
contact_specs = ContactPlugin(neighbor_order=2)
for x1 in range(len(cell_types)):
    contact_specs.param_new(type_1="Medium", type_2=cell_types[x1], energy=16)
    for x2 in range(x1, len(cell_types)):
        contact_specs.param_new(type_1=cell_types[x1], type_2=cell_types[x2], energy=16)
CompuCellSetup.register_specs(contact_specs)

# Apply an initial configuration
blob_init_specs = BlobInitializer()
blob_init_specs.region_new(width=5, radius=20, center=(50, 50, 0), cell_types=cell_types)
CompuCellSetup.register_specs(blob_init_specs)

# Apply a PDE field named "F1" solved by DiffusionSolverFE
f1_solver_specs = DiffusionSolverFE()  # Instantiate solver
f1 = f1_solver_specs.field_new("F1")  # Declare a field for this solver
f1.diff_data.diff_global = 0.0  # Set global diffusion coefficient (redundant, default value is 0.0)
f1.diff_data.decay_global = 1E-4  # Set global decay coefficient
f1.diff_data.init_expression = "x / 100"  # Initialize with steady-state solution
# Set type-specific diffusion and decay coefficients
f1.diff_data.diff_types["Medium"] = 0.1
for ct in cell_types:
    f1.diff_data.decay_types[ct] = 0.01
# Set boundary conditions: Neumann on top and bottom, 0 on left, 1 on right
f1.bcs.y_min_type, f1.bcs.y_max_type = "Flux", "Flux"
f1.bcs.x_max_val = 1.0
f1_solver_specs.fluc_comp = True  # Enable fluctuation compensator
CompuCellSetup.register_specs(f1_solver_specs)

# Apply a PDE field named "F2" solved by ReactionDiffusionSolverFE
f2_solver_specs = ReactionDiffusionSolverFE()  # Instantiate solver
f2 = f2_solver_specs.field_new("F2")  # Declare a field for this solver
# Set global decay coefficient
f2.diff_data.decay_global = 1E-4  # Set global decay coefficient
# Set type-specific diffusion coefficient
f2.diff_data.diff_types["Medium"] = 0.2
# Set boundary conditions: Neumann on left and right, 0 on bottom, 1 on top
f2.bcs.x_min_type, f2.bcs.x_max_type = "Flux", "Flux"
f2.bcs.y_max_val = 1.0
f2_solver_specs.fluc_comp = True  # Enable fluctuation compensator
CompuCellSetup.register_specs(f2_solver_specs)

# Apply chemotaxis
chemotaxis_specs = ChemotaxisPlugin()
cs = chemotaxis_specs.param_new("F1", f1_solver_specs.registered_name)
cs.params_new(cell_types[0], lambda_chemotaxis)
cs.params_new(cell_types[1], -lambda_chemotaxis)
cs = chemotaxis_specs.param_new("F2", f2_solver_specs.registered_name)
cs.params_new(cell_types[2], lambda_chemotaxis)
cs.params_new(cell_types[3], -lambda_chemotaxis)
CompuCellSetup.register_specs(chemotaxis_specs)

# Define a steppable for varying chemotaxis and boundary conditions in simulation time


class ChemotaxisDemoSteppable(SteppableBasePy):

    def step(self, mcs):
        """
        Implements periodic changes in chemotaxis and boundary conditions
        """
        # Implement periodic change of chemotaxis sense using steering
        if mcs % 1000 == 0:
            specs_chemo: ChemotaxisPlugin = self.specs.chemotaxis
            for f in specs_chemo.fields:
                for t in specs_chemo[f].cell_types:
                    specs_chemo[f][t].lambda_chemo *= -1.0

            specs_chemo.steer()

        # Implement a sine curve on the non-zero boundary conditions using steering
        import math
        val = (1.0 + math.sin(mcs / 250)) / 2.0
        specs_f1_solver: DiffusionSolverFE = self.specs.diffusion_solver_fe
        specs_f1_solver.fields["F1"].bcs.x_max_val = val
        specs_f1_solver.steer()
        specs_f2_solver: ReactionDiffusionSolverFE = self.specs.reaction_diffusion_solver_fe
        specs_f2_solver.fields["F2"].bcs.y_max_val = val
        specs_f2_solver.steer()


CompuCellSetup.register_steppable(steppable=ChemotaxisDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
