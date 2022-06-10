"""
SteadyStateSolver Demo

Adjust the sliders to change the boundary conditions of chemoattractants for two cell types.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin, ChemotaxisPlugin
from cc3d.core.PyCoreSpecs import SteadyStateDiffusionSolver
from cc3d.core.PyCoreSpecs import UniformInitializer
from cc3d.core.PySteppables import *

# Declare cell types
cell_types = ["T1", "T2", "Wall"]

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=100,
                                        dim_y=100,
                                        steps=100000,
                                        neighbor_order=2,
                                        boundary_x="Periodic",
                                        boundary_y="Periodic"))

# Specify cell types
cell_type_specs = CellTypePlugin(*cell_types)
cell_type_specs.frozen_set("Wall", True)
CompuCellSetup.register_specs(cell_type_specs)

# Apply a volume constraint
volume_specs = VolumePlugin()
for ct in cell_types:
    volume_specs.param_new(ct, target_volume=25, lambda_volume=2)
CompuCellSetup.register_specs(volume_specs)

# Apply basic adhesion modeling
contact_specs = ContactPlugin(neighbor_order=2)
contact_specs.param_new(type_1="Medium", type_2=cell_types[0], energy=16)
contact_specs.param_new(type_1="Medium", type_2=cell_types[1], energy=16)
contact_specs.param_new(type_1=cell_types[0], type_2=cell_types[0], energy=4)
contact_specs.param_new(type_1=cell_types[0], type_2=cell_types[1], energy=11)
contact_specs.param_new(type_1=cell_types[1], type_2=cell_types[1], energy=16)
CompuCellSetup.register_specs(contact_specs)

# Apply a PDE field named "F1" solved by SteadyStateDiffusionSolver
ss_solver_specs = SteadyStateDiffusionSolver()
f1 = ss_solver_specs.field_new("F1")
f1.diff_data.diff_global = 0.1
f2 = ss_solver_specs.field_new("F2")
f2.diff_data.diff_global = 0.1
CompuCellSetup.register_specs(ss_solver_specs)

# Apply chemotaxis
chemotaxis_specs = ChemotaxisPlugin()
cs = chemotaxis_specs.param_new("F1", "SteadyStateDiffusionSolver")
cs.params_new(cell_types[0], lambda_chemo=1E3)
cs = chemotaxis_specs.param_new("F2", "KernelDiffusionSolver")
cs.params_new(cell_types[1], lambda_chemo=1E3)
CompuCellSetup.register_specs(chemotaxis_specs)

# Apply an initial configuration
blob_init_specs = UniformInitializer()
blob_init_specs.region_new(pt_min=(0, 6, 0), pt_max=(100, 15, 1), width=5, cell_types=[cell_types[0]])
blob_init_specs.region_new(pt_min=(0, 86, 0), pt_max=(100, 95, 1), width=5, cell_types=[cell_types[1]])
blob_init_specs.region_new(pt_min=(0, 0, 0), pt_max=(100, 5, 1), width=5, cell_types=[cell_types[2]])
blob_init_specs.region_new(pt_min=(0, 95, 0), pt_max=(100, 100, 1), width=5, cell_types=[cell_types[2]])
CompuCellSetup.register_specs(blob_init_specs)


class SteadyStateSolverDemoSteppable(SteppableBasePy):

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in SteadyStateDiffusionSolver
        for changing parameters on-the-fly by the user during simulation execution.
        """
        val_dict = {"val": 0.0, "min_val": 0.0, "max_val": 1.0, "decimal_precision": 2, "widget_name": "slider"}
        self.add_steering_param(name="F1Top", **val_dict)
        self.add_steering_param(name="F1Bottom", **val_dict)
        self.add_steering_param(name="F2Top", **val_dict)
        self.add_steering_param(name="F2Bottom", **val_dict)

    def process_steering_panel_data(self):
        """
        Updates SteadyStateDiffusionSolver data based on changes in the steering panel made by the user
        """
        ss_solver: SteadyStateDiffusionSolver = self.specs.steady_state_diffusion_solver
        ss_solver.fields["F1"].bcs.y_max_val = self.get_steering_param("F1Top")
        ss_solver.fields["F1"].bcs.y_min_val = self.get_steering_param("F1Bottom")
        ss_solver.fields["F2"].bcs.y_max_val = self.get_steering_param("F2Top")
        ss_solver.fields["F2"].bcs.y_min_val = self.get_steering_param("F2Bottom")
        ss_solver.steer()


CompuCellSetup.register_steppable(steppable=SteadyStateSolverDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
