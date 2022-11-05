"""
Surface Plugin demo

Adjust the sliders to see how surface tension and other mechanisms affect the ability of one cell type to find another
cell type and neutralize a substance they release

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin
from cc3d.core.PyCoreSpecs import SurfacePlugin, SecretionPlugin, ChemotaxisPlugin
from cc3d.core.PyCoreSpecs import UniformInitializer
from cc3d.core.PyCoreSpecs import ReactionDiffusionSolverFE
from cc3d.core.PySteppables import *

# Declare cell type names
cell_types = ["T1", "T2"]

# Declare simulation size
dim_x = 100
dim_y = 100

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=dim_x,
                                        dim_y=dim_y,
                                        steps=100000,
                                        neighbor_order=2,
                                        boundary_x="Periodic",
                                        boundary_y="Periodic"))

# Specify cell types
CompuCellSetup.register_specs(CellTypePlugin(*cell_types))

# Apply a volume constraint
volume_specs = VolumePlugin()
for ct in cell_types:
    volume_specs.param_new(ct, target_volume=25, lambda_volume=2)
CompuCellSetup.register_specs(volume_specs)

# Apply basic adhesion modeling
contact_specs = ContactPlugin(2)
for x1 in range(len(cell_types)):
    contact_specs.param_new(type_1="Medium", type_2=cell_types[x1], energy=10)
    for x2 in range(x1, len(cell_types)):
        contact_specs.param_new(type_1=cell_types[x1], type_2=cell_types[x2], energy=30)
CompuCellSetup.register_specs(contact_specs)

# Apply two interacting PDE field named "F1" and "F2" solved by ReactionDiffusionSolverFE
diff_solver_specs = ReactionDiffusionSolverFE()
f1 = diff_solver_specs.field_new("F1")
f2 = diff_solver_specs.field_new("F2")
f1.bcs.x_min_type = "Periodic"
f1.bcs.y_min_type = "Periodic"
f2.bcs.x_min_type = "Periodic"
f2.bcs.y_min_type = "Periodic"
f1.diff_data.diff_global = 0.1
f2.diff_data.diff_global = 0.1
f1.diff_data.decay_global = 1E-2
f2.diff_data.decay_global = 1E-3
f1.diff_data.additional_term = "-0.1 * F1 * F2"
f2.diff_data.additional_term = "-0.01 * F1 * F2"
CompuCellSetup.register_specs(diff_solver_specs)

# Apply chemotaxis
chemo_specs = ChemotaxisPlugin()
cs1 = chemo_specs.param_new("F1", diff_solver_specs.registered_name)
cs2 = chemo_specs.param_new("F2", diff_solver_specs.registered_name)
cs1.params_new("T2", lambda_chemo=1E3, towards="Medium")
cs1.params_new("T1", lambda_chemo=-1E3, towards="Medium")
CompuCellSetup.register_specs(chemo_specs)

# Apply secretion
secr_specs = SecretionPlugin()
f1_secr = secr_specs.field_new(f1.field_name)
f2_secr = secr_specs.field_new(f2.field_name)
f1_secr.params_new("T1", 0.01)
f2_secr.params_new("T2", 0.01, contact_type="T1")
CompuCellSetup.register_specs(secr_specs)

# Apply a surface constraint
surface_specs = SurfacePlugin()
for ct in cell_types:
    surface_specs.param_new(ct, target_surface=20.0, lambda_surface=0.0)
CompuCellSetup.register_specs(surface_specs)

# Apply an initial configuration
unif_init_specs = UniformInitializer()
unif_init_specs.region_new(pt_min=(5, 5, 0), pt_max=(dim_x-5, dim_y-5, 1),
                           gap=5, width=5, cell_types=cell_types)
CompuCellSetup.register_specs(unif_init_specs)

# Define a steppable for providing instructions on creating and handling a steering panel when running in Player


class SurfaceDemoSteppable(SteppableBasePy):

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in SurfacePlugin and ChemotaxisPlugin
        for changing parameters on-the-fly by the user during simulation execution.
        """
        specs_cell_type = self.specs.cell_type
        specs_surface = self.specs.surface
        specs_chemotaxis = self.specs.chemotaxis

        for ct in specs_cell_type.cell_types:
            if ct == "Medium":
                continue

            self.add_steering_param(name=f"lambda_surf_{ct}", val=specs_surface[ct].lambda_surface,
                                    min_val=0.0, max_val=5.0, decimal_precision=1, widget_name="slider")
            self.add_steering_param(name=f"target_surf_{ct}", val=specs_surface[ct].target_surface,
                                    min_val=10.0, max_val=30.0, decimal_precision=1, widget_name="slider")

            for f in ["F1", "F2"]:

                s = self._get_secretion_specs(field_name=f, cell_type=ct)
                if s is not None:
                    self.add_steering_param(name=f"secr_{f}_{ct}", val=s.value,
                                            min_val=0.0, max_val=0.01, decimal_precision=3, widget_name="slider")

        for f in ["F1", "F2"]:
            for ct in specs_chemotaxis[f].cell_types:
                self.add_steering_param(name=f"lambda_chemo_{f}_{ct}", val=specs_chemotaxis[f][ct].lambda_chemo,
                                        min_val=-1E3, max_val=1E3, decimal_precision=0, widget_name="slider")

    def process_steering_panel_data(self):
        """
        Updates SurfacePlugin and ChemotaxisPlugin data based on changes in the steering panel made by the user
        """
        for ct in self.specs.cell_type.cell_types:
            if ct == "Medium":
                continue

            self.specs.surface[ct].lambda_surface = self.get_steering_param(f"lambda_surf_{ct}")
            self.specs.surface[ct].target_surface = self.get_steering_param(f"target_surf_{ct}")

            for f in ["F1", "F2"]:
                s = self._get_secretion_specs(field_name=f, cell_type=ct)
                if s is not None:
                    s.value = self.get_steering_param(f"secr_{f}_{ct}")

        for f in ["F1", "F2"]:
            for ct in self.specs.chemotaxis[f].cell_types:
                self.specs.chemotaxis[f][ct].lambda_chemo = self.get_steering_param(f"lambda_chemo_{f}_{ct}")

        self.specs.surface.steer()
        self.specs.secretion.steer()
        self.specs.reaction_diffusion_solver_fe.steer()
        self.specs.chemotaxis.steer()

    def _get_secretion_specs(self, field_name: str, cell_type: str):
        specs_secretion = self.specs.secretion
        field_specs = specs_secretion.fields[field_name]
        for s in field_specs.params:
            if s.cell_type == cell_type:
                return s
        return None


CompuCellSetup.register_steppable(steppable=SurfaceDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
