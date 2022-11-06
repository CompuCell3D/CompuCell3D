"""
AdhesionFlex Plugin Demo

Adjust the sliders to change the binding energy of two molecules, each of which is only on the surface of a particular
cell type. A third cell type has neither molecule.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin
from cc3d.core.PyCoreSpecs import UniformInitializer
from cc3d.core.PyCoreSpecs import AdhesionFlexPlugin
from cc3d.core.PySteppables import *

# Declare cell type names
cell_types = ["T1", "T2", "T3"]

# Declare simulation size
dim_x = 100
dim_y = 100

# Specify metadata with multithreading
CompuCellSetup.register_specs(Metadata(num_processors=4))

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=dim_x,
                                        dim_y=dim_y,
                                        steps=1000000,
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
contact_specs = ContactPlugin(neighbor_order=2)
for idx1 in range(len(cell_types)):
    contact_specs.param_new(type_1="Medium", type_2=cell_types[idx1], energy=16)
    for idx2 in range(idx1, len(cell_types)):
        contact_specs.param_new(type_1=cell_types[idx1], type_2=cell_types[idx2], energy=16)
CompuCellSetup.register_specs(contact_specs)

# Apply adhesion modeling by molecular species
adhesion_specs = AdhesionFlexPlugin(neighbor_order=2)
molecules = ["M1", "M2"]
adhesion_specs.density_new(molecule="M1", cell_type="T1", density=1.0)
adhesion_specs.density_new(molecule="M2", cell_type="T2", density=1.0)
formula = adhesion_specs.formula_new("Binary", "min(Molecule1,Molecule2)")
formula.param_set(molecules[0], molecules[0], 0.0)
formula.param_set(molecules[0], molecules[1], 0.0)
formula.param_set(molecules[1], molecules[1], 0.0)
CompuCellSetup.register_specs(adhesion_specs)

# Apply an initial configuration
unif_init_specs = UniformInitializer()
unif_init_specs.region_new(width=5,
                           pt_min=(0, 0, 0), pt_max=(dim_x, dim_y, 1),
                           cell_types=["T1", "T1", "T2", "T2", "T3"])
CompuCellSetup.register_specs(unif_init_specs)

# Define a steppable for providing instructions on creating and handling a steering panel when running in Player


class AdhesionDemoSteppable(SteppableBasePy):

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in AdhesionFlex plugin
        for changing parameters on-the-fly by the user during simulation execution.
        """
        specs_adhesion_flex: AdhesionFlexPlugin = self.specs.adhesion_flex

        molecules = specs_adhesion_flex.molecules
        for idx1 in range(len(molecules)):
            for idx2 in range(idx1, len(molecules)):
                m1, m2 = molecules[idx1], molecules[idx2]
                self.add_steering_param(name=f"{m1}-{m2}",
                                        val=specs_adhesion_flex.formula["Binary"][m1][m2],
                                        min_val=-10, max_val=10, decimal_precision=0, widget_name="slider")

    def process_steering_panel_data(self):
        """
        Updates AdhesionFlex plugin data based on changes in the steering panel made by the user
        """
        specs_adhesion_flex: AdhesionFlexPlugin = self.specs.adhesion_flex

        molecules = specs_adhesion_flex.molecules
        for idx1 in range(len(molecules)):
            for idx2 in range(idx1, len(molecules)):
                m1, m2 = molecules[idx1], molecules[idx2]
                specs_adhesion_flex.formula["Binary"][m1][m2] = self.get_steering_param(f"{m1}-{m2}")

        specs_adhesion_flex.steer()


CompuCellSetup.register_steppable(steppable=AdhesionDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
