"""
ContactInternal Plugin Demo

Adjust the sliders to change the adhesion of cellular compartments between cells, as well as the adhesion of
cellular compartments within cells.

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin
from cc3d.core.PyCoreSpecs import UniformInitializer
from cc3d.core.PyCoreSpecs import ContactLocalFlexPlugin, ContactInternalPlugin, PixelTrackerPlugin
from cc3d.core.PyCoreSpecs import FocalPointPlasticityPlugin
from cc3d.core.PySteppables import *

# Declare simulation size
dim_x = 105
dim_y = 105

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=dim_x,
                                        dim_y=dim_y,
                                        steps=100000,
                                        neighbor_order=2,
                                        boundary_x="Periodic",
                                        boundary_y="Periodic"))

# Specify cell types
CompuCellSetup.register_specs(CellTypePlugin("A", "B", "C"))

# Apply a volume constraint
volume_specs = VolumePlugin()
volume_specs.param_new("A", lambda_volume=10, target_volume=10)
volume_specs.param_new("B", lambda_volume=10, target_volume=10)
volume_specs.param_new("C", lambda_volume=10, target_volume=30)
CompuCellSetup.register_specs(volume_specs)

# Apply local adhesion modeling
contact_specs = ContactLocalFlexPlugin(neighbor_order=3)
contact_specs.param_new(type_1="Medium", type_2="A", energy=30)
contact_specs.param_new(type_1="Medium", type_2="B", energy=10)
contact_specs.param_new(type_1="Medium", type_2="C", energy=20)
contact_specs.param_new(type_1="A", type_2="A", energy=10)
contact_specs.param_new(type_1="A", type_2="B", energy=20)
contact_specs.param_new(type_1="A", type_2="C", energy=12)
contact_specs.param_new(type_1="B", type_2="B", energy=15)
contact_specs.param_new(type_1="B", type_2="C", energy=12)
contact_specs.param_new(type_1="C", type_2="C", energy=20)
CompuCellSetup.register_specs(contact_specs)

# Apply adhesion modeling to intracellular compartments
contact_intern_specs = ContactInternalPlugin(neighbor_order=3)
contact_intern_specs.param_new(type_1="A", type_2="B", energy=30)
contact_intern_specs.param_new(type_1="A", type_2="C", energy=5)
contact_intern_specs.param_new(type_1="B", type_2="C", energy=5)
CompuCellSetup.register_specs(contact_intern_specs)

# Use PixelTracker plugin
CompuCellSetup.register_specs(PixelTrackerPlugin())

# Apply an initial configuration
unif_init_specs = UniformInitializer()
unif_init_specs.region_new(gap=0, width=7, pt_min=(0, 0, 0), pt_max=(dim_x, dim_y, 1), cell_types=["C"])
CompuCellSetup.register_specs(unif_init_specs)

# Apply links
fpp_specs = FocalPointPlasticityPlugin()
fpp_specs.params_new("A", "A", lambda_fpp=5, activation_energy=-50, target_distance=5, max_distance=20, max_junctions=2)
fpp_specs.params_new("A", "B", lambda_fpp=5, activation_energy=-50, target_distance=10, max_distance=20, internal=True)
CompuCellSetup.register_specs(fpp_specs)

# Define a steppable for providing instructions on creating and handling a steering panel when running in Player


class ContactInternalDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        super().__init__(self, frequency)

        self._params_extern = dict()
        self._params_intern = dict()

    def start(self):
        """
        Initializes random intracellular configurations
        """
        from random import shuffle

        specs_volume: VolumePlugin = self.specs.volume
        target_volume_A = specs_volume["A"].target_volume
        target_volume_B = specs_volume["B"].target_volume

        for cell in self.cell_list_by_type(self.cell_type.C):
            pixel_list = [px.pixel for px in self.get_cell_pixel_list(cell)]
            shuffle(pixel_list)
            pixel_list_A = [pixel_list.pop(x) for x in range(target_volume_A)]
            pixel_list_B = [pixel_list.pop(x) for x in range(target_volume_B)]
            cell_A = self.new_cell(self.cell_type.A)
            self.inventory.reassignClusterId(cell_A, cell.clusterId)
            cell_B = self.new_cell(self.cell_type.B)
            self.inventory.reassignClusterId(cell_B, cell.clusterId)
            for px in pixel_list_A:
                self.cell_field[px.x, px.y, px.z] = cell_A
            for px in pixel_list_B:
                self.cell_field[px.x, px.y, px.z] = cell_B

    def add_steering_panel(self):
        """
        Adds a steering panel and populates with available data in contact plugins
        for changing parameters on-the-fly by the user during simulation execution.
        """
        specs_cell_type: CellTypePlugin = self.specs.cell_type
        specs_contact_local_flex: ContactLocalFlexPlugin = self.specs.contact_local_flex
        specs_contact_internal: ContactInternalPlugin = self.specs.contact_internal

        def_ins = {"min_val": 0.0, "max_val": 50.0, "decimal_precision": 0, "widget_name": "slider"}

        for ct1 in specs_cell_type.cell_types:
            for ct2 in specs_contact_local_flex.types_specified(ct1):
                if ct1 == "Medium" or ct2 == "Medium":
                    continue
                name = f"ext_{ct1}_{ct2}"
                try:
                    self.add_steering_param(name=name, val=specs_contact_local_flex[ct1][ct2].energy, **def_ins)
                    self._params_extern[name] = (ct1, ct2)
                except KeyError:
                    pass
            for ct2 in specs_contact_internal.types_specified(ct1):
                if ct1 == "Medium" or ct2 == "Medium":
                    continue
                name = f"int_{ct1}_{ct2}"
                try:
                    self.add_steering_param(name=name, val=specs_contact_internal[ct1][ct2].energy, **def_ins)
                    self._params_intern[name] = (ct1, ct2)
                except KeyError:
                    pass

    def process_steering_panel_data(self):
        """
        Updates contact plugin data based on changes in the steering panel made by the user
        """
        specs_contact_local_flex: ContactLocalFlexPlugin = self.specs.contact_local_flex
        specs_contact_internal: ContactInternalPlugin = self.specs.contact_internal

        for name, keys in self._params_extern.items():
            specs_contact_local_flex[keys[0]][keys[1]].energy = self.get_steering_param(name)
        for name, keys in self._params_intern.items():
            specs_contact_internal[keys[0]][keys[1]].energy = self.get_steering_param(name)

        specs_contact_local_flex.steer()
        specs_contact_internal.steer()


CompuCellSetup.register_steppable(steppable=ContactInternalDemoSteppable(frequency=1))

# Run it!
CompuCellSetup.run()
