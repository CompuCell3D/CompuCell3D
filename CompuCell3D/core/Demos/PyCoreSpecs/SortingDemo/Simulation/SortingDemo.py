"""
Cell Sorting Demo

The original demonstration of the Cellular Potts model, all in Python!

Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN
"""

from cc3d import CompuCellSetup
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, CellTypePlugin, VolumePlugin, ContactPlugin, BlobInitializer

# Specify metadata with multithreading
CompuCellSetup.register_specs(Metadata(num_processors=4))

# Specify Potts with basic simulation specs
CompuCellSetup.register_specs(PottsCore(dim_x=100, dim_y=100, steps=16000, neighbor_order=2))

# Specify cell types
CompuCellSetup.register_specs(CellTypePlugin("Condensing", "NonCondensing"))

# Apply a volume constraint
volume_specs = VolumePlugin()
volume_specs.param_new("Condensing", target_volume=25, lambda_volume=2)
volume_specs.param_new("NonCondensing", target_volume=25, lambda_volume=2)
CompuCellSetup.register_specs(volume_specs)

# Apply basic adhesion modeling
contact_specs = ContactPlugin(neighbor_order=2)
contact_specs.param_new(type_1="Medium", type_2="Condensing", energy=16)
contact_specs.param_new(type_1="Medium", type_2="NonCondensing", energy=16)
contact_specs.param_new(type_1="Condensing", type_2="Condensing", energy=2)
contact_specs.param_new(type_1="Condensing", type_2="NonCondensing", energy=11)
contact_specs.param_new(type_1="NonCondensing", type_2="NonCondensing", energy=16)
CompuCellSetup.register_specs(contact_specs)

# Apply an initial configuration
blob_init_specs = BlobInitializer()
blob_init_specs.region_new(width=5, radius=20, center=(50, 50, 0), cell_types=["Condensing", "NonCondensing"])
CompuCellSetup.register_specs(blob_init_specs)

# Run it!
CompuCellSetup.run()
