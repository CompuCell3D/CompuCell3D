"""
This example demonstrates how to specify cell shape and contact-inhibited chemotaxis.

Adapated from
    Merks, Roeland MH, et al.
    "Cell elongation is key to in silico replication of in vitro vasculogenesis and subsequent remodeling."
    Developmental biology 289.1 (2006): 44-54.

    Merks, Roeland MH, et al.
    "Contact-inhibited chemotaxis in de novo and sprouting blood-vessel growth."
    PLoS Comput Biol 4.9 (2008): e1000163.

Use the sliders to adjust chemotaxis and elongation during angiogensis
"""

__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"

from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
from cc3d.core.PySteppables import SteppableBasePy
from cc3d.core.PyCoreSpecs import Metadata, PottsCore
from cc3d.core.PyCoreSpecs import CellTypePlugin, VolumePlugin, ContactPlugin, ChemotaxisPlugin
from cc3d.core.PyCoreSpecs import LengthConstraintPlugin, ConnectivityGlobalPlugin
from cc3d.core.PyCoreSpecs import DiffusionSolverFE
from cc3d.core.PyCoreSpecs import UniformInitializer


# Define a steppable that builds a wall during startup


class ElongationDemoSteppable(SteppableBasePy):

    def start(self):
        """
        Builds a wall at the beginning of simulation
        """
        self.build_wall(self.cell_type.Wall)


def main():

    # Declare simulation details
    dim_x = dim_y = 200
    site_len = 2E-6  # Length of a voxel side
    step_len = 30.0  # Period of a simulation step

    # Declare secretion rate
    secr_rate = 1.8E-4 * step_len

    # Specify metadata with multithreading
    specs = [
        Metadata(num_processors=4),
        PottsCore(dim_x=dim_x, dim_y=dim_y)
    ]

    # Specify cell types
    cell_type_specs = CellTypePlugin("T1", "Wall")
    cell_type_specs.frozen_set("Wall", True)
    specs.append(cell_type_specs)

    # Apply a volume constraint
    volume_specs = VolumePlugin()
    volume_specs.param_new("T1", target_volume=50, lambda_volume=2)
    specs.append(volume_specs)

    # Apply basic adhesion modeling
    contact_specs = ContactPlugin(neighbor_order=2)
    contact_specs.param_new(type_1="Medium", type_2="T1", energy=10)
    contact_specs.param_new(type_1="T1", type_2="T1", energy=20)
    contact_specs.param_new(type_1="T1", type_2="Wall", energy=50)
    specs.append(contact_specs)

    # Apply an initial configuration
    unif_init_specs = UniformInitializer()
    unif_init_specs.region_new(pt_min=(7 + 5, 7 + 5, 0), pt_max=(dim_x - 7 - 5, dim_y - 7 - 5, 1),
                               gap=10, width=7, cell_types=["T1"])
    specs.append(unif_init_specs)

    # Apply a PDE field named "F1" solved by DiffusionSolverFE
    diff_solver_specs = DiffusionSolverFE()
    f1 = diff_solver_specs.field_new("F1")
    f1.diff_data.diff_global = 1E-13 / (site_len * site_len) * step_len
    f1.diff_data.decay_types["Medium"] = secr_rate
    f1.secretion_data_new("T1", secr_rate)
    f1.bcs.x_min_type = "Periodic"
    f1.bcs.y_min_type = "Periodic"
    specs.append(diff_solver_specs)

    # Apply chemotaxis
    chemo_specs = ChemotaxisPlugin()
    cs = chemo_specs.param_new(field_name="F1", solver_name="DiffusionSolverFE")
    cs.params_new("T1", lambda_chemo=100, towards="Medium")
    specs.append(chemo_specs)

    # Apply a length constraint
    len_specs = LengthConstraintPlugin()
    len_specs.params_new("T1", lambda_length=1, target_length=20)
    specs.append(len_specs)

    # Apply a connectivity constraint
    connect_specs = ConnectivityGlobalPlugin(fast=True)
    connect_specs.cell_type_append("T1")
    specs.append(connect_specs)

    #####################
    # Simulation Launch #
    #####################
    # Initialize a CC3D simulation service instance and register all simulation specification.
    cc3d_sim = CC3DSimService()
    cc3d_sim.register_specs(specs)
    cc3d_sim.register_steppable(steppable=ElongationDemoSteppable)
    cc3d_sim.run()
    cc3d_sim.init()
    cc3d_sim.start()

    #################
    # Visualization #
    #################
    # Show a single frame for the cell field and "F1" to visualize simulation data as it is generated.
    cc3d_sim.visualize()
    frame_f1 = cc3d_sim.visualize()
    frame_f1.field_name = "F1"
    frame_f1.draw()

    #############
    # Execution #
    #############

    # Wait for the user to trigger execution
    input('Press any key to continue...')

    # Execute 10k steps
    while cc3d_sim.current_step < 10000:
        cc3d_sim.step()

    # Report performance
    print(cc3d_sim.profiler_report)

    # Wait for the user to trigger termination
    input('Press any key to close...')


if __name__ == '__main__':
    main()
