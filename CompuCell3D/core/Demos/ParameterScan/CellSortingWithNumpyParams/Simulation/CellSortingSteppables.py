from cc3d.core.PySteppables import *

MYVAR = {{MYVAR}}
MYVAR1 = {{MYVAR1}}


class CellSortingSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # accessing current parameter scan iteration from steppable
        iteration = self.param_scan_iteration

        # creating file in the parameter scan main output folder - the top level one where we also store
        # subfolders with outputs from individual parameter scan iteration simulations
        file_handle, output_path = self.open_file_in_parameter_scan_main_output_folder(f'demo_file_{iteration}')
        file_handle.close()

        # opening file in simulation output folder - it is different than parameter scan main output folder - it is
        # two level down
        file_handle_1, output_path_1 = self.open_file_in_simulation_output_folder(f'simulation_output_demo_file')
        file_handle_1.close()

    def step(self, mcs):
        # type here the code that will run every _frequency MCS
        global MYVAR

        print('MYVAR=', MYVAR)
        for cell in self.cell_list:
            if cell.type == self.DARK:
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = -0.5  # force component pointing along X axis - towards positive X's
