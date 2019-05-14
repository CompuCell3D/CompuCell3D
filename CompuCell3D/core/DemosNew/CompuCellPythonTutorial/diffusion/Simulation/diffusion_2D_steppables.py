from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
from pathlib import Path


class ConcentrationFieldDumperSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        file_name = "diffusion_output/FGF_" + str(mcs) + ".dat"
        fgf_field = self.field.FGF

        # persistent_globals object stores many global variables that are shared among simulation objects
        # one such variable is output_directory
        output_dir = CompuCellSetup.persistent_globals.output_directory

        if output_dir is not None:
            output_path = Path(output_dir).joinpath(file_name)
            # create folder to store data
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as fileHandle:
                for i, j, k in self.everyPixel():
                    fileHandle.write("%d\t%d\t%d\t%f\n" % (i, j, k, fgf_field[i, j, k]))