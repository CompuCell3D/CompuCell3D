from cc3d.core.PySteppables import *
from pathlib import Path


class ConcentrationFieldDumperSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        file_name = "diffusion_output/FGF_" + str(mcs) + ".dat"
        fgf_field = self.field.FGF

        output_dir = self.output_dir

        if output_dir is not None:
            output_path = Path(output_dir).joinpath(file_name)
            # create folder to store data
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as file_handle:
                for i, j, k in self.every_pixel():
                    file_handle.write("%d\t%d\t%d\t%f\n" % (i, j, k, fgf_field[i, j, k]))