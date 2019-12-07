import cc3d
# import cc3d.CompuCellSetup as CompuCelllSetup
from cc3d import CompuCellSetup
from pathlib import Path

if mcs == 800:
    output_dir = self.output_dir
    print()
    if output_dir is not None:
        output_path = Path(output_dir).joinpath('step_' + str(mcs).zfill(3) + '.dat')

        # this creates output folder if necessary (unlikely that this line is neede)
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        #output_path = field.piff
        with open(str(output_path), 'w') as fout:

            attr_field = self.field.MigC
            for x, y, z in self.every_pixel():
                fout.write('{} {} {} {}\n'.format(x, y, z, attr_field[x, y, z]))