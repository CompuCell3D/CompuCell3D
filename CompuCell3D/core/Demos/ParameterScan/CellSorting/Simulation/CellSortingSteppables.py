from cc3d.core.PySteppables import *

MYVAR = {{MYVAR}}
MYVAR1 = {{MYVAR1}}


class CellSortingSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        # type here the code that will run every _frequency MCS
        global MYVAR

        print('MYVAR=', MYVAR)
        for cell in self.cell_list:
            if cell.type == self.DARK:
                # Make sure ExternalPotential plugin is loaded
                cell.lambdaVecX = -0.5  # force component pointing along X axis - towards positive X's
