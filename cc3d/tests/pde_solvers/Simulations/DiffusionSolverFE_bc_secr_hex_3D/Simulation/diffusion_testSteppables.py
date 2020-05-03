from cc3d.tests.test_utils.UnittestSteppables import *


class DiffusionTestSteppable(UnittestSteppablePdeSolver):
    def __init__(self, frequency=100):
        UnittestSteppablePdeSolver.__init__(self, frequency)
        self.generate_reference_flag = True
        self.test_mcs = 500
        self.fields_to_store = ['FGF']
        self.model_output_core = Path(__file__).parent.parent.joinpath('reference_output', 'model.output')
        self.model_path = str(Path(__file__).parent.parent)

    def start(self):
        #creating a cell (it is frozen - see XML)
        cell = self.new_cell(self.A)

        self.cell_field[self.dim.x/2:self.dim.x/2 + 3,  self.dim.y/2:self.dim.y/2 + 3, self.dim.z/2:self.dim.z/2 + 3] = cell



        