from cc3d.tests.test_utils.UnittestSteppables import *


class DiffusionSolverSteppable(UnittestSteppablePdeSolver):
    def __init__(self, frequency=100):
        UnittestSteppablePdeSolver.__init__(self, frequency)
        self.generate_reference_flag = False
        self.test_mcs = 500
        self.model_output_csv = Path(__file__).parent.parent.joinpath('reference_output', 'model.output.csv')
        self.model_path = str(Path(__file__).parent.parent)
