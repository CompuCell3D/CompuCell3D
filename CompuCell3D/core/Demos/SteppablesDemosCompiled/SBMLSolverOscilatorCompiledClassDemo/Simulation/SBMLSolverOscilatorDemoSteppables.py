from cc3d.core.PySteppables import *


class SBMLSolverOscilatorDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)

        initial_conditions = {'S1': 0.0, 'S2': 1.0}
        self.add_sbml_to_cell_types(
            model_file='Simulation/oscli.sbml',
            model_name='OSCIL',
            cell_types=[self.NONCONDENSING],
            step_size=0.02,
            initial_conditions=initial_conditions,
        )




