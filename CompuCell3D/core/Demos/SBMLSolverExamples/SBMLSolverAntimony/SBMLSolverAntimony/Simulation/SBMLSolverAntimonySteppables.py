from cc3d.core.PySteppables import *


class SBMLSolverSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Antimony model string
        model_string = """model test_1_Antimony()
        # Model
        S1 =>  S2; k1*S1
        
        # Initial conditions
        S1 = 1
        S2 = 0
        k1 = 1
        end"""
        # Antimony model file containing same model as model_string
        model_file = 'Simulation/test_1_Antimony.txt'

        # adding options that setup SBML solver integrator
        # these are optional but useful when encountering integration instabilities

        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)

        # Apply model_string to first ten cells, and model_file to the rest: should result in a uniform model assignment
        for cell in self.cell_list:
            if cell.id < 10:
                self.add_antimony_to_cell(model_string=model_string, model_name='dp', cell=cell, step_size=0.0025)
            else:
                self.add_antimony_to_cell(model_file=model_file, model_name='dp', cell=cell, step_size=0.0025)

    def step(self, mcs):
        self.timestep_sbml()

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return


# Demo: accessing SBML values for further manipulation/coupling with other components
class IdFieldVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.create_scalar_field_cell_level_py("IdFieldS1")
        self.create_scalar_field_cell_level_py("IdFieldS2")

    def step(self, mcs):
        id_field_s1 = self.field.IdFieldS1
        id_field_s2 = self.field.IdFieldS2
        for cell in self.cell_list:
            sbml_values = cell.sbml.dp.values()
            id_field_s1[cell] = sbml_values[0]
            id_field_s2[cell] = sbml_values[1]
            if cell.id == 1:
                print(sbml_values)
