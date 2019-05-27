from random import uniform
from cc3d.core.PySteppables import *


class DeltaNotchClass(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):

        # adding options that setup SBML solver integrator
        # these are optional but useful when encounteting integration instabilities
        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)

        model_file = 'Simulation/DN_Collier.sbml'
        self.add_sbml_to_cell_types(model_file=model_file, model_name='DN', cell_types=[self.TYPEA], step_size=0.2)

        for cell in self.cell_list:
            dn_model = cell.sbml.DN

            dn_model['D'] = uniform(0.9, 1.0)
            dn_model['N'] = uniform(0.9, 1.0)

            cell.dict['D'] = dn_model['D']
            cell.dict['N'] = dn_model['N']

    def step(self, mcs):

        for cell in self.cell_list:
            delta_tot = 0.0
            nn = 0
            for neighbor, commonSurfaceArea in self.get_cell_neighbor_data_list(cell):
                if neighbor:
                    nn += 1

                    delta_tot += neighbor.sbml.DN['D']
            if nn > 0:
                D_avg = delta_tot / nn

            cell.sbml.DN['Davg'] = D_avg
            cell.dict['D'] = D_avg
            cell.dict['N'] = cell.sbml.DN['N']

        self.timestep_sbml()


class DNVisualizationSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        self.create_scalar_field_cell_level_py("Delta")
        self.create_scalar_field_cell_level_py("Notch")

    def step(self, mcs):
        delta = self.field.Delta
        notch = self.field.Notch
        delta.clear()
        notch.clear()

        for cell in self.cell_list:
            delta[cell] = cell.dict['D']
            notch[cell] = cell.dict['N']
