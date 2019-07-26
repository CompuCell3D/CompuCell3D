from cc3d.core.PySteppables import *


class SBMLSolverOscilatorDemoSteppable(SteppableBasePy):

    def __init__(self, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.pW = None

    def start(self):
        self.pW = self.add_new_plot_window(title='S1 concentration', x_axis_title='MonteCarlo Step (MCS)',
                                           y_axis_title='Variables')
        self.pW.addPlot('S1', _style='Dots', _color='red', _size=5)

        # iterating over all cells in simulation        
        for cell in self.cell_list:
            # you can access/manipulate cell properties here
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

        # SBML SOLVER

        # adding options that setup SBML solver integrator - these are optional
        # but useful when encountering integration instabilities
        options = {'relative': 1e-10, 'absolute': 1e-12}
        # options={'relative':1e-10,'absolute':1e-12}
        self.set_sbml_global_options(options)

        model_file = 'Simulation/oscli.sbml'  # this can be e.g. partial path 'Simulation/oscli.sbml'
        step_size = 0.02

        initial_conditions = {}
        initial_conditions['S1'] = 0.0
        initial_conditions['S2'] = 1.0
        self.add_sbml_to_cell_types(model_file=model_file, model_name='OSCIL', cell_types=[self.NONCONDENSING],
                                    step_size=step_size, initial_conditions=initial_conditions)

    def step(self, mcs):
        if not self.pW:
            self.pW = self.addNewPlotWindow(_title='S1 concentration', _xAxisTitle='MonteCarlo Step (MCS)',
                                            _yAxisTitle='Variables')
            self.pW.add_plot('S1', style='Dots', color='red', size=5)

        added = False
        for cell in self.cell_list:

            if cell.type == self.NONCONDENSING:
                print(cell.sbml)
                state = self.get_sbml_state(model_name='OSCIL', cell=cell)
                concentration = state['S1']
                cell.targetVolume = 25 + 10 * concentration

                if not added:
                    self.pW.addDataPoint("S1", mcs, concentration)
                    added = True

        if mcs > 2:
            for cell in self.cell_list:
                if cell.type == self.NONCONDENSING:
                    sbml_model = cell.sbml._rr_OSCIL
                    print('sbml_model=',sbml_model)
                    print('S1=',sbml_model['S1'])
                    break

        if mcs == 3:
            for cell in self.cell_list:
                if cell.type == self.NONCONDENSING:
                    state = cell.sbml.OSCIL
                    state['S1'] = 1.3
                    # cell.sbml.OSCIL['S1'] = 1.3

        self.pW.showAllPlots()
        self.timestep_sbml()
