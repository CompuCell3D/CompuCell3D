from cc3d.core.PySteppables import *


class SBMLSolverSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # adding options that setup SBML solver integrator
        # these are optional but useful when encountering integration instabilities

        options = {'relative': 1e-10, 'absolute': 1e-12}
        self.set_sbml_global_options(options)

        model_file = 'Simulation/test_1.xml'

        initial_conditions = {}
        initial_conditions['S1'] = 0.00020
        initial_conditions['S2'] = 0.000002

        self.addSBMLToCellIds(_modelFile=model_file, _modelName='dp', _ids=range(1, 11), _stepSize=0.5,
                              _initialConditions=initial_conditions)

        self.addFreeFloatingSBML(_modelFile=model_file, _modelName='Medium_dp', _stepSize=0.5,
                                 _initialConditions=initial_conditions)
        self.addFreeFloatingSBML(_modelFile=model_file, _modelName='Medium_dp1', _stepSize=0.5,
                                 _initialConditions=initial_conditions)
        self.addFreeFloatingSBML(_modelFile=model_file, _modelName='Medium_dp2')
        self.addFreeFloatingSBML(_modelFile=model_file, _modelName='Medium_dp3')
        self.addFreeFloatingSBML(_modelFile=model_file, _modelName='Medium_dp4')

        # self.add_sbml_to_cell_ids(model_file=model_file, model_name='dp', cell_ids=list(range(1, 11)), step_size=0.5,
        #                           initial_conditions=initial_conditions)
        #
        # self.add_free_floating_sbml(model_file=model_file, model_name='Medium_dp', step_size=0.5,
        #                             initial_conditions=initial_conditions)
        # self.add_free_floating_sbml(model_file=model_file, model_name='Medium_dp1', step_size=0.5,
        #                             initial_conditions=initial_conditions)
        #
        # self.add_free_floating_sbml(model_file=model_file, model_name='Medium_dp2')
        # self.add_free_floating_sbml(model_file=model_file, model_name='Medium_dp3')
        # self.add_free_floating_sbml(model_file=model_file, model_name='Medium_dp4')

        cell_20 = self.attemptFetchingCellById(20)

        # self.add_sbml_to_cell(model_file=model_file, model_name='dp', cell=cell_20)
        self.addSBMLToCell(_modelFile=model_file,_modelName='dp',_cell=cell_20)


    def step(self, mcs):
        self.timestep_sbml()

        cell_20 = self.fetch_cell_by_id(20)
        print('cell_20, dp=', cell_20.sbml.dp.values())

        print('Free Floating Medium_dp2', self.sbml.Medium_dp2.values())
        if mcs == 3:
            Medium_dp2 = self.sbml.Medium_dp2
            Medium_dp2['S1'] = 10
            Medium_dp2['S2'] = 0.5

        if mcs == 5:
            self.delete_sbml_from_cell_ids(model_name='dp', cell_ids=list(range(1, 11)))

        if mcs == 7:
            cell_25 = self.fetch_cell_by_id(25)
            self.copy_sbml_simulators(from_cell=cell_20, to_cell=cell_25)
