import os
from typing import Union
import warnings
import types
from cc3d.cpp import CompuCell
from cc3d import CompuCellSetup

try:
    import roadrunner
    from .RoadRunnerPy import RoadRunnerPy

    roadrunner_available = True
except ImportError:
    roadrunner_available = False


# this function will replace all API functions which refer to SBMLSolver in the event that
# no SBMLSolver event is installed

def SBMLSolverError(self, *args, **kwrds):
    import inspect
    line = inspect.stack()[1][2]
    call = inspect.stack()[1][4]
    raise AttributeError('SBMLSolverError line :' + str(line) + ' call:' + str(
        call) + ' Trying to access one of the SBML solver methods'
                ' but SBMLSolver engine (e.g. RoadRunner) has not been installed with your CompuCell3D package')


class SBMLSolverHelper(object):
    @classmethod
    def remove_attribute(cls, name):
        print('cls=', cls)
        return delattr(cls, name)

    def __init__(self):

        # in case user passes simulate options we set the here 
        # this dictionary translates old options valid for earlier rr versions to new ones
        self.option_name_dict = {
            'relative': 'relative_tolerance',
            'absolute': 'absolute_tolerance',
            'steps': 'maximum_num_steps'
        }
        print(dir(self))
        if not roadrunner_available:
            sbml_solver_api = ['add_free_floating_sbml', 'add_sbml_to_cell', 'add_sbml_to_cell_ids',
                               'add_sbml_to_cell_types',
                               'copy_sbml_simulators', 'delete_free_floating_sbml', 'delete_sbml_from_cell',
                               'delete_sbml_from_cell_ids', 'delete_sbml_from_cell_types',
                               'get_sbml_global_options', 'get_sbml_simulator', 'get_sbml_state',
                               'get_sbml_state_as_python_dict', 'get_sbml_value', 'normalize_path',
                               'set_sbml_global_options', 'set_sbml_state', 'set_sbml_value', 'set_step_size_for_cell',
                               'set_step_size_for_cell_ids', 'set_step_size_for_cell_types',
                               'set_step_size_for_free_floating_sbml',
                               'timestep_cell_sbml', 'timestep_free_floating_sbml', 'timestep_sbml']

            for api_name in sbml_solver_api:
                SBMLSolverHelper.remove_attribute(api_name)
                setattr(SBMLSolverHelper, api_name, types.MethodType(SBMLSolverError, SBMLSolverHelper))

    def __default_mutable_type(self, obj: Union[object, list, dict], obj_default: Union[list, dict] = None) -> Union[
        object, list, dict]:
        """
        helper function  - deals with default mutable arguments of function
        :param obj: {}
        :param obj_default:
        :return:
        """
        if obj is None:
            return obj_default
        else:
            return obj

    def add_sbml_to_cell(self, model_file: str = '', model_name: str = '', cell: object = None, step_size: float = 1.0,
                         initial_conditions: Union[None, dict] = None, options: Union[None, dict] = None,
                         current_state_sbml: object = None) -> None:
        """
        Attaches RoadRunner SBML solver to a particular cell. The sbml solver is stored as an element
        of the cell's dictionary - cell.dict['SBMLSolver'][_modelName]. The function has a dual operation mode.
        When user provides current_state_sbml, cell model_name, step_size the add_sbml_to_cell function creates a clone
        of a solver whose state is described by the current_state_sbml . If current_state_sbml is None
        then the new SBML solver
        is being created,  SBML file (model_file) loaded and initial conditions are applied.
        It is important to always set
        ste_size to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path

        :param model_name: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver']
        dictionary

        :param cell: {CellG object} cc3d cell object

        :param step_size:  time step- determines how much in "real" time units timestep() fcn advances SBML solver

        :param initial_conditions: initial conditions dictionary

        :param options: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        :param current_state_sbml:  string representation  of the SBML representing current state of the solver.

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})

        core_model_name = model_name
        if core_model_name == '':
            core_model_name, ext = os.path.splitext(os.path.basename(model_file))

        if not model_file:
            warnings.warn('\n\n\n _modelFile argument not provided to addSBMLToCell. '
                          'This will prevent proper restart of the simulation'
                          'You may ignore this warning if you are not '
                          'serializing simulation for future restarts', RuntimeWarning)

        model_path_normalized = self.normalize_path(model_file)

        dict_attrib = CompuCell.getPyAttrib(cell)

        sbml_dict = {}
        if 'SBMLSolver' in dict_attrib:
            sbml_dict = dict_attrib['SBMLSolver']
        else:
            dict_attrib['SBMLSolver'] = sbml_dict

        if current_state_sbml is None:
            rr = RoadRunnerPy(_path=model_file)
            # setting stepSize
            rr.stepSize = step_size
            # loading SBML and LLVM-ing it
            rr.loadSBML(_externalPath=model_path_normalized)

        else:
            rr = RoadRunnerPy(sbml=current_state_sbml)
            # setting stepSize
            rr.stepSize = step_size

            # setting up paths - IMPORTANT FOR RESTARTING
            rr.path = model_file
            if os.path.exists(model_path_normalized):
                rr.absPath = model_path_normalized

        # storing rr instance in the cell dictionary
        sbml_dict[core_model_name] = rr

        # setting initial conditions - this has to be done after loadingSBML
        for name, value in initial_conditions.items():
            # have to catch exceptions in case initial conditions contain
            # "unsettable" entries such as reaction rate etc...
            try:
                rr.model[name] = value
            except:
                pass
                # we are turning off dynamic python properties because rr is not used in the interactive mode.
                # rr.options.disablePythonDynamicProperties = True

        # setting output results array size 
        rr.selections = []  # by default we do not request any output array at each intergration step

        if options:
            for name, value in options.items():

                try:
                    setattr(rr.getIntegrator(), name, value)
                except AttributeError:
                    setattr(rr.getIntegrator(), self.option_name_dict[name], value)
        else:

            # check for global options

            global_options = self.get_sbml_global_options()
            if global_options:
                for name, value in global_options.items():
                    try:
                        setattr(rr.getIntegrator(), name, value)
                    except (AttributeError, ValueError):
                        setattr(rr.getIntegrator(), self.option_name_dict[name], value)
                        # setattr(rr.simulateOptions,name,value)

    def get_sbml_global_options(self):
        """
        returns global options for the SBML solver - deprecated as newer version of CC3D
        :return {dict}: global SBML solver options
        """
        pg = CompuCellSetup.persistent_globals
        return pg.global_sbml_simulator_options

    def set_sbml_global_options(self, options: dict) -> None:
        """
        Deprecated  - sets global SBML options
        :param options:
        :return: None
        """

        pg = CompuCellSetup.persistent_globals
        pg.global_sbml_simulator_options = options

    def add_sbml_to_cell_types(self, model_file: str = '', model_name: str = '', cell_types: Union[None, list] = None,
                               step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                               options: Union[None, dict] = None) -> None:
        """
        Adds SBML Solver to all cells of given cell type - internally it calls addSBMLToCell(fcn).
        Used during initialization of the simulation. It is important to always set
        _stepSize to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_name: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver']
        dictionary

        :param cell_types: list of cell types
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})
        cell_types = self.__default_mutable_type(cell_types, [])

        if 'steps' in list(options.keys()):
            warnings.warn('-----------WARNING-----------------\n'
                          ' steps option for SBML solver is deprecated.',
                          RuntimeWarning)

        for cell in self.cellListByType(*cell_types):
            self.add_sbml_to_cell(model_file=model_file, model_name=model_name, cell=cell, step_size=step_size,
                                  initial_conditions=initial_conditions, options=options)

    def add_sbml_to_cell_ids(self, model_file: str, model_name: str = '', cell_ids: Union[None, list] = None,
                             step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                             options: Union[None, dict] = None) -> None:
        """
        Adds SBML Solver to all cells of given cell ids - internally it calls add_sbml_to_cell fcn.
        Used during initialization of the simulation. It is important to always set
        step_size to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path

        :param model_name: name of the model - this is a label used to store mode in the
         cell.dict['SBMLSolver'] dictionary

        :param cell_ids: list of cell ids

        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver

        :param initial_conditions: initial conditions dictionary

        :param options: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})
        cell_ids = self.__default_mutable_type(cell_ids, [])

        if 'steps' in list(options.keys()):
            warnings.warn('-----------WARNING-----------------\n\n steps option for SBML solver is deprecated.',
                          RuntimeWarning)

        for cell_id in cell_ids:
            cell = self.inventory.attemptFetchingCellById(cell_id)
            if not cell:
                continue

            self.add_sbml_to_cell(model_file=model_file, model_name=model_name, cell=cell, step_size=step_size,
                                  initial_conditions=initial_conditions, options=options)

    def add_free_floating_sbml(self, model_file: str, model_name: str = '', step_size: float = 1.0,
                               initial_conditions: Union[None, dict] = None, options: Union[None, dict] = None):
        """
        Adds free floating SBML model - not attached to any cell. The model will be identified/referenced by the _modelName
        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_name: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        In the newer versions of RR this might be not necessary. The keys that are supported are the following:

        absolute - determines absolute tolerance default 1e-10
        relative - determines relative tolerance default 1e-5
        stiff - determines if using stiff solver or not default False

        :return: None
        """
        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})

        model_path_normalized = self.normalize_path(model_file)
        try:
            f = open(model_path_normalized, 'r')
            f.close()
        except IOError as e:
            if self.simulator.getBasePath() != '':
                model_path_normalized = os.path.abspath(
                    os.path.join(self.simulator.getBasePath(), model_path_normalized))

        rr = RoadRunnerPy(_path=model_file)
        rr.loadSBML(_externalPath=model_path_normalized)

        # setting stepSize
        rr.stepSize = step_size

        # storing
        pg = CompuCellSetup.persistent_globals
        pg.free_floating_sbml_simulators[model_name] = rr

        # setting initial conditions - this has to be done after loadingSBML
        for name, value in initial_conditions.items():
            rr.model[name] = value

            # setting output results array size
        rr.selections = []  # by default we do not request any output array at each integration step

        # in case user passes simulate options we set the here        
        if options:
            for name, value in options.items():

                try:
                    setattr(rr.getIntegrator(), name, value)
                except (AttributeError, ValueError) as e:
                    setattr(rr.getIntegrator(), self.option_name_dict[name], value)
        else:
            # check for global options
            global_options = self.get_sbml_global_options()
            if global_options:
                for name, value in global_options.items():
                    try:
                        setattr(rr.getIntegrator(), name, value)
                    except (AttributeError, ValueError) as e:
                        setattr(rr.getIntegrator(), self.option_name_dict[name], value)

    def delete_sbml_from_cell_ids(self, model_name: str, cell_ids: Union[None, list] = None) -> None:
        """
        Deletes  SBML model from cells whose ids match those stered int he _ids list
        :param model_name: model name
        :param cell_ids: list of cell ids
        :return:
        """
        """
        
        :param _modelName {str}: 
        :param _ids {list}: 
        :return: None
        """
        cell_ids = self.__default_mutable_type(cell_ids, [])

        for cell_id in cell_ids:
            cell = self.inventory.attemptFetchingCellById(cell_id)
            if not cell:
                continue

            dict_attrib = CompuCell.getPyAttrib(cell)
            try:
                sbml_dict = dict_attrib['SBMLSolver']
                del sbml_dict[model_name]
            except LookupError as e:
                pass

    def delete_sbml_from_cell_types(self, model_name: str, cell_types: Union[None, list] = None) -> None:
        """
        Deletes  SBML model from cells whose type match those stered in the cell_types list
        :param model_name: model name
        :param cell_types: list of cell cell types
        :return:
        """
        """
        
        :param _modelName {str}: 
        :param types: 
        :return: None

        """
        cell_types = self.__default_mutable_type(cell_types, [])

        for cell in self.cellListByType(*cell_types):
            dict_attrib = CompuCell.getPyAttrib(cell)
            try:
                sbml_dict = dict_attrib['SBMLSolver']
                del sbml_dict[model_name]
            except LookupError:
                pass

    def delete_sbml_from_cell(self, model_name: str = '', cell: object = None) -> None:
        """
        Deletes SBML from a particular cell
        :param model_name: model name
        :param cell: CellG cell obj
        :return: None
        """

        dict_attrib = CompuCell.getPyAttrib(cell)
        try:
            sbml_dict = dict_attrib['SBMLSolver']
            del sbml_dict[model_name]
        except LookupError:
            pass

    def delete_free_floating_sbml(self, model_name: str) -> None:
        """
        Deletes free floating SBLM mo
        del
        :param model_name: model name
        :return: None
        """
        pg = CompuCellSetup.persistent_globals

        try:
            del pg.free_floating_sbml_simulators[model_name]
        except LookupError:
            pass

    def timestep_cell_sbml(self):
        """
        advances (integrats forward) models stored as attributes of cells
        :return: None
        """

        # time-stepping SBML attached to cells
        for cell in self.cellList:
            dict_attrib = CompuCell.getPyAttrib(cell)
            if 'SBMLSolver' in dict_attrib:
                sbml_dict = dict_attrib['SBMLSolver']

                for model_name, rrTmp in sbml_dict.items():
                    # integrating SBML
                    rrTmp.timestep()

    def set_step_size_for_cell(self, model_name: str = '', cell: object = None, step_size: float = 1.0):
        """
        Sets integration step size for SBML model attached to _cell

        :param model_name: model name
        :param cell: CellG cell object
        :param step_size: integration step size
        :return: None
        """
        dict_attrib = CompuCell.getPyAttrib(cell)

        try:
            sbmlSolver = dict_attrib['SBMLSolver'][model_name]
        except LookupError:
            return

        sbmlSolver.stepSize = step_size

    def set_step_size_for_cell_ids(self, model_name: str = '', cell_ids: Union[None, list] = None,
                                   step_size: float = 1.0) -> None:

        """
        Sets integration step size for SBML model attached to cells of given ids

        :param model_name:  model name
        :param cell_ids : list of cell ids
        :param step_size : integration step size
        :return: None
        """
        cell_ids = self.__default_mutable_type(cell_ids, [])

        for cell_id in cell_ids:
            cell = self.inventory.attemptFetchingCellById(cell_id)
            if not cell:
                continue

            self.set_step_size_for_cell(model_name=model_name, cell=cell, step_size=step_size)

    def set_step_size_for_cell_types(self, model_name: str = '', cell_types: Union[None, list] = None,
                                     step_size=1.0) -> None:
        """
        Sets integration step size for SBML model attached to cells of given cell types

        :param model_name: model name
        :param cell_types: list of cell types
        :param step_size: integration step size
        :return: None
        """
        cell_types = self.__default_mutable_type(cell_types, [])

        for cell in self.cellListByType(*cell_types):
            self.set_step_size_for_cell(model_name=model_name, cell=cell, step_size=step_size)

    @staticmethod
    def set_step_size_for_free_floating_sbml(model_name: str = '', step_size: float = 1.0) -> None:

        """
        Sets integration step size for free floating SBML
        :param model_name: model name
        :param step_size: integration time step
        :return: None
        """

        pg = CompuCellSetup.persistent_globals
        try:
            sbml_solver = pg.free_floating_sbml_simulators[model_name]
        except LookupError:
            return

        sbml_solver.stepSize = step_size

    def timestep_free_floating_sbml(self):
        """
        Integrates forward all free floating SBML solvers
        :return: None
        """
        pg = CompuCellSetup.persistent_globals

        for model_name, rr in pg.free_floating_sbml_simulators.items():
            rr.timestep()

    def timestep_sbml(self):
        """
        Integrates forward all free floating SBML solvers and all sbmlsolvers attached to cells
        :return: None
        """
        self.timestep_cell_sbml()
        self.timestep_free_floating_sbml()

    def get_sbml_simulator(self, model_name: str, cell: object = None) -> Union[object, None]:
        """
        Returns a reference to RoadRunnerPy or None
        :param model_name: model name
        :param cell: CellG cell object
        :return {instance of RoadRunnerPy} or {None}:
        """

        pg = CompuCellSetup.persistent_globals
        if not cell:
            try:

                return pg.free_floating_sbml_simulators[model_name]

            except LookupError:
                return None
        else:
            try:
                dict_attrib = CompuCell.getPyAttrib(cell)
                return dict_attrib['SBMLSolver'][model_name]
            except LookupError:
                return None

    def get_sbml_state(self, model_name: str, cell: object = None) -> Union[None, dict]:
        """
        Returns dictionary-like object representing state of the SBML solver - instance of the RoadRunner.model
        which behaves as a python dictionary but has many entries some of which are non-assignable /non-mutable

        :param model_name: model name
        :param cell: CellG object
        :return {instance of RoadRunner.model}: dict-like object
        """
        # might use roadrunner.SelectionRecord.STATE_VECTOR to limit dictionary iterations
        # to only values which are settable
        # for now, though, we return full rr.model dictionary-like object

        sbml_simulator = self.get_sbml_simulator(model_name, cell)
        try:
            return sbml_simulator.model
        except:
            if cell:
                raise RuntimeError("Could not find model " + model_name + ' attached to cell.id=', cell.id)
            else:
                raise RuntimeError("Could not find model " + model_name + ' in the list of free floating SBML models')

    def get_sbml_state_as_python_dict(self, model_name: str, cell: object = None) -> dict:
        """
        Returns Python dictionary representing state of the SBML solver

        :param model_name: model name
        :param cell: CellG object
        :return : dictionary representing state of the SBML Solver
        """
        return self.get_sbml_state(model_name, cell)

    def set_sbml_state(self, model_name: str, cell: object = None, state: Union[None, dict] = None) -> bool:
        """
        Sets SBML state for the solver - only for advanced uses. Requires detailed knowledge of how underlying
        SBML solver (roadrunner) works
        :param model_name: model name
        :param cell: CellG object
        :param state : dictionary with state variables to set
        :return: None
        """
        state = self.__default_mutable_type(state, {})
        sbml_simulator = self.get_sbml_simulator(model_name, cell)

        if not sbml_simulator:
            return False
        else:

            if state == sbml_simulator.model:  # no need to do anything when all the state changes are done on model
                return True

            for name, value in state.items():
                try:
                    sbml_simulator.model[name] = value
                except:  # in case user decides to set unsettable quantities e.g. reaction rates
                    pass

            return True

    def get_sbml_value(self, model_name: str, value_name: str, cell: object = None) -> float:
        """
        Retrieves value of the SBML state variable
        :param model_name: model name
        :param value_name: name of the state variable
        :param cell: CellG object
        :return: value of the state variable
        """
        sbml_simulator = self.get_sbml_simulator(model_name, cell)
        if not sbml_simulator:
            if cell:
                raise RuntimeError("Could not find model " + model_name + ' attached to cell.id=', cell.id)
            else:
                raise RuntimeError("Could not find model " + model_name + ' in the list of free floating SBML models')
        else:
            return sbml_simulator[value_name]

    def set_sbml_value(self, model_name: str, value_name: str, value: float = 0.0, cell: object = None) -> bool:
        """
        Sets SBML solver state variable
        :param model_name: model name
        :param value_name: name of the stae variable
        :param value: value of the state variable
        :param cell: CellG object
        :return: None
        """
        sbml_simulator = self.get_sbml_simulator(model_name, cell)
        if not sbml_simulator:
            return False
        else:
            sbml_simulator.model[value_name] = value
            return True

    def copy_sbml_simulators(self, from_cell: object, to_cell: object, sbml_names: Union[list, None] = None,
                             options: Union[dict, None] = None):
        """
        Copies SBML solvers (with their states - effectively clones the solver) from one cell to another
        :param from_cell: source CellG cell
        :param to_cell: target CellG cell
        :param sbml_names: list of SBML model name whose solver are to be copied
        :param options: - deprecated - list of SBML solver options
        :return: None
        """
        sbml_names = self.__default_mutable_type(sbml_names, [])
        options = self.__default_mutable_type(options, {})

        sbml_names_to_copy = []
        if not (len(sbml_names)):
            # if user does not specify _sbmlNames we copy all SBML networks
            try:
                dict_attrib = CompuCell.getPyAttrib(from_cell)
                sbml_dict = dict_attrib['SBMLSolver']
                sbml_names_to_copy = list(sbml_dict.keys())
            except LookupError as e:
                pass
        else:
            sbml_names_to_copy = sbml_names

        try:
            dict_attrib_from = CompuCell.getPyAttrib(from_cell)
            sbml_dict_from = dict_attrib_from['SBMLSolver']
        except LookupError:
            return

        try:
            dict_attrib_to = CompuCell.getPyAttrib(to_cell)
            # sbml_dict_to = dict_attrib_to['SBMLSolver']
        except LookupError:
            pass
            # if _toCell does not have SBMLSolver dictionary entry we simply add it
            # dict_attrib_to['SBMLSolver'] = {}
            # sbml_dict_to = dict_attrib_to['SBMLSolver']

        for sbml_name in sbml_names_to_copy:
            rr_from = sbml_dict_from[sbml_name]
            current_state_sbml = sbml_dict_from[sbml_name].getCurrentSBML()
            self.add_sbml_to_cell(
                model_file=rr_from.path,  # necessary to get deserialization working properly
                model_name=sbml_name,
                cell=to_cell,
                step_size=rr_from.stepSize,
                options=options,
                current_state_sbml=current_state_sbml
            )

    def normalize_path(self, path: str) -> str:
        """
        Checks if file exists and if not it joins basepath (path to the root of the cc3d project) with path
        :param path: relative path to CC3D resource
        :return {str}: absolute path to CC3D resource
        """

        path_normalized = path
        try:
            f = open(path_normalized, 'r')
            f.close()
        except IOError:
            if self.simulator.getBasePath() != '':
                path_normalized = os.path.abspath(os.path.join(self.simulator.getBasePath(), path_normalized))

        return path_normalized
