import os
from typing import List, Union
import types
from cc3d.cpp import CompuCell
from cc3d import CompuCellSetup
from deprecated import deprecated
from random import randint
from cc3d.core.logging import log_py


# Test for Antimony installation
try:
    import antimony

    antimony_available = True
except (ImportError, OSError) :
    antimony_available = False

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


def AntimonyTranslatorError(self, getAntimonyMessage=False, *args, **kwrds):
    import inspect
    line = inspect.stack()[1][2]
    call = inspect.stack()[1][4]
    error_string = 'AntimonyTranslatorError line :' + str(line) + ' call:' + str(call)
    if getAntimonyMessage:
        error_message = antimony.getLastError()
        error_string = error_string + ' Antimony returned an error with the following messages: \n' + error_message
    else:
        error_string = error_string + ' Trying to access one of the Antimony translator methods but Antimony libraries (e.g. libAntimony) has not been installed with your CompuCell3D package'
    raise AttributeError(error_string)


class SBMLSolverHelper(object):
    """
    Supporting class for deploying SBML, Antimony and CellML model specification via
    :class:`cc3d.core.PySteppables.SteppableBasePy`
    """

    @classmethod
    def remove_attribute(cls, name):
        return delattr(cls, name)

    def __init__(self):

        # in case user passes simulate options we set the here
        # this dictionary translates old options valid for earlier rr versions to new ones

        self.option_name_dict = {
            'relative': 'relative_tolerance',
            'absolute': 'absolute_tolerance',
            'steps': 'maximum_num_steps'
        }

        if not roadrunner_available:
            sbml_solver_api = ['add_free_floating_sbml', 'add_sbml_to_cell', 'add_sbml_to_link',
                               'add_sbml_to_cell_ids', 'add_sbml_to_cell_types', 'clone_sbml_simulators',
                               'copy_sbml_simulators', 'delete_free_floating_sbml', 'delete_sbml_from_cell',
                               'delete_sbml_from_link', 'delete_sbml_from_cell_ids', 'delete_sbml_from_cell_types',
                               'get_sbml_global_options', 'get_sbml_simulator', 'get_sbml_state',
                               'get_sbml_state_as_python_dict', 'get_sbml_value', 'normalize_path',
                               'set_sbml_global_options', 'set_sbml_state', 'set_sbml_value', 'set_step_size_for_cell',
                               'set_step_size_for_link', 'set_step_size_for_cell_ids', 'set_step_size_for_cell_types',
                               'set_step_size_for_free_floating_sbml',
                               'timestep_cell_sbml', 'timestep_link_sbml', 'timestep_free_floating_sbml',
                               'timestep_sbml']

            for api_name in sbml_solver_api:
                SBMLSolverHelper.remove_attribute(api_name)
                setattr(SBMLSolverHelper, api_name, types.MethodType(SBMLSolverError, SBMLSolverHelper))

        if not antimony_available:
            antimony_translator_api = ['add_antimony_to_cell', 'add_antimony_to_cell_ids',
                                       'add_antimony_to_cell_types', 'add_free_floating_antimony',
                                       'add_antimony_to_link',
                                       'add_cellml_to_cell', 'add_cellml_to_cell_ids', 'add_cellml_to_cell_types',
                                       'add_cellml_to_link',
                                       'add_free_floating_cellml', 'translate_to_sbml_string']

            for api_name in antimony_translator_api:
                SBMLSolverHelper.remove_attribute(api_name)
                setattr(SBMLSolverHelper, api_name, types.MethodType(AntimonyTranslatorError, SBMLSolverHelper))

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

    def set_gillespie_integrator_seed(self, seed):
        if seed > self.get_gillespie_integrator_max_seed():
            raise ValueError(
                f"Seed is greater than max allowed value for the seed : {self.get_gillespie_integrator_max_seed()}")
        CompuCellSetup.persistent_globals.gillespie_integrator_seed = seed

    def get_gillespie_integrator_seed(self) -> Union[int, None]:

        try:
            return CompuCellSetup.persistent_globals.gillespie_integrator_seed
        except AttributeError:
            return None

    def get_gillespie_integrator_max_seed(self) -> int:

        try:
            return CompuCellSetup.persistent_globals.gillespie_integrator_max_seed
        except AttributeError:
            return int(2e9)

    def set_integrator_seed(self, integrator_name: str, rr: RoadRunnerPy):
        max_seed = self.get_gillespie_integrator_max_seed()
        if integrator_name.lower().strip() == "gillespie":
            seed = self.get_gillespie_integrator_seed()
            if seed is not None:
                rr.integrator.seed = seed
            else:
                # unless user requests fixed seed we are randomizing the seed of the gillespie integrators
                rr.integrator.seed = randint(0, int(max_seed))

    @deprecated(version='4.0.0', reason="You should use : add_sbml_to_cell")
    def addSBMLToCell(self, _modelFile='', _modelName='', _cell=None, _stepSize=1.0, _initialConditions={},
                      _coreModelName='', _modelPathNormalized='', _options=None, _currentStateSBML=None):

        return self.add_sbml_to_cell(model_file=_modelFile, model_name=_modelName, cell=_cell, step_size=_stepSize,
                                     initial_conditions=_initialConditions, options=_options,
                                     current_state_sbml=_currentStateSBML)

    def add_sbml_to_cell(self, model_file: str = '', model_string: str = '', model_name: str = '', cell: object = None,
                         step_size: float = 1.0,
                         initial_conditions: Union[None, dict] = None, options: Union[None, dict] = None,
                         current_state_sbml: object = None,
                         integrator: str = None) -> None:
        """
        Attaches :class:`~cc3d.core.RoadRunnerPy.RoadRunnerPy` instance to a particular cell. The sbml solver is stored
        as an element of the cell's dictionary - cell.dict['SBMLSolver'][_modelName]. The function has a dual operation
        mode. When user provides current_state_sbml, cell model_name, step_size the add_sbml_to_cell function creates a
        clone of a solver whose state is described by the current_state_sbml . If current_state_sbml is None
        then the new SBML solver
        is being created,  SBML file (model_file) or string (model_string) loaded and initial conditions are applied.
        It is important to always set
        ste_size to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param str model_file:
            name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path

        :param str model_string: string of SBML file

        :param str model_name:
            name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary

        :param cc3d.cpp.CompuCell.CellG cell: cc3d cell object

        :param float step_size: time step- determines how much in "real" time units timestep() fcn advances SBML solver

        :param dict initial_conditions: initial conditions dictionary, optional

        :param dict options: dictionary that currently only defines what type of ODE solver to choose.
            In the newer versions of RR this might be not necessary. The keys that are supported are the following:
              - absolute - determines absolute tolerance default 1e-10
              - relative - determines relative tolerance default 1e-5
              - stiff - determines if using stiff solver or not default False

        :param current_state_sbml: string representation of the SBML representing current state of the solver.

        :param integrator: name of integrator; passed to ``RoadRunner.setIntegrator()``;
            only applied if ``current_state_sbml`` is None

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})

        core_model_name = model_name
        if core_model_name == '':
            core_model_name, ext = os.path.splitext(os.path.basename(model_file))

        if model_string == '':
            if not model_file:
                log_py(CompuCell.LOG_WARNING,
                       '\n\n\n _modelFile argument not provided to addSBMLToCell. '
                       'This will prevent proper restart of the simulation'
                       'You may ignore this warning if you are not '
                       'serializing simulation for future restarts')

            model_path_normalized = self.normalize_path(model_file)
        else:
            model_path_normalized = ''

        dict_attrib = CompuCell.getPyAttrib(cell)

        sbml_dict = {}
        if 'SBMLSolver' in dict_attrib:
            sbml_dict = dict_attrib['SBMLSolver']
        else:
            dict_attrib['SBMLSolver'] = sbml_dict

        if current_state_sbml is None:
            rr = RoadRunnerPy(_path=model_file, _modelString=model_string)
            # setting stepSize
            rr.stepSize = step_size
            # loading SBML and LLVM-ing it
            rr.loadSBML(_externalPath=model_path_normalized, _modelString=model_string)
            if integrator is not None:
                rr.setIntegrator(name=integrator)
                self.set_integrator_seed(integrator_name=integrator, rr=rr)

        else:
            rr = RoadRunnerPy(sbml=current_state_sbml)
            # setting stepSize
            rr.stepSize = step_size

            # setting up paths - IMPORTANT FOR RESTARTING
            rr.path = model_file
            rr.modelString = model_string
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

    def add_sbml_to_link(self,
                         link: CompuCell.FocalPointPlasticityLinkBase,
                         model_file: str = '',
                         model_string: str = '',
                         model_name: str = '',
                         step_size: float = 1.0,
                         initial_conditions: Union[None, dict] = None,
                         options: Union[None, dict] = None,
                         current_state_sbml: object = None,
                         integrator: str = None) -> None:
        """
        Same functionality as :meth:`add_sbml_to_cell`, but for a link.

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of SBML file
        :param model_name: name of the model - this is a label used to store mode in the cell.dict['SBMLSolver']
            dictionary
        :param link: {FocalPointPlasticityLinkBase object} link object (of any type)
        :param step_size:  time step- determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
            In the newer versions of RR this might be not necessary. The keys that are supported are the following:
        :param current_state_sbml:  string representation  of the SBML representing current state of the solver.
        :param integrator: name of integrator; passed to ``RoadRunner.setIntegrator()``;
            only applied if ``current_state_sbml`` is None
        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})

        core_model_name = model_name
        if core_model_name == '':
            core_model_name, ext = os.path.splitext(os.path.basename(model_file))

        if model_string == '':
            if not model_file:
                log_py(CompuCell.LOG_WARNING,
                       '\n\n\n _modelFile argument not provided to addSBMLToCell. '
                       'This will prevent proper restart of the simulation'
                       'You may ignore this warning if you are not '
                       'serializing simulation for future restarts')

            model_path_normalized = self.normalize_path(model_file)
        else:
            model_path_normalized = ''

        dict_attrib = link.dict

        sbml_dict = {}
        if CompuCell.FocalPointPlasticityLinkBase.__sbml__ in dict_attrib:
            sbml_dict = dict_attrib[CompuCell.FocalPointPlasticityLinkBase.__sbml__]
        else:
            dict_attrib[CompuCell.FocalPointPlasticityLinkBase.__sbml__] = sbml_dict

        if current_state_sbml is None:
            rr = RoadRunnerPy(_path=model_file, _modelString=model_string)
            # setting stepSize
            rr.stepSize = step_size
            # loading SBML and LLVM-ing it
            rr.loadSBML(_externalPath=model_path_normalized, _modelString=model_string)
            if integrator is not None:
                rr.setIntegrator(name=integrator)
                self.set_integrator_seed(integrator_name=integrator, rr=rr)

        else:
            rr = RoadRunnerPy(sbml=current_state_sbml)
            # setting stepSize
            rr.stepSize = step_size

            # setting up paths - IMPORTANT FOR RESTARTING
            rr.path = model_file
            rr.modelString = model_string
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

    def add_antimony_to_cell(self, model_file: str = '', model_string: str = '', model_name: str = '',
                             cell: object = None, step_size: float = 1.0,
                             initial_conditions: Union[None, dict] = None, options: Union[None, dict] = None,
                             current_state_sbml: object = None,
                             integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell`, but with Antimony model specification
        Note that initial conditions can be specified either in the Antimony model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param str model_file: name of the Antimony file
        :param str model_string: string of Antimony file
        :param str model_name: name of the model
        :param cc3d.cpp.CompuCell.CellG cell: cc3d cell object
        :param float step_size: time step
        :param dict initial_conditions: initial conditions dictionary, optional
        :param dict options: dictionary that currently only defines what type of ODE solver to choose.
        :param current_state_sbml: string representation of the SBML representing current state of the solver.
        :return: None
        """
        translated_model_string, main_module_name = self.translate_to_sbml_string(model_file=model_file,
                                                                                  model_string=model_string)
        if model_name == '':
            model_name = main_module_name
        self.add_sbml_to_cell(model_string=translated_model_string, model_name=model_name,
                              cell=cell, step_size=step_size, initial_conditions=initial_conditions,
                              options=options, current_state_sbml=current_state_sbml, integrator=integrator)

    def add_antimony_to_link(self,
                             link: CompuCell.FocalPointPlasticityLinkBase,
                             model_file: str = '',
                             model_string: str = '',
                             model_name: str = '',
                             step_size: float = 1.0,
                             initial_conditions: Union[None, dict] = None,
                             options: Union[None, dict] = None,
                             current_state_sbml: object = None,
                             integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_link`, but with Antimony model specification
        Note that initial conditions can be specified either in the Antimony model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence
        """
        translated_model_string, main_module_name = self.translate_to_sbml_string(model_file=model_file,
                                                                                  model_string=model_string)
        if model_name == '':
            model_name = main_module_name
        self.add_sbml_to_link(link=link, model_string=translated_model_string, model_name=model_name,
                              step_size=step_size, initial_conditions=initial_conditions,
                              options=options, current_state_sbml=current_state_sbml, integrator=integrator)

    def add_cellml_to_cell(self, model_file: str = '', model_string: str = '', model_name: str = '',
                           cell: object = None, step_size: float = 1.0,
                           initial_conditions: Union[None, dict] = None, options: Union[None, dict] = None,
                           current_state_sbml: object = None,
                           integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell`, but with CellML model specification
        Note that initial conditions can be specified either in the CellML model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param str model_file: name of the CellML file
        :param str model_string: string of CellML file
        :param str model_name: name of the model
        :param cc3d.cpp.CompuCell.CellG cell: cc3d cell object
        :param float step_size: time step
        :param dict initial_conditions: initial conditions dictionary, optional
        :param dict options: dictionary that currently only defines what type of ODE solver to choose.
        :param current_state_sbml: string representation of the SBML representing current state of the solver.
        :return: None
        """
        self.add_antimony_to_cell(model_file=model_file, model_string=model_string, model_name=model_name,
                                  cell=cell, step_size=step_size,
                                  initial_conditions=initial_conditions, options=options,
                                  current_state_sbml=current_state_sbml, integrator=integrator)

    def add_cellml_to_link(self,
                           link: CompuCell.FocalPointPlasticityLinkBase,
                           model_file: str = '',
                           model_string: str = '',
                           model_name: str = '',
                           step_size: float = 1.0,
                           initial_conditions: Union[None, dict] = None,
                           options: Union[None, dict] = None,
                           current_state_sbml: object = None,
                           integrator: str = None) -> None:
        """
        Same as add_sbml_to_link, but with CellML model specification
        Note that initial conditions can be specified either in the CellML model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence
        """
        self.add_antimony_to_link(link=link, model_file=model_file, model_string=model_string, model_name=model_name,
                                  step_size=step_size,
                                  initial_conditions=initial_conditions, options=options,
                                  current_state_sbml=current_state_sbml, integrator=integrator)

    @deprecated(version='4.0.0', reason="You should use : get_sbml_global_options")
    def getSBMLGlobalOptions(self):
        return self.get_sbml_global_options()

    def get_sbml_global_options(self):
        """
        returns global options for the SBML solver - deprecated as newer version of CC3D

        :return: global SBML solver options
        :rtype: dict
        """
        pg = CompuCellSetup.persistent_globals
        return pg.global_sbml_simulator_options

    @deprecated(version='4.0.0', reason="You should use : set_sbml_global_options")
    def setSBMLGlobalOptions(self, _options):
        return self.set_sbml_global_options(_options)

    def set_sbml_global_options(self, options: dict) -> None:
        """
        Deprecated - sets global SBML options

        :param options:
        :return: None
        """

        pg = CompuCellSetup.persistent_globals
        pg.global_sbml_simulator_options = options

    @deprecated(version='4.0.0', reason="You should use : add_sbml_to_cell_types")
    def addSBMLToCellTypes(self, _modelFile='', _modelName='', _types=[], _stepSize=1.0, _initialConditions={},
                           _options={}):
        return self.add_sbml_to_cell_types(model_file=_modelFile, model_name=_modelName, cell_types=_types,
                                           step_size=_stepSize, initial_conditions=_initialConditions, options=_options)

    def add_sbml_to_cell_types(self, model_file: str = '', model_string: str = '', model_name: str = '',
                               cell_types: Union[None, list] = None,
                               step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                               options: Union[None, dict] = None,
                               integrator: str = None) -> None:
        """
        Adds SBML Solver to all cells of given cell type - internally it calls addSBMLToCell(fcn).
        Used during initialization of the simulation. It is important to always set
        _stepSize to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of SBML file
        :param model_name:
            name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary

        :param cell_types: list of cell types
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
            In the newer versions of RR this might be not necessary. The keys that are supported are the following:
              - absolute - determines absolute tolerance default 1e-10
              - relative - determines relative tolerance default 1e-5
              - stiff - determines if using stiff solver or not default False

        :param integrator: name of integrator; passed to ``RoadRunner.setIntegrator()``

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})
        cell_types = self.__default_mutable_type(cell_types, [])

        if 'steps' in list(options.keys()):
            log_py(CompuCell.LOG_WARNING,
                   '-----------WARNING-----------------\n'
                   ' steps option for SBML solver is deprecated.')

        for cell in self.cellListByType(*cell_types):
            self.add_sbml_to_cell(model_file=model_file, model_string=model_string, model_name=model_name, cell=cell,
                                  step_size=step_size,
                                  initial_conditions=initial_conditions, options=options, integrator=integrator)

    def add_antimony_to_cell_types(self, model_file: str = '', model_string: str = '',
                                   model_name: str = '', cell_types: Union[None, list] = None,
                                   step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                                   options: Union[None, dict] = None,
                                   integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell_types`, but with Antimony model specification
        Note that initial conditions can be specified either in the Antimony model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param str model_file: name of the Antimony file
        :param str model_string: string of Antimony file
        :param str model_name: name of the model
        :param cell_types: list of cell types, optional
        :type cell_types: list of int
        :param float step_size: time step
        :param dict initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        translated_model_string, main_module_name = self.translate_to_sbml_string(model_file=model_file,
                                                                                  model_string=model_string)
        if model_name == '':
            model_name = main_module_name
        self.add_sbml_to_cell_types(model_string=translated_model_string, model_name=model_name,
                                    cell_types=cell_types, step_size=step_size,
                                    initial_conditions=initial_conditions, options=options, integrator=integrator)

    def add_cellml_to_cell_types(self, model_file: str = '', model_string: str = '',
                                 model_name: str = '', cell_types: Union[None, list] = None,
                                 step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                                 options: Union[None, dict] = None,
                                 integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell_types`, but with CellML model specification
        Note that initial conditions can be specified either in the CellML model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param str model_file: name of the CellML file
        :param str model_string: string of CellML file
        :param str model_name: name of the model
        :param cell_types: list of cell types, optional
        :type cell_types: list of int
        :param float step_size: time step
        :param dict initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        self.add_antimony_to_cell_types(model_file=model_file, model_string=model_string,
                                        model_name=model_name, cell_types=cell_types, step_size=step_size,
                                        initial_conditions=initial_conditions, options=options, integrator=integrator)

    @deprecated(version='4.0.0', reason="You should use : add_sbml_to_cell_ids")
    def addSBMLToCellIds(self, _modelFile, _modelName='', _ids=[], _stepSize=1.0, _initialConditions={}, _options={}):
        return self.add_sbml_to_cell_ids(model_file=_modelFile, model_name=_modelName, cell_ids=_ids,
                                         step_size=_stepSize, initial_conditions=_initialConditions, options=_options)

    def add_sbml_to_cell_ids(self, model_file: str = '', model_string: str = '', model_name: str = '',
                             cell_ids: Union[None, list] = None,
                             step_size: float = 1.0, initial_conditions: Union[None, dict] = None,
                             options: Union[None, dict] = None,
                             integrator: str = None) -> None:
        """
        Adds SBML Solver to all cells of given cell ids - internally it calls add_sbml_to_cell fcn.
        Used during initialization of the simulation. It is important to always set
        step_size to make sure that after calling timestep() fcn the solver advances appropriate delta time

        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path

        :param model_string: string of SBML file

        :param model_name: name of the model - this is a label used to store mode in the
         cell.dict['SBMLSolver'] dictionary

        :param cell_ids: list of cell ids

        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver

        :param initial_conditions: initial conditions dictionary

        :param options: dictionary that currently only defines what type of ODE solver to choose.
            In the newer versions of RR this might be not necessary. The keys that are supported are the following:
              - absolute - determines absolute tolerance default 1e-10
              - relative - determines relative tolerance default 1e-5
              - stiff - determines if using stiff solver or not default False

        :param integrator: name of integrator; passed to ``RoadRunner.setIntegrator()``

        :return: None
        """

        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})
        cell_ids = self.__default_mutable_type(cell_ids, [])

        if 'steps' in list(options.keys()):
            log_py(CompuCell.LOG_WARNING,
                   '-----------WARNING-----------------\n\n steps option for SBML solver is deprecated.')

        for cell_id in cell_ids:
            cell = self.inventory.attemptFetchingCellById(cell_id)
            if not cell:
                continue

            self.add_sbml_to_cell(model_file=model_file, model_string=model_string, model_name=model_name, cell=cell,
                                  step_size=step_size,
                                  initial_conditions=initial_conditions, options=options, integrator=integrator)

    def add_antimony_to_cell_ids(self, model_file: str = '', model_string: str = '', model_name: str = '',
                                 cell_ids: Union[None, list] = None, step_size: float = 1.0,
                                 initial_conditions: Union[None, dict] = None,
                                 options: Union[None, dict] = None,
                                 integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell_ids`, but with Antimony model specification
        Note that initial conditions can be specified either in the Antimony model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param model_file:
            name of the Antimony file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of Antimony file
        :param model_name: name of the model
        :param cell_ids: list of cell ids
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        translated_model_string, main_module_name = self.translate_to_sbml_string(model_file=model_file,
                                                                                  model_string=model_string)
        if model_name == '':
            model_name = main_module_name
        self.add_sbml_to_cell_ids(model_string=translated_model_string, model_name=model_name,
                                  cell_ids=cell_ids, step_size=step_size, initial_conditions=initial_conditions,
                                  options=options, integrator=integrator)

    def add_cellml_to_cell_ids(self, model_file: str = '', model_string: str = '', model_name: str = '',
                               cell_ids: Union[None, list] = None, step_size: float = 1.0,
                               initial_conditions: Union[None, dict] = None,
                               options: Union[None, dict] = None,
                               integrator: str = None) -> None:
        """
        Same as :meth:`add_sbml_to_cell_ids`, but with CellML model specification
        Note that initial conditions can be specified either in the CellML model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param model_file:
            name of the CellML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of CellML file
        :param model_name: name of the model
        :param cell_ids: list of cell ids
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        self.add_antimony_to_cell_ids(model_file=model_file, model_string=model_string,
                                      model_name=model_name, cell_ids=cell_ids, step_size=step_size,
                                      initial_conditions=initial_conditions,
                                      options=options, integrator=integrator)

    @deprecated(version='4.0.0', reason="You should use : add_free_floating_sbml")
    def addFreeFloatingSBML(self, _modelFile, _modelName, _stepSize=1.0, _initialConditions={}, _options={}):
        return self.add_free_floating_sbml(model_file=_modelFile, model_name=_modelName, step_size=_stepSize,
                                           initial_conditions=_initialConditions, options=_options)

    def add_free_floating_sbml(self,
                               model_file: str = '',
                               model_string: str = '',
                               model_name: str = '',
                               step_size: float = 1.0,
                               initial_conditions: Union[None, dict] = None,
                               options: Union[None, dict] = None,
                               integrator: str = None):
        """
        Adds free floating SBML model - not attached to any cell. The model will be identified/referenced by the _modelName
        :param model_file: name of the SBML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of SBML file
        :param model_name:
            name of the model - this is a label used to store mode in the cell.dict['SBMLSolver'] dictionary
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
            In the newer versions of RR this might be not necessary. The keys that are supported are the following:
              - absolute - determines absolute tolerance default 1e-10
              - relative - determines relative tolerance default 1e-5
              - stiff - determines if using stiff solver or not default False

        :param integrator: name of integrator; passed to ``RoadRunner.setIntegrator()``

        :return: None
        """
        initial_conditions = self.__default_mutable_type(initial_conditions, {})
        options = self.__default_mutable_type(options, {})

        if model_string == '':
            model_path_normalized = self.normalize_path(model_file)
            try:
                f = open(model_path_normalized, 'r')
                f.close()
            except IOError as e:
                if self.simulator.getBasePath() != '':
                    model_path_normalized = os.path.abspath(
                        os.path.join(self.simulator.getBasePath(), model_path_normalized))
        else:
            model_path_normalized = ''

        rr = RoadRunnerPy(_path=model_file, _modelString=model_string)
        rr.loadSBML(_externalPath=model_path_normalized, _modelString=model_string)

        # setting stepSize
        rr.stepSize = step_size
        if integrator is not None:
            rr.setIntegrator(name=integrator)
            self.set_integrator_seed(integrator_name=integrator, rr=rr)

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

    def add_free_floating_antimony(self, model_file: str = '', model_string: str = '',
                                   model_name: str = '', step_size: float = 1.0,
                                   initial_conditions: Union[None, dict] = None,
                                   options: Union[None, dict] = None,
                                   integrator: str = None):
        """
        Same as :meth:`add_free_floating_sbml`, but with Antimony model specification
        Note that initial conditions can be specified either in the Antimony model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param model_file: name of the Antimony file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of Antimony file
        :param model_name: name of the model
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        translated_model_string, main_module_name = self.translate_to_sbml_string(model_file=model_file,
                                                                                  model_string=model_string)
        if model_name == '':
            model_name = main_module_name
        self.add_free_floating_sbml(model_file=model_file, model_string=translated_model_string,
                                    model_name=model_name, step_size=step_size,
                                    initial_conditions=initial_conditions, options=options, integrator=integrator)

    def add_free_floating_cellml(self, model_file: str = '', model_string: str = '',
                                 model_name: str = '', step_size: float = 1.0,
                                 initial_conditions: Union[None, dict] = None,
                                 options: Union[None, dict] = None,
                                 integrator: str = None):
        """
        Same as :meth:`add_free_floating_sbml`, but with CellML model specification
        Note that initial conditions can be specified either in the CellML model specification,
        or with initial_conditions. If both are specified, initial_conditions takes precedence

        :param model_file: name of the CellML file - can be relative path (e.g. Simulation/file.sbml) or absolute path
        :param model_string: string of CellML file
        :param model_name: name of the model
        :param step_size: time step - determines how much in "real" time units timestep() fcn advances SBML solver
        :param initial_conditions: initial conditions dictionary
        :param options: dictionary that currently only defines what type of ODE solver to choose.
        :return: None
        """
        self.add_free_floating_antimony(model_file=model_file, model_string=model_string,
                                        model_name=model_name, step_size=step_size,
                                        initial_conditions=initial_conditions, options=options, integrator=integrator)

    @deprecated(version='4.0.0', reason="You should use : delete_sbml_from_cell_ids")
    def deleteSBMLFromCellIds(self, _modelName, _ids=[]):
        return self.delete_sbml_from_cell_ids(model_name=_modelName, cell_ids=_ids)

    def delete_sbml_from_cell_ids(self, model_name: str, cell_ids: Union[None, list] = None) -> None:
        """
        Deletes SBML model from cells whose ids match those stored in the _ids list

        :param model_name: model name
        :param cell_ids: list of cell ids
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

    @deprecated(version='4.0.0', reason="You should use : delete_sbml_from_cell_types")
    def deleteSBMLFromCellTypes(self, _modelName, _types=[]):
        return self.delete_sbml_from_cell_types(model_name=_modelName, cell_types=_types)

    def delete_sbml_from_cell_types(self, model_name: str, cell_types: Union[None, list] = None) -> None:
        """
        Deletes  SBML model from cells whose type match those stered in the cell_types list

        :param str model_name: model name
        :param cell_types: list of cell types
        :type cell_types: list of int
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

    @deprecated(version='4.0.0', reason="You should use : delete_sbml_from_cell")
    def deleteSBMLFromCell(self, _modelName='', _cell=None):
        return self.delete_sbml_from_cell(model_name=_modelName, cell=_cell)

    def delete_sbml_from_cell(self, model_name: str = '', cell: object = None) -> None:
        """
        Deletes SBML from a particular cell

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell obj
        :return: None
        """

        dict_attrib = CompuCell.getPyAttrib(cell)
        try:
            sbml_dict = dict_attrib['SBMLSolver']
            del sbml_dict[model_name]
        except LookupError:
            pass

    def delete_sbml_from_link(self,
                              model_name: str,
                              link: CompuCell.FocalPointPlasticityLinkBase) -> None:
        """
        Deletes SBML from a particular link

        :param model_name: model name
        :param link: CellG cell obj
        :return: None
        """

        try:
            sbml_dict = link.dict['SBMLSolver']
            del sbml_dict[model_name]
        except LookupError:
            pass

    @deprecated(version='4.0.0', reason="You should use : delete_free_floating_sbml")
    def deleteFreeFloatingSBML(self, _modelName):
        return self.delete_free_floating_sbml(model_name=_modelName)

    def delete_free_floating_sbml(self, model_name: str) -> None:
        """
        Deletes free floating SBLM model

        :param model_name: model name
        :return: None
        """
        pg = CompuCellSetup.persistent_globals

        try:
            del pg.free_floating_sbml_simulators[model_name]
        except LookupError:
            pass

    @deprecated(version='4.0.0', reason="You should use : timestep_cell_sbml")
    def timestepCellSBML(self):
        return self.timestep_cell_sbml()

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

    def timestep_link_sbml(self):
        """
        Advanced models stored as attributes of links

        :return: None
        """
        if self.focal_point_plasticity_plugin is None:
            return

        def inner(link_list):
            for link in link_list:
                try:
                    sbml_dict = link.dict[CompuCell.FocalPointPlasticityLinkBase.__sbml__]
                    for rr in sbml_dict.values():
                        rr.timestep()
                except KeyError:
                    pass

        inner(self.get_focal_point_plasticity_link_list())
        inner(self.get_focal_point_plasticity_internal_link_list())
        inner(self.get_focal_point_plasticity_anchor_list())

    @deprecated(version='4.0.0', reason="You should use : set_step_size_for_cell")
    def setStepSizeForCell(self, _modelName='', _cell=None, _stepSize=1.0):
        return self.set_step_size_for_cell(model_name=_modelName, cell=_cell, step_size=_stepSize)

    def set_step_size_for_cell(self, model_name: str = '', cell: object = None, step_size: float = 1.0):
        """
        Sets integration step size for SBML model attached to _cell

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param float step_size: integration step size
        :return: None
        """
        dict_attrib = CompuCell.getPyAttrib(cell)

        try:
            sbmlSolver = dict_attrib['SBMLSolver'][model_name]
        except LookupError:
            return

        sbmlSolver.stepSize = step_size

    def set_step_size_for_link(self,
                               model_name: str,
                               link: CompuCell.FocalPointPlasticityLinkBase,
                               step_size: float):
        """
        Sets integration step size for SBML model attached to a link

        :param model_name: model name
        :param link: link object
        :param step_size: integration step size
        :return: None
        """
        try:
            sbml_solver = link.dict[CompuCell.FocalPointPlasticityLinkBase.__sbml__][model_name]
        except KeyError:
            return
        sbml_solver.stepSize = step_size

    @deprecated(version='4.0.0', reason="You should use : set_step_size_for_cell_ids")
    def setStepSizeForCellIds(self, _modelName='', _ids=[], _stepSize=1.0):
        return self.set_step_size_for_cell_ids(model_name=_modelName, cell_ids=_ids, step_size=_stepSize)

    def set_step_size_for_cell_ids(self, model_name: str = '', cell_ids: Union[None, list] = None,
                                   step_size: float = 1.0) -> None:
        """
        Sets integration step size for SBML model attached to cells of given ids

        :param str model_name:  model name
        :param cell_ids: list of cell ids, optional
        :type cell_ids: list of int
        :param float step_size: integration step size
        :return: None
        """
        cell_ids = self.__default_mutable_type(cell_ids, [])

        for cell_id in cell_ids:
            cell = self.inventory.attemptFetchingCellById(cell_id)
            if not cell:
                continue

            self.set_step_size_for_cell(model_name=model_name, cell=cell, step_size=step_size)

    @deprecated(version='4.0.0', reason="You should use : set_step_size_for_cell_types")
    def setStepSizeForCellTypes(self, _modelName='', _types=[], _stepSize=1.0):
        return self.set_step_size_for_cell_types(model_name=_modelName, cell_types=_types, step_size=_stepSize)

    def set_step_size_for_cell_types(self, model_name: str = '', cell_types: Union[None, list] = None,
                                     step_size=1.0) -> None:
        """
        Sets integration step size for SBML model attached to cells of given cell types

        :param str model_name: model name
        :param cell_types: list of cell types
        :type cell_types: list of int
        :param float step_size: integration step size
        :return: None
        """
        cell_types = self.__default_mutable_type(cell_types, [])

        for cell in self.cellListByType(*cell_types):
            self.set_step_size_for_cell(model_name=model_name, cell=cell, step_size=step_size)

    @deprecated(version='4.0.0', reason="You should use : set_step_size_for_free_floating_sbml")
    def setStepSizeForFreeFloatingSBML(self, _modelName='', _stepSize=1.0):
        return SBMLSolverHelper.set_step_size_for_free_floating_sbml(model_name=_modelName, step_size=_stepSize)

    def set_step_size_for_free_floating_sbml(self, model_name: str = '', step_size: float = 1.0) -> None:

        """
        Sets integration step size for free floating SBML

        :param str model_name: model name
        :param float step_size: integration time step
        :return: None
        """

        pg = CompuCellSetup.persistent_globals
        try:
            sbml_solver = pg.free_floating_sbml_simulators[model_name]
        except LookupError:
            return

        sbml_solver.stepSize = step_size

    @deprecated(version='4.0.0', reason="You should use : timestep_free_floating_sbml")
    def timestepFreeFloatingSBML(self):
        return self.timestep_free_floating_sbml()

    def timestep_free_floating_sbml(self):
        """
        Integrates forward all free floating SBML solvers

        :return: None
        """
        pg = CompuCellSetup.persistent_globals

        for model_name, rr in pg.free_floating_sbml_simulators.items():
            rr.timestep()

    @deprecated(version='4.0.0', reason="You should use : timestep_sbml")
    def timestepSBML(self):
        return self.timestep_sbml()

    def timestep_sbml(self):
        """
        Integrates forward all free floating SBML solvers and all sbmlsolvers attached to cells

        :return: None
        """
        self.timestep_cell_sbml()
        self.timestep_link_sbml()
        self.timestep_free_floating_sbml()

    @deprecated(version='4.0.0', reason="You should use : get_sbml_simulator")
    def getSBMLSimulator(self, _modelName, _cell=None):
        return self.get_sbml_simulator(model_name=_modelName, cell=_cell)

    @classmethod
    def get_sbml_simulator(cls,
                           model_name: str,
                           cell: CompuCell.CellG = None,
                           link: CompuCell.FocalPointPlasticityLinkBase = None) -> Union[object, None]:
        """
        Returns a reference to :class:`~cc3d.core.RoadRunnerPy.RoadRunnerPy` or None

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :return instance of RoadRunnerPy or None
        :rtype: RoadRunnerPy or None
        """

        pg = CompuCellSetup.persistent_globals
        if not cell and not link:
            try:

                return pg.free_floating_sbml_simulators[model_name]

            except LookupError:
                return None
        elif cell:
            try:
                dict_attrib = CompuCell.getPyAttrib(cell)
                return dict_attrib['SBMLSolver'][model_name]
            except LookupError:
                return None
        else:
            try:
                return getattr(link.sbml, model_name)
            except KeyError:
                return None

    @deprecated(version='4.0.0', reason="You should use : get_sbml_state")
    def getSBMLState(self, _modelName, _cell=None):
        return self.get_sbml_state(model_name=_modelName, cell=_cell)

    def get_sbml_state(self,
                       model_name: str,
                       cell: CompuCell.CellG = None,
                       link: CompuCell.FocalPointPlasticityLinkBase = None) -> Union[None, dict]:
        """
        Returns dictionary-like object representing state of the SBML solver - instance of the RoadRunner.model
        which behaves as a python dictionary but has many entries some of which are non-assignable /non-mutable

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :return: instance of RoadRunner.model
        :rtype: dict-like object
        """
        # might use roadrunner.SelectionRecord.STATE_VECTOR to limit dictionary iterations
        # to only values which are settable
        # for now, though, we return full rr.model dictionary-like object

        sbml_simulator = self.get_sbml_simulator(model_name, cell, link=link)
        try:
            return sbml_simulator.model
        except:
            if cell:
                raise RuntimeError("Could not find model " + model_name + ' attached to cell.id=', cell.id)
            else:
                raise RuntimeError("Could not find model " + model_name + ' in the list of free floating SBML models')

    @deprecated(version='4.0.0', reason="You should use : get_sbml_state_as_python_dict")
    def getSBMLStateAsPythonDict(self, _modelName, _cell=None):
        return self.get_sbml_state_as_python_dict(model_name=_modelName, cell=_cell)

    def get_sbml_state_as_python_dict(self,
                                      model_name: str,
                                      cell: CompuCell.CellG = None,
                                      link: CompuCell.FocalPointPlasticityLinkBase = None) -> dict:
        """
        Returns Python dictionary representing state of the SBML solver

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :return : dictionary representing state of the SBML Solver
        :rtype: dict
        """
        return self.get_sbml_state(model_name=model_name, cell=cell, link=link)

    @deprecated(version='4.0.0', reason="You should use : set_sbml_state")
    def setSBMLState(self, _modelName, _cell=None, _state={}):
        return self.set_sbml_state(model_name=_modelName, cell=_cell, state=_state)

    def set_sbml_state(self,
                       model_name: str,
                       cell: CompuCell.CellG = None,
                       link: CompuCell.FocalPointPlasticityLinkBase = None,
                       state: Union[None, dict] = None) -> bool:
        """
        Sets SBML state for the solver - only for advanced uses. Requires detailed knowledge of how underlying
        SBML solver (roadrunner) works

        :param str model_name: model name
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :param dict state: dictionary with state variables to set, optional
        :return: None
        """
        state = self.__default_mutable_type(state, {})
        sbml_simulator = self.get_sbml_simulator(model_name=model_name, cell=cell, link=link)

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

    @deprecated(version='4.0.0', reason="You should use : get_sbml_value")
    def getSBMLValue(self, _modelName, _valueName='', _cell=None):
        return self.get_sbml_value(model_name=_modelName, value_name=_valueName, cell=_cell)

    def get_sbml_value(self,
                       model_name: str,
                       value_name: str,
                       cell: CompuCell.CellG = None,
                       link: CompuCell.FocalPointPlasticityLinkBase = None) -> float:
        """
        Retrieves value of the SBML state variable

        :param str model_name: model name
        :param str value_name: name of the state variable
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :return: value of the state variable
        :rtype: float
        """
        sbml_simulator = self.get_sbml_simulator(model_name=model_name, cell=cell, link=link)
        if not sbml_simulator:
            if cell:
                raise RuntimeError("Could not find model " + model_name + ' attached to cell.id=', cell.id)
            else:
                raise RuntimeError("Could not find model " + model_name + ' in the list of free floating SBML models')
        else:
            return sbml_simulator[value_name]

    @deprecated(version='4.0.0', reason="You should use : set_sbml_value")
    def setSBMLValue(self, _modelName, _valueName='', _value=0.0, _cell=None):
        return self.set_sbml_value(model_name=_modelName, value_name=_valueName, value=_value, cell=_cell)

    def set_sbml_value(self,
                       model_name: str,
                       value_name: str,
                       value: float = 0.0,
                       cell: CompuCell.CellG = None,
                       link: CompuCell.FocalPointPlasticityLinkBase = None) -> bool:
        """
        Sets SBML solver state variable

        :param str model_name: model name
        :param str value_name: name of the stae variable
        :param float value: value of the state variable
        :param cc3d.cpp.CompuCell.CellG cell: cell object
        :param link: link object
        :return: True if set
        :rtype: bool
        """
        sbml_simulator = self.get_sbml_simulator(model_name=model_name, cell=cell, link=link)
        if not sbml_simulator:
            return False
        else:
            sbml_simulator.model[value_name] = value
            return True

    @deprecated(version='4.0.0', reason="You should use : copy_sbml_simulators")
    def copySBMLs(self, _fromCell, _toCell, _sbmlNames=[], _options=None):
        return self.copy_sbml_simulators(from_cell=_fromCell, to_cell=_toCell, sbml_names=_sbmlNames, options=_options)

    @deprecated(version='4.2.5', reason="You should use : clone_sbml_simulators")
    def copy_sbml_simulators(self,
                             from_cell: CompuCell.CellG,
                             to_cell: CompuCell.CellG,
                             sbml_names: Union[list, None] = None,
                             options: Union[dict, None] = None):
        """
        Copies SBML solvers (with their states - effectively clones the solver) from one cell to another

        :param cc3d.cpp.CompuCell.CellG from_cell: source cell
        :param cc3d.cpp.CompuCell.CellG to_cell: target cell
        :param sbml_names: list of SBML model name whose solver are to be copied, optional
        :type sbml_names: list of str
        :param options: Deprecated - list of SBML solver options
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
            # Assume same integrator; user can decide what to do with settings
            rr_to = self.get_sbml_simulator(model_name=sbml_name, cell=to_cell)
            rr_to.setIntegrator(rr_from.getIntegrator().getName())

    def clone_sbml_simulators(self,
                              from_obj: Union[CompuCell.CellG, CompuCell.FocalPointPlasticityLinkBase],
                              to_obj: Union[CompuCell.CellG, CompuCell.FocalPointPlasticityLinkBase],
                              sbml_names: Union[List[str], None] = None) -> None:
        """
        Copies SBML solvers (with their states) - from one object to another.

        Currently supported object types are
            - CompuCell.CellG
            - CompuCell.FocalPointPlasticityLinkBase

        :param from_obj: source object with the SBML solvers to be copied
        :param to_obj: target object receiving the copied SBML solvers
        :param sbml_names:
        :type sbml_names: list or None
        :return: None
        """
        sbml_names = self.__default_mutable_type(sbml_names, [])

        supported_types = [CompuCell.CellG,
                           CompuCell.FocalPointPlasticityLinkBase]

        supported_from = False
        supported_to = False
        for st in supported_types:
            if isinstance(supported_from, st):
                supported_from = True
            if isinstance(supported_to, st):
                supported_to = True
        if not supported_from:
            raise TypeError('Source object type is not supported')
        if not supported_to:
            raise TypeError('Target object type is not supported')

        if isinstance(from_obj, CompuCell.CellG):
            dict_attrib_from = from_obj.dict
            if 'SBMLSolver' not in dict_attrib_from.keys():
                return
            sbml_dict_from = dict_attrib_from['SBMLSolver']
        else:
            dict_attrib_from = from_obj.dict
            if CompuCell.FocalPointPlasticityLinkBase.__sbml__ not in dict_attrib_from.keys():
                return
            sbml_dict_from = dict_attrib_from[CompuCell.FocalPointPlasticityLinkBase.__sbml__]

        sbml_names_to_copy = []
        if not (len(sbml_names)):
            # if user does not specify _sbmlNames we copy all SBML networks
            try:
                sbml_names_to_copy = list(sbml_dict_from.keys())
            except LookupError:
                pass
        else:
            sbml_names_to_copy = sbml_names

        for sbml_name in sbml_names_to_copy:
            rr_from = sbml_dict_from[sbml_name]
            current_state_sbml = sbml_dict_from[sbml_name].getCurrentSBML()
            if isinstance(to_obj, CompuCell.CellG):
                self.add_sbml_to_cell(
                    model_file=rr_from.path,  # necessary to get deserialization working properly
                    model_name=sbml_name,
                    step_size=rr_from.stepSize,
                    current_state_sbml=current_state_sbml,
                    cell=to_obj
                )
                # Assume same integrator; user can decide what to do with settings
                rr_to = self.get_sbml_simulator(model_name=sbml_name, cell=to_obj)
            else:
                self.add_sbml_to_link(
                    model_file=rr_from.path,  # necessary to get deserialization working properly
                    model_name=sbml_name,
                    step_size=rr_from.stepSize,
                    current_state_sbml=current_state_sbml,
                    link=to_obj,
                )
                rr_to = self.get_sbml_simulator(model_name=sbml_name, link=to_obj)
            rr_to.setIntegrator(rr_from.getIntegrator().getName())

    @deprecated(version='4.0.0', reason="You should use : normalize_path")
    def normalizePath(self, _path):
        return self.normalize_path(path=_path)

    def normalize_path(self, path: str) -> str:
        """
        Checks if file exists and if not it joins basepath (path to the root of the cc3d project) with path

        :param str path: relative path to CC3D resource
        :return: absolute path to CC3D resource
        :rtype: str
        """

        path_normalized = path
        try:
            f = open(path_normalized, 'r')
            f.close()
        except IOError:
            if self.simulator.getBasePath() != '':
                path_normalized = os.path.abspath(os.path.join(self.simulator.getBasePath(), path_normalized))

        return path_normalized

    def translate_to_sbml_string(self, model_file: str = '', model_string: str = ''):
        """
        Returns string of SBML model specification translated from Antimony or CellML model specification file or string

        :param str model_file: relative path to model specification file
        :param str model_string: model specification string
        :return: string of SBML model specification, string of main module name
        :rtype: (str, str)
        """

        # Just to be sure, call clear previous loads
        antimony.clearPreviousLoads()

        # Loading from model string or file?
        if model_file == '':
            res_load = antimony.loadString(model_string)
        else:
            model_path_normalized = self.normalize_path(model_file)
            res_load = antimony.loadFile(model_path_normalized)

        if res_load == -1:
            AntimonyTranslatorError(self, getAntimonyMessage=True)

        # Get main loaded module
        main_module_name = antimony.getMainModuleName()
        if not main_module_name:
            AntimonyTranslatorError(self, getAntimonyMessage=True)

        # Return string of SBML model specification
        translated_model_string = antimony.getSBMLString(main_module_name)
        if not translated_model_string:
            AntimonyTranslatorError(self, getAntimonyMessage=True)
        else:
            return translated_model_string, main_module_name

