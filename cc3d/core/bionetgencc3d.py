from copy import deepcopy
import os
import random
import shutil
import sys
import tempfile
import typing

from cc3d import CompuCellSetup
from cc3d.core.RoadRunnerPy import RoadRunnerPy
from cc3d.cpp.CompuCell import getPyAttrib

try:
    from bionetgen.xmlapi.model import bngmodel
    bng_available = True
except ImportError:
    bng_available = False


# These are taken from CellG Python class definition
__bng_fetcher__ = '__bng_fetcher'
__bng_solvers__ = '__bng_solvers'
# Supported BNG solver types
__sim_type_roadrunner__ = 'libRR'


class BNGUnavailableError(EnvironmentError):
    pass


class BNGParserError(ValueError):
    pass


class GlobalBNGFetcher:
    def __getattribute__(self, item):
        try:
            return CompuCellSetup.persistent_globals.free_floating_bng_simulators[item]
        except KeyError:
            return None


class BNGModelCC3D(bngmodel):
    def __init__(self, model_file: str = '', model_string: str = '', model_name: str = ''):
        """
        Derived class for supporting bngmodel instances in CC3D
        :param model_file: {str} path to BioNetGen model file
        :param model_string: {str} BioNetGen model string
        :param model_name: {str} BioNetGen model name
        """
        if not bng_available:
            raise BNGUnavailableError("BioNetGen not installed.")

        rnd_fp = None
        # If specs are from a model string, write model string to temporary file and set flag to remove
        if len(model_string) > 0:
            pg = CompuCellSetup.persistent_globals
            rnd_fp = os.path.join(pg.output_directory, f'_tmp_bng_{random.randint(0, sys.maxsize)}.bngl')
            with open(rnd_fp, "w", newline="") as f:
                f.write(model_string)
            model_file = rnd_fp

        self.model_name = model_name

        bngmodel.__init__(self, bngl_model=model_file)

        if rnd_fp is not None:
            os.remove(rnd_fp)

        self.simulator = None

        self._sbml_model_string = None

    @property
    def sbml_model_string(self) -> str:
        """
        Returns the BioNetGen model in SBML
        :return: {str} SBML model string
        """
        if self._sbml_model_string is None:
            self.add_action("generate_network", [("overwrite", 1)])
            self.add_action("writeSBML", [])
            # temporary folder
            tdir = tempfile.mkdtemp()
            tpath = os.path.join(tdir, 'tmp')
            print("Making temporaries:", tpath)
            # write the sbml
            self.write_xml(tpath, xml_type="sbml")
            # TODO: Only clear the writeSBML action
            # by adding a mechanism to do so
            self.actions.clear_actions()

            # Get the model string
            with open(tpath, 'r') as f:
                self._sbml_model_string = f.read()
            shutil.rmtree(tdir)
        return self._sbml_model_string

    def _setup_roadrunner(self) -> RoadRunnerPy:
        """
        Instantiates and returns a RoadRunnerPy instance with the BioNetGen model
        :return: {RoadRunnerPy} RoadRunnerPy instance
        """
        rr = RoadRunnerPy(_modelString=self.sbml_model_string)
        rr.loadSBML(_modelString=self.sbml_model_string)
        return rr

    def setup_simulator(self, sim_type=__sim_type_roadrunner__) -> typing.Union[RoadRunnerPy]:
        """
        Instantiates and returns a supported simulator loaded with the BioNetGen model
        :param sim_type: {str} type of BioNetGen model simulator
        :return: {Union[RoadRunnerPy]} BioNetGen model simulator
        """
        if sim_type == __sim_type_roadrunner__:
            self.simulator = self._setup_roadrunner()
        return self.simulator


class BNGCC3D(object):
    def __init__(self, model_file: str = '', model_string: str = '', model_name: str = '', model: BNGModelCC3D = None,
                 sim_type: str = __sim_type_roadrunner__, step_size: float = 1.0, solver_opts: dict = None):
        """
        User-facing class for interacting with BioNetGen models in CC3D
        :param model_file: {str} path to BioNetGen model file
        :param model_string: {str} BioNetGen model string
        :param model_name: {str} BioNetGen model name
        :param model: {BNGModelCC3D} existing model to instantiate with
        :param sim_type: {str} type of BioNetGen model simulator
        :param step_size: {float} BioNetGen simulation step size
        :param solver_opts: {dict} BioNetGen solver options
        """
        if not bng_available:
            raise BNGUnavailableError("BioNetGen not installed.")

        if model is None:
            try:
                model = BNGModelCC3D(model_file=model_file, model_string=model_string, model_name=model_name)
            except TypeError:
                raise BNGParserError()
        self.model_name = model_name

        self.simulator = model.setup_simulator(sim_type=sim_type)
        if solver_opts is None:
            # todo - Integrate solver options per sim type
            solver_opts = dict()

        self.set_step_size(step_size)

    def set_step_size(self, _step_size: float) -> None:
        """
        Set the simulation step size
        :param _step_size: {float} simulation step size
        :return: None
        """
        self.simulator.setStepSize(_step_size)

    def timestep(self, _num_steps=1, _step_size=-1.0) -> None:
        """
        Integrates the simulation
        :param _num_steps: {int} number of simulation steps to integrate
        :param _step_size: {float} size of simulation step
        :return: None
        """
        self.simulator.timestep(_num_steps, _step_size)


def bng_roadrunner(_bngl: str) -> RoadRunnerPy:
    """
    Get a RoadRunner instance from a BioNetGen model
    :param _bngl: {str} BioNetGen lanuage model specification
    :return: {RoadRunnerPy} RoadRunner instance
    """
    return BNGCC3D(model_string=_bngl, sim_type=__sim_type_roadrunner__).simulator


def bngl_to_sbml(_bngl: str) -> str:
    """
    Translate BioNetGen language model to SBML
    :param _bngl: {str} model in BioNetGen language
    :return: {str} model in SBML
    """
    return BNGModelCC3D(_bngl).sbml_model_string


class BNGSolverHelper(object):
    __modelspec__ = ['add_bng_to_cell', 'add_free_floating_bng']
    # For storing generated models, which can be very expensive to generate
    __bng_models__ = dict()

    def __init__(self):
        """
        Steppable interface for BioNetGen model deployment in CC3D
        """
        if not bng_available:
            self._remove_model_specs()
            self.bng = None
            return

        self.bng = GlobalBNGFetcher()

    @classmethod
    def _remove_model_specs(cls) -> None:
        """
        Removes model specifications from interface
        :return: None
        """
        [delattr(cls, x) for x in cls.__modelspec__ if hasattr(cls, x)]

    @staticmethod
    def timestep_free_floating_bng() -> None:
        """
        Perform time step on all free-floating BioNetGen models
        :return: None
        """
        if not bng_available:
            return
        [bngcc3d.timestep() for bngcc3d in CompuCellSetup.persistent_globals.free_floating_bng_simulators.values()]

    @staticmethod
    def timestep_cell_bng(_cell) -> None:
        """
        Perform time step on all BioNetGen models attached to a cell
        :param _cell: {CellG} cell instance
        :return: None
        """
        [bngcc3d.timestep() for bngcc3d in _cell.dict[__bng_solvers__].values()]

    @classmethod
    def get_bng_model(cls, _model_name: str) -> typing.Union[BNGModelCC3D, None]:
        """
        Get a previously generated BioNetGen model by name; returns None if no model exists
        :param _model_name: {str} BioNetGen model name
        :return: {Union[BNGModelCC3D, None]} BNGModelCC3D instance if it exists, otherwise None
        """
        try:
            return cls.__bng_models__[_model_name]
        except KeyError:
            return None

    @classmethod
    def generate_bng_model(cls, model_name: str, model_file: str = '', model_string: str = '',
                           sim_type: str = __sim_type_roadrunner__, store: bool = False) -> BNGModelCC3D:
        """
        Generate a BioNetGen model
        :param model_name: {str} name of model
        :param model_file: {str} path to model file
        :param model_string: {str} model string
        :param sim_type: {str} BioNetGen simulation type
        :param store: {bool} store model for later usage; useful if the BioNetGen model requires a long time to generate
        :return: {BNGModelCC3D} BioNetGen model instance
        """
        bng_model = BNGModelCC3D(model_file=model_file, model_string=model_string, model_name=model_name)
        if store:
            if sim_type == __sim_type_roadrunner__:
                bng_model.sbml_model_string
            cls.__bng_models__[model_name] = bng_model
        return bng_model

    @classmethod
    def get_bngcc3d(cls, model_name: str, model_file: str = '', model_string: str = '',
                    step_size: float = 1.0, sim_type: str = __sim_type_roadrunner__,
                    solver_opts: dict = None) -> BNGCC3D:
        """
        Get a BNGCC3D instance by BioNetGen model name
        :param model_name: {str} name of BioNetGen model
        :param model_file: {str} path to model file; only needed if generating a new BioNetGen model from a file
        :param model_string: {str} model string; only needed if generating a new BioNetGen model from a string
        :param step_size: {float} BioNetGen simulation step size
        :param sim_type: {str} BioNetGen simulation type
        :param solver_opts: {dict} BioNetGen solver options
        :return: {BNGCC3D} instance of BNGCC3D for the BioNetGen model
        """
        bng_model = cls.get_bng_model(model_name)
        if bng_model is None:
            bng_model = cls.generate_bng_model(model_name=model_name,
                                               model_file=model_file,
                                               model_string=model_string,
                                               sim_type=sim_type,
                                               store=True)
        bng_model_copy = deepcopy(bng_model)
        return BNGCC3D(model=bng_model_copy, sim_type=sim_type, step_size=step_size, solver_opts=solver_opts)

    @classmethod
    def add_bng_to_cell(cls, _cell, model_name: str, model_file: str = '', model_string: str = '',
                        step_size: float = 1.0, sim_type: str = __sim_type_roadrunner__,
                        solver_opts: dict = None) -> None:
        """
        Attach a BioNetGen model simulator to a cell
        :param _cell: {CellG} a cell
        :param model_name: {str} BioNetGen model name
        :param model_file: {str} path to model file
        :param model_string: {str} model string
        :param step_size: {float} BioNetGen simulation step size
        :param sim_type: {str} BioNetGen simulation type
        :param solver_opts: {dict} BioNetGen solver options
        :return: None
        """
        dict_attrib = getPyAttrib(_cell)
        if __bng_solvers__ not in dict_attrib.keys():
            dict_attrib[__bng_solvers__] = dict()
        solver_dict = dict_attrib[__bng_solvers__]
        bngcc3d = cls.get_bngcc3d(model_name=model_name,
                                  model_file=model_file, model_string=model_string,
                                  step_size=step_size, sim_type=sim_type, solver_opts=solver_opts)
        solver_dict[model_name] = bngcc3d

    @classmethod
    def add_free_floating_bng(cls, model_name: str, model_file: str = '', model_string: str = '',
                              step_size: float = 1.0, sim_type: str = __sim_type_roadrunner__,
                              solver_opts: dict = None) -> None:
        """
        Attach a free-floating BioNetGen model simulator
        :param model_name: {str} BioNetGen model name
        :param model_file: {str} path to model file
        :param model_string: {str} model string
        :param step_size: {float} BioNetGen simulation step size
        :param sim_type: {str} BioNetGen simulation type
        :param solver_opts: {dict} BioNetGen solver options
        :return: None
        """
        bngcc3d = cls.get_bngcc3d(model_name=model_name,
                                  model_file=model_file, model_string=model_string,
                                  step_size=step_size, sim_type=sim_type, solver_opts=solver_opts)
        CompuCellSetup.persistent_globals.free_floating_bng_simulators[model_name] = bngcc3d

    @staticmethod
    def delete_bng_from_cell(_cell, _model_name: str) -> None:
        """
        Delete a BioNetGen model simulator from a cell
        :param _cell: {CellG} a cell
        :param _model_name: {str} BioNetGen model name
        :return: None
        """
        dict_attrib = getPyAttrib(_cell)
        try:
            del dict_attrib[__bng_solvers__][_model_name]
        except LookupError:
            pass

    @staticmethod
    def delete_free_floating_bng(_model_name: str) -> None:
        """
        Delete a free-floating BioNetGen model simulator
        :param _model_name: {str} BioNetGen model name
        :return: None
        """
        del CompuCellSetup.persistent_globals.free_floating_bng_simulators[_model_name]

    @staticmethod
    def copy_bng_simulators(from_cell, to_cell) -> None:
        """
        Copy BioNetGen model simulators from one cell to another
        :param from_cell: {CellG} cell from which all BioNetGen models should be copied
        :param to_cell: {CellG} cell to which all BioNetGen models should be copied
        :return: None
        """
        to_cell.dict[__bng_solvers__].update({k: deepcopy(v) for k, v in from_cell.dict[__bng_solvers__].items()})

    @staticmethod
    def available_bng_models(cell=None) -> typing.List[str]:
        if cell is not None:
            return list(cell.dict[__bng_solvers__].keys())
        return list(CompuCellSetup.persistent_globals.free_floating_bng_simulators.keys())
