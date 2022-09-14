"""
MaBoSS Python integration
=========================
Written by T.J. Sego, Ph.D.

Biocomplexity Institute

Indiana University

Bloomington, IN, U.S.A.

Overview
========
Defines helpers and steppable class features for using MaBoSS in CC3D Python

"""
import os
import tempfile
from typing import Dict, Optional
from cc3d.cpp import CompuCell
from cc3d.core.CellInventoryWatcher import CellInventoryWatcher
from cc3d import CompuCellSetup
try:
    from cc3dext import MaBoSSCC3DPy
    maboss_engine_type = MaBoSSCC3DPy.CC3DMaBoSSEngine
    maboss_engine_container_type = MaBoSSCC3DPy.CC3DMaBoSSEngineContainer
    __has_extension__ = True
except ModuleNotFoundError:
    __has_extension__ = False
    maboss_engine_type = object
    maboss_engine_container_type = object


class NoMaBoSSExtensionError(Exception):
    """Simple exception to notify of missing required extension"""
    pass


class MaBoSSInventoryWatcher(CellInventoryWatcher):
    """
    Inventory watcher that maintains synchronization between the cell and engine inventories
    """

    def on_cell_remove(self, cell: CompuCell.CellG):
        if CompuCellSetup.persistent_globals.maboss_simulators is not None:
            CompuCellSetup.persistent_globals.maboss_simulators.remCell(cell.id)


maboss_inventory_watcher: Optional[MaBoSSInventoryWatcher] = None


def maboss_model(bnd_file: str = None,
                 bnd_str: str = None,
                 cfg_file: str = None,
                 cfg_str: str = None,
                 time_step: float = 1.0,
                 time_tick: float = 1.0,
                 discrete_time: bool = False,
                 seed: int = None,
                 istate: Dict[str, bool] = None) -> maboss_engine_type:
    """
    Instantiate a MaBoSS simulation instance from files and/or strings.

    :param bnd_file: path to a network file (required unless bnd_str is specified)
    :type bnd_file: str
    :param bnd_str: network multiline string (required unless bnd_file is specified)
    :type bnd_str: str
    :param cfg_file: path to a configuration file (required unless cfg_str is specified)
    :type cfg_file: str
    :param cfg_str: configuration multiline string (required unless cfg_file is specified)
    :type cfg_str: str
    :param time_step: period of one simulation step (default 1.0)
    :type time_step: float
    :param time_tick: period of one evaluation (default 1.0)
    :type time_tick: float
    :param discrete_time: flag to use discrete time (default False)
    :type discrete_time: bool
    :param seed: random generator seed (default 0)
    :type seed: int
    :param istate: initial state, by name and state value
    :return: MaBoSS simulation instance
    :rtype: MaBoSSCC3DPy.CC3DMaBoSSEngine
    """
    if not __has_extension__:
        raise NoMaBoSSExtensionError

    if bnd_file is None:
        if bnd_str is None:
            raise ValueError("No network specified")
        bnd_file_obj, bnd_file = tempfile.mkstemp(text=True, suffix='.bnd')
        with os.fdopen(bnd_file_obj, 'w') as f:
            f.write(bnd_str)

    if cfg_file is None:
        if cfg_str is None:
            raise ValueError("No configuration specified")
        cfg_file_obj, cfg_file = tempfile.mkstemp(text=True, suffix='.cfg')
        with os.fdopen(cfg_file_obj, 'w') as f:
            f.write(cfg_str)

    kwargs = {'ctbndl_file': bnd_file,
              'cfg_file': cfg_file,
              'stepSize': time_step}
    if seed is not None:
        kwargs['seed'] = seed
        kwargs['cfgSeed'] = False
    engine = MaBoSSCC3DPy.CC3DMaBoSSEngine(**kwargs)
    engine.run_config.setDiscreteTime(discrete_time=discrete_time)
    engine.run_config.setTimeTick(time_tick=time_tick)
    if istate is not None:
        for k, v in istate:
            engine[k].state = v
            engine[k].istate = v

    if bnd_str is not None:
        os.remove(path=bnd_file)
    if cfg_str is not None:
        os.remove(path=cfg_file)

    return engine


def maboss_container_stats(maboss_container: maboss_engine_container_type,
                           model_name: str,
                           node_name: str) -> dict:
    """
    Returns a dictionary with summary statistics of a node of a model in an engine container, including

    - ``num_true``: the total number of cells with the model and a true node state
    - ``num_false``: the total number of cells with the model and a false node state

    :param maboss_container: engine container
    :param model_name: name of model
    :param node_name: name of node
    :return: dictionary of statistics
    """
    if not __has_extension__:
        raise NoMaBoSSExtensionError

    engines = maboss_container.getEnginesByName(model_name)
    num_true = 0
    num_false = 0
    for e in engines.values():
        if e[node_name].state:
            num_true += 1
        else:
            num_false += 1

    return {'num_true': num_true,
            'num_false': num_false}


class MaBoSSHelper:
    """
    Interface for using MaBoSS in a cc3d steppable.

    To be used with :class:`cc3d.core.PySteppables.SteppableBasePy`.
    """
    @staticmethod
    def add_maboss_to_cell(cell: CompuCell.CellG,
                           model_name: str,
                           bnd_file: str = None,
                           bnd_str: str = None,
                           cfg_file: str = None,
                           cfg_str: str = None,
                           time_step: float = 1.0,
                           time_tick: float = 1.0,
                           discrete_time: bool = False,
                           seed: int = 0,
                           istate: Dict[str, bool] = None) -> None:
        """
        Add a MaBoSS simulation instance to a cell

        :param cell: a cell
        :type cell: CompuCell.CellG
        :param model_name: name of the model
        :type model_name: str
        :param bnd_file: path to a network file (required unless bnd_str is specified)
        :type bnd_file: str
        :param bnd_str: network multiline string (required unless bnd_file is specified)
        :type bnd_str: str
        :param cfg_file: path to a configuration file (required unless cfg_str is specified)
        :type cfg_file: str
        :param cfg_str: configuration multiline string (required unless cfg_file is specified)
        :type cfg_str: str
        :param time_step: period of one simulation step (default 1.0)
        :type time_step: float
        :param time_tick: period of one evaluation (default 1.0)
        :type time_tick: float
        :param discrete_time: flag to use discrete time (default False)
        :type discrete_time: bool
        :param seed: random generator seed (default 0)
        :type seed: int
        :param istate: initial state, by name and state value
        :return: None
        """
        engine = maboss_model(bnd_file=bnd_file,
                              bnd_str=bnd_str,
                              cfg_file=cfg_file,
                              cfg_str=cfg_str,
                              time_step=time_step,
                              time_tick=time_tick,
                              discrete_time=discrete_time,
                              seed=seed,
                              istate=istate)
        if CompuCell.CellG.__maboss__ not in cell.dict.keys():
            cell.dict[CompuCell.CellG.__maboss__] = {}

        if CompuCellSetup.persistent_globals.maboss_simulators is None:
            CompuCellSetup.persistent_globals.maboss_simulators = MaBoSSCC3DPy.CC3DMaBoSSEngineContainer()
            global maboss_inventory_watcher
            potts: CompuCell.Potts3D = CompuCellSetup.persistent_globals.simulator.getPotts()
            cell_inventory: CompuCell.CellInventory = potts.getCellInventory()
            maboss_inventory_watcher = MaBoSSInventoryWatcher(cell_inventory)
        CompuCellSetup.persistent_globals.maboss_simulators.addEngine(engine, cell.id, model_name)

        cell.dict[CompuCell.CellG.__maboss__][model_name] = engine

    @staticmethod
    def delete_maboss_from_cell(cell: CompuCell.CellG, model_name: str):
        """
        Remove a MaBoSS simulation instance from a cell

        :param cell: a cell
        :type cell: CompuCell.CellG
        :param model_name: name of the model
        :type model_name: str
        :return: None
        """
        del cell.dict[CompuCell.CellG.__maboss__][model_name]

        if CompuCellSetup.persistent_globals.maboss_simulators is not None:
            CompuCellSetup.persistent_globals.maboss_simulators.remEngine(cell.id, model_name)

    @staticmethod
    def maboss_model(bnd_file: str = None,
                     bnd_str: str = None,
                     cfg_file: str = None,
                     cfg_str: str = None,
                     time_step: float = 1.0,
                     time_tick: float = 1.0,
                     discrete_time: bool = False,
                     seed: int = 0,
                     istate: Dict[str, bool] = None) -> maboss_engine_type:
        """
        Instantiate a MaBoSS simulation instance from files and/or strings.

        :param bnd_file: path to a network file (required unless bnd_str is specified)
        :type bnd_file: str
        :param bnd_str: network multiline string (required unless bnd_file is specified)
        :type bnd_str: str
        :param cfg_file: path to a configuration file (required unless cfg_str is specified)
        :type cfg_file: str
        :param cfg_str: configuration multiline string (required unless cfg_file is specified)
        :type cfg_str: str
        :param time_step: period of one simulation step (default 1.0)
        :type time_step: float
        :param time_tick: period of one evaluation (default 1.0)
        :type time_tick: float
        :param discrete_time: flag to use discrete time (default False)
        :type discrete_time: bool
        :param seed: random generator seed (default 0)
        :type seed: int
        :param istate: initial state, by name and state value
        :return: MaBoSS simulation instance
        :rtype: MaBoSSCC3DPy.CC3DMaBoSSEngine
        """
        return maboss_model(bnd_file=bnd_file,
                            bnd_str=bnd_str,
                            cfg_file=cfg_file,
                            cfg_str=cfg_str,
                            time_step=time_step,
                            time_tick=time_tick,
                            discrete_time=discrete_time,
                            seed=seed,
                            istate=istate)

    def timestep_maboss(self) -> None:
        """
        Step all existing MaBoSS simulations

        :return: None
        """
        if CompuCellSetup.persistent_globals.maboss_simulators is not None:
            CompuCellSetup.persistent_globals.maboss_simulators.step()

    def maboss_stats(self, model_name: str, node_name: str) -> dict:
        """
        Returns a dictionary with summary statistics of a node of a model, including

        - ``num_true``: the total number of cells with the model and a true node state
        - ``num_false``: the total number of cells with the model and a false node state

        :param model_name: name of model
        :param node_name: name of node
        :return: dictionary of statistics
        """
        if CompuCellSetup.persistent_globals.maboss_simulators:
            return maboss_container_stats(CompuCellSetup.persistent_globals.maboss_simulators, model_name, node_name)
        return {}


if not __has_extension__:
    MaBoSSHelper = object
