from os.path import dirname, join
from pathlib import Path
from cc3d.cpp import CompuCell
from cc3d.core.XMLDomUtils import XMLIdLocator
from cc3d.CompuCellSetup import init_modules, parseXML
from cc3d.core.CMLFieldHandler import CMLFieldHandler
from cc3d.cpp import PlayerPython
import weakref

# import cc3d.CompuCellSetup as CompuCellSetup
# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup

# -------------------- legacy API emulation ----------------------------------------
def getCoreSimulationObjects():
    """

    :return:
    """
    return None, None

def initializeSimulationObjects(*args,**kwds):
    """

    :param args:
    :param kwds:
    :return:
    """

def mainLoop(*args, **kwds):
    """

    :param args:
    :param kwds:
    :return:
    """
    CompuCellSetup.run()


# -------------------- enf of legacy API emulation ----------------------------------------


def initialize_simulation_objects(sim, simthread):
    """

    :param sim:
    :param simthread:
    :return:
    """
    sim.extraInit()


def initialize_cc3d():
    """

    :return:
    """
    CompuCellSetup.persistent_globals.simulator, CompuCellSetup.persistent_globals.simthread = get_core_simulation_objects()

    simulator = CompuCellSetup.persistent_globals.simulator
    simthread = CompuCellSetup.persistent_globals.simthread

    # CompuCellSetup.persistent_globals.steppable_registry.simulator = simulator
    CompuCellSetup.persistent_globals.steppable_registry.simulator = weakref.ref(simulator)

    initialize_simulation_objects(simulator, simthread)

    CompuCellSetup.persistent_globals.simulation_initialized = True
    # print(' initialize cc3d CompuCellSetup.persistent_globals=',CompuCellSetup.persistent_globals)

def determine_main_loop_fcn():
    """
    Based on persisten globals this fcn determines which mainLoop function
    CC3D shold use
    :return: {object} function to use as a mainLoop
    """


    if CompuCellSetup.persistent_globals.simthread is None:
        return main_loop
    else:
        player_type = CompuCellSetup.persistent_globals.player_type
        if player_type == 'CMLResultReplay':
            # result replay
            return main_loop_player_cml_result_replay
        else:
            # "regular" run
            return main_loop_player

def run():
    """

    :return:
    """
    persistent_globals = CompuCellSetup.persistent_globals
    simulation_initialized = persistent_globals.simulation_initialized
    if not simulation_initialized:
        initialize_cc3d()
        # print(' run(): CompuCellSetup.persistent_globals=', CompuCellSetup.persistent_globals)
        # print(' run(): CompuCellSetup.persistent_globals.simulator=', CompuCellSetup.persistent_globals.simulator)
        persistent_globals.steppable_registry.core_init()

        #initializing extra visualization fields
        field_registry = persistent_globals.field_registry
        potts = persistent_globals.simulator.getPotts()
        cellField = potts.getCellFieldG()
        dim = cellField.getDim()
        field_registry.dim = dim
        field_registry.simthread= persistent_globals.simthread

        field_registry.create_fields()


    simulator = CompuCellSetup.persistent_globals.simulator
    simthread = CompuCellSetup.persistent_globals.simthread
    steppable_registry = CompuCellSetup.persistent_globals.steppable_registry

    main_loop_fcn = determine_main_loop_fcn()

    main_loop_fcn(simulator, simthread=simthread, steppableRegistry=steppable_registry)
    # mainLoop(simulator, simthread=simthread, steppableRegistry=steppable_registry)


def register_steppable(steppable):
    """

    :param steppable:{SteppableBasePy object}
    :return: None
    """
    steppable_registry = CompuCellSetup.persistent_globals.steppable_registry

    # # initializing steppables with the sim object
    # steppable.simulator = steppable_registry.simulator
    # steppable.core_init()

    steppable_registry.registerSteppable(_steppable=steppable)


def get_core_simulation_objects():
    # xml_fname = r'd:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_2D.xml'
    # cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    # CompuCell.initializePlugins()

    persistent_globals = CompuCellSetup.persistent_globals

    simulator = CompuCell.Simulator()
    simthread = None
    # todo 5 - fix logic regarding simthread initialization
    if persistent_globals.simthread is not None:
        simthread = persistent_globals.simthread

    simulator.setBasePath(join(dirname(persistent_globals.simulation_file_name)))

    print("Simulation basepath=",simulator.getBasePath())




    xml_fname = CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript


    cc3d_xml2_obj_converter = parseXML(xml_fname=xml_fname)
    #  cc3d_xml2_obj_converter cannot be garbage colected hence goes to persisten storage declared at the global level
    # in CompuCellSetup
    persistent_globals.cc3d_xml_2_obj_converter = cc3d_xml2_obj_converter

    # locating all XML elements with attribute id - presumably to be used for programmatic steering
    persistent_globals.xml_id_locator = XMLIdLocator(root_elem=persistent_globals.cc3d_xml_2_obj_converter.root)
    persistent_globals.xml_id_locator.locate_id_elements()


    # cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

    # print('CompuCellSetup.cc3dSimulationDataHandler=', CompuCellSetup.cc3dSimulationDataHandler)
    # print('cc3dSimulationDataHandler.cc3dSimulationData.pythonScript=',
    #       CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
    init_modules(simulator, cc3d_xml2_obj_converter)
    #
    # # sim.initializeCC3D()
    # # at this point after initialize cc3d stepwe can start querieg sim object.
    # # print('num_steps=', sim.getNumSteps())
    #
    # # sim.start()
    #
    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    CompuCell.initializePlugins()
    print("simulator=", simulator)
    simulator.initializeCC3D()
    # sim.extraInit()

    return simulator, simthread


def incorporate_script_steering_changes(simulator)->None:
    """
    Iterates over the list of modified xml elements and  schedules XML updates

    :return:None
    """

    persistent_globals = CompuCellSetup.persistent_globals
    xml_id_locator = persistent_globals.xml_id_locator

    # passing information about modules that need to be updated to C++ code
    # super_parent means XML element of the CC3D module  e.g. Plugin, Steppable or Potts
    for dirty_module_id, dirty_module_xml_elem in xml_id_locator.dirty_super_parents.items():
        simulator.updateCC3DModule(dirty_module_xml_elem)

    # resetting xml_id_locator.recently_accessed_elems
    xml_id_locator.reset()


def init_lattice_snapshot_objects():
    """
    Initializes cml filed handler , field storage and field extractor
    This trio of objects are responsible for outputting lattice snapshots
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals

    sim = persistent_globals.simulator
    dim = sim.getPotts().getCellFieldG().getDim()

    field_storage = PlayerPython.FieldStorage()
    field_extractor = PlayerPython.FieldExtractor()
    field_extractor.setFieldStorage(field_storage)

    persistent_globals.persistent_holder['field_storage'] = field_storage
    persistent_globals.persistent_holder['field_extractor'] = field_extractor

    field_storage.allocateCellField(dim)

    field_extractor.init(sim)

    cml_field_handler = CMLFieldHandler()
    persistent_globals.cml_field_handler = cml_field_handler

    cml_field_handler.initialize(field_storage=field_storage)


def store_lattice_snapshot(cur_step:int)->None:
    """
    Stores complete lattice snapshots
    :param cur_step:
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals
    output_frequency = persistent_globals.output_frequency

    cml_field_handler =persistent_globals.cml_field_handler

    if output_frequency and cml_field_handler and (not cur_step % output_frequency):
        cml_field_handler.write_fields(cur_step)


def init_screenshot_manager():
    """
    Initializes screenshot manager
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals

    screenshot_data_fname = str(Path(persistent_globals.simulation_file_name).joinpath(
        'screenshot_data/screenshots.json'))

    # screenshot_mgr = ScreenshotManagerCore()


def main_loop(sim, simthread, steppableRegistry):
    """

    :param sim:
    :param simthread:
    :param steppableRegistry:
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals

    steppableRegistry = CompuCellSetup.persistent_globals.steppable_registry
    if not steppableRegistry is None:
        steppableRegistry.init(sim)

    max_num_steps = sim.getNumSteps()

    init_lattice_snapshot_objects()


    sim.start()
    if not steppableRegistry is None:
        steppableRegistry.start()

    cur_step = 0

    while cur_step < max_num_steps :
        if CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            runFinishFlag = False
            break
        sim.step(cur_step)

        if not steppableRegistry is None:
            steppableRegistry.step(cur_step)

        store_lattice_snapshot(cur_step=cur_step)

        # passing Python-script-made changes in XML to C++ code
        incorporate_script_steering_changes(simulator=sim)

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()


        cur_step += 1


def extra_init_simulation_objects(sim, simthread, _restartEnabled=False):
    print("Simulation basepath extra init=", sim.getBasePath())

    sim.extraInit()  # after all xml steppables and plugins have been loaded we call extraInit to complete initialization
    # simthread.preStartInit()
    sim.start()

    # sends signal to player  to prepare for the upcoming simulation
    simthread.postStartInit()

    # waits for player  to complete initialization
    simthread.waitForPlayerTaskToFinish()




def main_loop_player(sim, simthread, steppableRegistry):
    """

    :param sim:
    :param simthread:
    :param steppableRegistry:
    :return:
    """
    steppableRegistry = CompuCellSetup.persistent_globals.steppable_registry
    simthread = CompuCellSetup.persistent_globals.simthread


    extra_init_simulation_objects(sim, simthread, _restartEnabled=False)

    # simthread.waitForInitCompletion()
    # simthread.waitForPlayerTaskToFinish()


    if not steppableRegistry is None:
        steppableRegistry.init(sim)

    max_num_steps = sim.getNumSteps()

    # called in extraInitSimulationObjects
    # sim.start()

    if not steppableRegistry is None:
        steppableRegistry.start()

    runFinishFlag = True

    cur_step = 0
    while cur_step < max_num_steps:
        simthread.beforeStep(_mcs=cur_step)
        if simthread.getStopSimulation() or CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            runFinishFlag = False
            break

        sim.step(cur_step)

        # steering using GUI. GUI steering overrides steering done in the steppables
        simthread.steerUsingGUI(sim)

        if not steppableRegistry is None:
            steppableRegistry.step(cur_step)


        # passing Python-script-made changes in XML to C++ code
        incorporate_script_steering_changes(simulator=sim)

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()

        simthread.loopWork(cur_step)
        simthread.loopWorkPostEvent(cur_step)

        cur_step += 1

    if runFinishFlag:
        # # we emit request to finish simulation
        # simthread.emitFinishRequest()
        # # then we wait for GUI thread to unlock the finishMutex - it will only happen when all tasks in the GUI thread are completed (especially those that need simulator object to stay alive)
        # simthread.finishMutex.lock()
        # simthread.finishMutex.unlock()
        # # at this point GUI thread finished all the tasks for which simulator had to stay alive  and we can proceed to destroy simulator
        #
        # sim.finish()
        # if sim.getRecentErrorMessage() != "":
        #     raise CC3DCPlusPlusError(sim.getRecentErrorMessage())
        # steppableRegistry.finish()
        sim.cleanAfterSimulation()
        simthread.simulationFinishedPostEvent(True)
        steppableRegistry.clean_after_simulation()
        print ("CALLING FINISH")
    else:
        sim.cleanAfterSimulation()

        # # sim.unloadModules()
        print( "CALLING UNLOAD MODULES NEW PLAYER")
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            simthread.simulationFinishedPostEvent(True)

        steppableRegistry.clean_after_simulation()


def main_loop_player_cml_result_replay(sim, simthread, steppableRegistry):
    """

    :param sim:
    :param simthread:
    :param steppableRegistry:
    :return:
    """