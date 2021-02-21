import time
from math import floor
from os.path import dirname, join
from cc3d.cpp import CompuCell

from cc3d.core.XMLDomUtils import XMLIdLocator
from cc3d.CompuCellSetup import init_modules, parseXML
from cc3d.core.CMLFieldHandler import CMLFieldHandler
from cc3d.core.GraphicsUtils.ScreenshotManagerCore import ScreenshotManagerCore
from cc3d.cpp import PlayerPython
from cc3d.core.GraphicsOffScreen import GenericDrawer
from cc3d.core.BasicSimulationData import BasicSimulationData
import warnings
import time
import weakref
from cc3d import CompuCellSetup
from cc3d.core import RestartManager
from cc3d.CompuCellSetup.simulation_utils import check_for_cpp_errors
from cc3d.core.Validation.sanity_checkers import validate_cc3d_entity_identifier
from cc3d.CompuCellSetup.cluster_utils import check_nanohub_and_count


# -------------------- legacy API emulation ----------------------------------------
def getCoreSimulationObjects():
    """

    :return:
    """
    return None, None


def initializeSimulationObjects(*args, **kwds):
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

def initialize_cc3d():
    """

    :return:
    """
    CompuCellSetup.persistent_globals.simulator, \
    CompuCellSetup.persistent_globals.simthread = get_core_simulation_objects()

    check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

    simulator = CompuCellSetup.persistent_globals.simulator

    # CompuCellSetup.persistent_globals.steppable_registry.simulator = simulator
    CompuCellSetup.persistent_globals.steppable_registry.simulator = weakref.ref(simulator)

    CompuCellSetup.persistent_globals.simulation_initialized = True


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
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        persistent_globals.steppable_registry.core_init()
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        # initializing extra visualization fields
        field_registry = persistent_globals.field_registry
        potts = persistent_globals.simulator.getPotts()
        cell_field = potts.getCellFieldG()
        dim = cell_field.getDim()
        field_registry.dim = dim
        field_registry.simthread = persistent_globals.simthread

        field_registry.create_fields()

    simulator = CompuCellSetup.persistent_globals.simulator
    simthread = CompuCellSetup.persistent_globals.simthread
    steppable_registry = CompuCellSetup.persistent_globals.steppable_registry

    main_loop_fcn = determine_main_loop_fcn()

    main_loop_fcn(simulator, simthread=simthread, steppable_registry=steppable_registry)


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


def print_profiling_report(py_steppable_profiler_report, compiled_code_run_time, total_run_time):
    """
    prints profiling information after simulation finishes running
    :param py_steppable_profiler_report:
    :param compiled_code_run_time:
    :param total_run_time:
    :return:
    """
    profiling_format = '{:>32.32}: {:11.2f} ({:5.1%})'

    print('\n\n')
    print('------------------PERFORMANCE REPORT:----------------------')
    print('-----------------------------------------------------------')
    print("TOTAL RUNTIME ", convert_time_interval_to_hmsm(total_run_time))
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print('PYTHON STEPPABLE RUNTIMES')
    # print py_steppable_profiler_report

    totStepTime = 0

    for steppableName, steppableObjectHash, run_time_ms in py_steppable_profiler_report:
        print(profiling_format.format(steppableName, int(run_time_ms) / 1000., int(run_time_ms) / total_run_time))

        totStepTime += run_time_ms

    print('-----------------------------------------------------------')

    print(profiling_format.format('Total Steppable Time', int(totStepTime) / 1000., int(totStepTime) / total_run_time))

    print(profiling_format.format('Compiled Code (C++) Run Time', int(compiled_code_run_time) / 1000.,
                                  int(compiled_code_run_time) / total_run_time))

    print(profiling_format.format('Other Time', int(total_run_time - compiled_code_run_time - totStepTime) / 1000.,
                                  int(total_run_time - compiled_code_run_time - totStepTime) / total_run_time))

    print('-----------------------------------------------------------')
    print()


def convert_time_interval_to_hmsm(time_interval):
    """
    Converts timestamp to human readable format
    :param time_interval:
    :return:
    """
    time_interval = int(time_interval)
    hours = time_interval / (3600 * 1000)
    minutes_interval = time_interval % (3600 * 1000)
    minutes = minutes_interval / (60 * 1000)
    seconds_interval = minutes_interval % (60 * 1000)
    seconds = seconds_interval / 1000
    miliseconds = seconds_interval % 1000

    def s_int_fl(x):
        return str(int(floor(x)))

    if hours > 1.0:
        out_str = s_int_fl(hours) + " h : " + s_int_fl(minutes) + " m : " + s_int_fl(seconds) + " s : " + str(
            miliseconds) + " ms"

    elif minutes > 1.0:
        out_str = s_int_fl(minutes) + " m : " + s_int_fl(seconds) + " s : " + str(miliseconds) + " ms"

    elif seconds > 1.0:
        out_str = s_int_fl(seconds) + " s : " + str(miliseconds) + " ms"

    else:
        out_str = str(miliseconds) + " ms"

    return out_str + ' = ' + str(time_interval / 1000.0) + ' s'


def get_core_simulation_objects():
    persistent_globals = CompuCellSetup.persistent_globals

    simulator = CompuCell.Simulator()
    simthread = None
    # todo 5 - fix logic regarding simthread initialization
    if persistent_globals.simthread is not None:
        simthread = persistent_globals.simthread
        simulator.setNewPlayerFlag(True)

    simulator.setBasePath(join(dirname(persistent_globals.simulation_file_name)))

    xml_fname = CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript

    if persistent_globals.cc3d_xml_2_obj_converter is None:
        # We only call parseXML if previous steps do not initialize XML tree
        # Such situation usually happen when we specify XML tree using Python API in configure_simulation function
        # that is typically called from the Python main script

        cc3d_xml2_obj_converter = parseXML(xml_fname=xml_fname)
        #  cc3d_xml2_obj_converter cannot be garbage colected hence goes to persisten storage
        #  declared at the global level in CompuCellSetup
        persistent_globals.cc3d_xml_2_obj_converter = cc3d_xml2_obj_converter

    # locating all XML elements with attribute id - presumably to be used for programmatic steering
    persistent_globals.xml_id_locator = XMLIdLocator(root_elem=persistent_globals.cc3d_xml_2_obj_converter.root)
    persistent_globals.xml_id_locator.locate_id_elements()

    init_modules(simulator, persistent_globals.cc3d_xml_2_obj_converter)

    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    CompuCell.initializePlugins()

    simulator.initializeCC3D()

    # sim.extraInit()

    return simulator, simthread


def incorporate_script_steering_changes(simulator) -> None:
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


def initialize_field_extractor_objects():
    """
    Initialzies field storage and field extractor objects
    Stores references to them in persistent_globals.persistent_holder dictionary
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


def init_lattice_snapshot_objects():
    """
    Initializes cml filed handler , field storage and field extractor
    This trio of objects are responsible for outputting lattice snapshots
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals

    initialize_field_extractor_objects()
    field_storage = persistent_globals.persistent_holder['field_storage']

    cml_field_handler = CMLFieldHandler()
    persistent_globals.cml_field_handler = cml_field_handler

    cml_field_handler.initialize(field_storage=field_storage)


def store_lattice_snapshot(cur_step: int) -> None:
    """
    Stores complete lattice snapshots
    :param cur_step:
    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals
    output_frequency = persistent_globals.output_frequency

    cml_field_handler = persistent_globals.cml_field_handler

    if output_frequency and cml_field_handler and (not cur_step % output_frequency):
        cml_field_handler.write_fields(cur_step)


def init_screenshot_manager() -> None:
    """
    Initializes screenshot manager. Requires that field extractor is set.

    :return:
    """

    persistent_globals = CompuCellSetup.persistent_globals

    screenshot_data_fname = join(dirname(persistent_globals.simulation_file_name), 'screenshot_data/screenshots.json')

    persistent_globals.screenshot_manager = ScreenshotManagerCore()

    try:
        field_extractor = persistent_globals.persistent_holder['field_extractor']
    except KeyError:
        initialize_field_extractor_objects()
        field_extractor = persistent_globals.persistent_holder['field_extractor']

    persistent_globals.create_output_dir()
    gd = GenericDrawer.GenericDrawer(boundary_strategy=persistent_globals.simulator.getBoundaryStrategy())
    gd.set_field_extractor(field_extractor=field_extractor)

    bsd = BasicSimulationData()
    bsd.fieldDim = persistent_globals.simulator.getPotts().getCellFieldG().getDim()
    bsd.numberOfSteps = persistent_globals.simulator.getNumSteps()

    # wiring screenshot manager
    persistent_globals.screenshot_manager.gd = gd
    persistent_globals.screenshot_manager.bsd = bsd
    persistent_globals.screenshot_manager.screenshot_number_of_digits = len(str(bsd.numberOfSteps))

    try:
        persistent_globals.screenshot_manager.read_screenshot_description_file(screenshot_data_fname)
    except:
        warnings.warn('Could not parse screenshot description file {screenshot_data_fname}. '
                      'If you want graphical screenshot output please generate '
                      'new screenshot description file from Player by pressing camera button on '
                      'select graphical visualizations. If you do not want screenshots'
                      'you may delete subfolder {scr_dir} or simply ingnore this message'.format(
            screenshot_data_fname=screenshot_data_fname, scr_dir=dirname(screenshot_data_fname)))
        time.sleep(5)


def store_screenshots(cur_step: int) -> None:
    """
    Stores screenshots
    :param cur_step:{int} current MCS
    :return: None
    """

    persistent_globals = CompuCellSetup.persistent_globals

    screenshot_output_frequency = persistent_globals.screenshot_output_frequency
    screenshot_manager = persistent_globals.screenshot_manager

    if screenshot_manager.has_ad_hoc_screenshots():
        screenshot_manager.output_screenshots(mcs=cur_step)

    if screenshot_output_frequency and screenshot_manager and (not cur_step % screenshot_output_frequency):
        screenshot_manager.output_screenshots(mcs=cur_step)


def extra_init_simulation_objects(sim, simthread, init_using_restart_snapshot_enabled=False):
    """
    Performs extra initializations. Also checks for validity of concentration field names (those defined in C++)
    :param sim:
    :param simthread:
    :param init_using_restart_snapshot_enabled:
    :return:
    """

    # after all xml steppables and plugins have been loaded we call extraInit to complete initialization
    sim.extraInit()
    concentration_field_names = CompuCell.getConcentrationFieldNames(sim)
    for conc_field_name in concentration_field_names:
        validate_cc3d_entity_identifier(entity_identifier=conc_field_name,
                                        entity_type_label='Concentration Field Label')

    # passing output directory to simulator object
    sim.setOutputDirectory(CompuCellSetup.persistent_globals.output_directory)
    # simthread.preStartInit()
    # we skip calling start functions of steppables if restart is enabled and we are using restart
    # directory to restart simulation from a given MCS
    if not init_using_restart_snapshot_enabled:
        sim.start()

    if simthread is not None:
        # sends signal to player  to prepare for the upcoming simulation
        simthread.postStartInit()

        # waits for player  to complete initialization
        simthread.waitForPlayerTaskToFinish()


def main_loop(sim, simthread, steppable_registry=None):
    """
    main loop for CML simulation
    :param sim:
    :param simthread:
    :param steppable_registry:
    :return:
    """
    t1 = time.time()
    compiled_code_run_time = 0.0

    pg = CompuCellSetup.persistent_globals
    steppable_registry = pg.steppable_registry

    pg.restart_manager = RestartManager.RestartManager(sim)
    restart_manager = pg.restart_manager
    restart_manager.output_frequency = pg.restart_snapshot_frequency
    restart_manager.allow_multiple_restart_directories = pg.restart_multiple_snapshots

    init_using_restart_snapshot_enabled = restart_manager.restart_enabled()
    # init_using_restart_snapshot_enabled = False
    sim.setRestartEnabled(init_using_restart_snapshot_enabled)
    check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

    check_nanohub_and_count()

    if init_using_restart_snapshot_enabled:
        print('WILL RESTART SIMULATION')
        restart_manager.loadRestartFiles()
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)
    else:
        print('WILL RUN SIMULATION FROM BEGINNING')

    extra_init_simulation_objects(sim, simthread,
                                  init_using_restart_snapshot_enabled=init_using_restart_snapshot_enabled)
    check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

    if steppable_registry is not None:
        steppable_registry.init(sim)

    init_lattice_snapshot_objects()
    init_screenshot_manager()

    if steppable_registry is not None and not init_using_restart_snapshot_enabled:
        steppable_registry.start()

    run_finish_flag = True

    restart_manager.prepare_restarter()
    beginning_step = restart_manager.get_restart_step()

    cur_step = beginning_step

    while cur_step < sim.getNumSteps():
        if CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            run_finish_flag = False
            break

        if steppable_registry is not None:
            steppable_registry.stepRunBeforeMCSSteppables(cur_step)

        compiled_code_begin = time.time()

        sim.step(cur_step)  # steering using steppables
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        compiled_code_end = time.time()

        compiled_code_run_time += (compiled_code_end - compiled_code_begin) * 1000

        if steppable_registry is not None:
            steppable_registry.step(cur_step)

        # restart manager will decide whether to output files or not based on its settings
        restart_manager.output_restart_files(cur_step)

        store_lattice_snapshot(cur_step=cur_step)
        store_screenshots(cur_step=cur_step)

        # passing Python-script-made changes in XML to C++ code
        incorporate_script_steering_changes(simulator=sim)

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        cur_step += 1

    if run_finish_flag:
        print("CALLING FINISH")
        steppable_registry.finish()
    else:
        steppable_registry.on_stop()

    t2 = time.time()
    print_profiling_report(py_steppable_profiler_report=steppable_registry.get_profiler_report(),
                           compiled_code_run_time=compiled_code_run_time, total_run_time=(t2 - t1) * 1000.0)


def main_loop_player(sim, simthread=None, steppable_registry=None):
    """
    main loop for GUI based simulations
    :param sim:
    :param simthread:
    :param steppable_registry:
    :return:
    """
    t1 = time.time()
    compiled_code_run_time = 0.0

    pg = CompuCellSetup.persistent_globals

    steppable_registry = pg.steppable_registry
    simthread = pg.simthread

    pg.restart_manager = RestartManager.RestartManager(sim)
    restart_manager = pg.restart_manager
    restart_manager.output_frequency = pg.restart_snapshot_frequency
    restart_manager.allow_multiple_restart_directories = pg.restart_multiple_snapshots

    init_using_restart_snapshot_enabled = restart_manager.restart_enabled()
    # init_using_restart_snapshot_enabled = False
    sim.setRestartEnabled(init_using_restart_snapshot_enabled)

    check_nanohub_and_count()

    if init_using_restart_snapshot_enabled:
        print('WILL RESTART SIMULATION')
        restart_manager.loadRestartFiles()
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)
    else:
        print('WILL RUN SIMULATION FROM BEGINNING')

    extra_init_simulation_objects(sim, simthread,
                                  init_using_restart_snapshot_enabled=init_using_restart_snapshot_enabled)

    check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

    # simthread.waitForInitCompletion()
    # simthread.waitForPlayerTaskToFinish()

    if not steppable_registry is None:
        steppable_registry.init(sim)

    # called in extraInitSimulationObjects
    # sim.start()

    if not steppable_registry is None and not init_using_restart_snapshot_enabled:
        steppable_registry.start()
        simthread.steppablePostStartPrep()

    run_finish_flag = True

    restart_manager.prepare_restarter()
    beginning_step = restart_manager.get_restart_step()

    if init_using_restart_snapshot_enabled:
        steppable_registry.restart_steering_panel()

    cur_step = beginning_step

    # initial drawing - we use mcs = -2 as a sentinel for mcs right after start function of steppables is completed
    simthread.loopWork(-2)
    simthread.loopWorkPostEvent(-2)

    while cur_step < sim.getNumSteps():
        simthread.beforeStep(_mcs=cur_step)
        if simthread.getStopSimulation() or CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            run_finish_flag = False
            break

        if steppable_registry is not None:
            steppable_registry.stepRunBeforeMCSSteppables(cur_step)

        compiled_code_begin = time.time()

        sim.step(cur_step)  # steering using steppables
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        compiled_code_end = time.time()

        compiled_code_run_time += (compiled_code_end - compiled_code_begin) * 1000

        # steering using GUI. GUI steering overrides steering done in the steppables
        simthread.steerUsingGUI(sim)

        if not steppable_registry is None:
            steppable_registry.step(cur_step)

        # restart manager will decide whether to output files or not based on its settings
        restart_manager.output_restart_files(cur_step)

        # passing Python-script-made changes in XML to C++ code
        incorporate_script_steering_changes(simulator=sim)

        # steer application will only update modules that uses requested using updateCC3DModule function from simulator
        sim.steer()
        check_for_cpp_errors(CompuCellSetup.persistent_globals.simulator)

        screen_update_frequency = simthread.getScreenUpdateFrequency()
        screenshot_frequency = simthread.getScreenshotFrequency()
        screenshot_output_flag = simthread.getImageOutputFlag()

        if pg.screenshot_manager is not None and pg.screenshot_manager.has_ad_hoc_screenshots():
            simthread.loopWork(cur_step)
            simthread.loopWorkPostEvent(cur_step)

        elif (screen_update_frequency > 0 and cur_step % screen_update_frequency == 0) or (
                screenshot_output_flag and screenshot_frequency > 0 and cur_step % screenshot_frequency == 0):

            simthread.loopWork(cur_step)
            simthread.loopWorkPostEvent(cur_step)

        cur_step += 1

    if run_finish_flag:
        # # we emit request to finish simulation
        simthread.emitFinishRequest()
        # # then we wait for GUI thread to unlock the finishMutex - it will only happen when all tasks
        # in the GUI thread are completed (especially those that need simulator object to stay alive)
        print("CALLING FINISH")

        simthread.waitForFinishingTasksToConclude()
        simthread.waitForPlayerTaskToFinish()
        steppable_registry.finish()
        sim.cleanAfterSimulation()
        simthread.simulationFinishedPostEvent(True)
        steppable_registry.clean_after_simulation()

    else:
        steppable_registry.on_stop()
        sim.cleanAfterSimulation()

        # # sim.unloadModules()
        print("CALLING UNLOAD MODULES NEW PLAYER")
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            simthread.simulationFinishedPostEvent(True)

        steppable_registry.clean_after_simulation()

    t2 = time.time()
    print_profiling_report(py_steppable_profiler_report=steppable_registry.get_profiler_report(),
                           compiled_code_run_time=compiled_code_run_time, total_run_time=(t2 - t1) * 1000.0)


def main_loop_player_cml_result_replay(sim, simthread, steppableRegistry):
    """

    :param sim:
    :param simthread:
    :param steppableRegistry:
    :return:
    """
