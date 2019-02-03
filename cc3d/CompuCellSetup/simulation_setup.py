import CompuCell
from cc3d.CompuCellSetup import init_modules, parseXML
# import cc3d.CompuCellSetup as CompuCellSetup
# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup

def initializeSimulationObjects(sim, simthread):
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
    CompuCellSetup.persistent_globals.simulator, CompuCellSetup.persistent_globals.simthread = getCoreSimulationObjects()
    simulator = CompuCellSetup.persistent_globals.simulator
    simthread = CompuCellSetup.persistent_globals.simthread

    CompuCellSetup.persistent_globals.steppable_registry.simulator = simulator

    initializeSimulationObjects(simulator, simthread)

    CompuCellSetup.persistent_globals.simulation_initialized = True
    # print(' initialize cc3d CompuCellSetup.persistent_globals=',CompuCellSetup.persistent_globals)

def determine_main_loop_fcn():
    """
    Based on persisten globals this fcn determines which mainLoop function
    CC3D shold use
    :return: {object} function to use as a mainLoop
    """
    if CompuCellSetup.persistent_globals.simthread is None:
        return mainLoop
    else:
        return mainLoopPlayer

def run():
    """

    :return:
    """
    simulation_initialized = CompuCellSetup.persistent_globals.simulation_initialized
    if not simulation_initialized:
        initialize_cc3d()
        # print(' run(): CompuCellSetup.persistent_globals=', CompuCellSetup.persistent_globals)
        # print(' run(): CompuCellSetup.persistent_globals.simulator=', CompuCellSetup.persistent_globals.simulator)
        CompuCellSetup.persistent_globals.steppable_registry.core_init()


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


def getCoreSimulationObjects():
    # xml_fname = r'd:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_2D.xml'
    # cc3dXML2ObjConverter = parseXML(xml_fname=xml_fname)

    # # this loads all plugins/steppables - need to recode it to make loading on-demand only
    # CompuCell.initializePlugins()

    simulator = CompuCell.Simulator()
    simthread = None
    # todo 5 - fix logic regarding simthread initialization
    if CompuCellSetup.persistent_globals.simthread is not None:
        simthread = CompuCellSetup.persistent_globals.simthread


    xml_fname = CompuCellSetup.cc3dSimulationDataHandler.cc3dSimulationData.xmlScript

    cc3d_xml2_obj_converter = parseXML(xml_fname=xml_fname)
    #  cc3d_xml2_obj_converter cannot be garbage colected hence goes to persisten storage declared at the global level
    # in CompuCellSetup
    CompuCellSetup.persistent_globals.cc3d_xml_2_obj_converter = cc3d_xml2_obj_converter

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




def mainLoop(sim, simthread, steppableRegistry):
    steppableRegistry = CompuCellSetup.persistent_globals.steppable_registry
    if not steppableRegistry is None:
        steppableRegistry.init(sim)

    max_num_steps = sim.getNumSteps()
    sim.start()
    if not steppableRegistry is None:
        steppableRegistry.start()

    cur_step = 0
    while cur_step < max_num_steps / 10:
        if CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            runFinishFlag = False;
            break
        sim.step(cur_step)
        if not steppableRegistry is None:
            steppableRegistry.step(cur_step)

        cur_step += 1


def mainLoopPlayer(sim, simthread, steppableRegistry):
    """

    :param sim:
    :param simthread:
    :param steppableRegistry:
    :return:
    """
    steppableRegistry = CompuCellSetup.persistent_globals.steppable_registry
    simthread = CompuCellSetup.persistent_globals.simthread

    if not steppableRegistry is None:
        steppableRegistry.init(sim)

    max_num_steps = sim.getNumSteps()
    sim.start()
    if not steppableRegistry is None:
        steppableRegistry.start()

    runFinishFlag = True

    cur_step = 0
    while cur_step < max_num_steps / 100:
        simthread.beforeStep(_mcs=cur_step)
        if simthread.getStopSimulation() or CompuCellSetup.persistent_globals.user_stop_simulation_flag:
            runFinishFlag = False;
            break

        sim.step(cur_step)
        if not steppableRegistry is None:
            steppableRegistry.step(cur_step)

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
        print ("CALLING FINISH")
    else:
        sim.cleanAfterSimulation()
        # # sim.unloadModules()
        print( "CALLING UNLOAD MODULES NEW PLAYER")
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            simthread.simulationFinishedPostEvent(True)
