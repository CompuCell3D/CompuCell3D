import time

from cc3d import CompuCellSetup
from cc3d.CompuCellSetup import (
    check_for_cpp_errors, incorporate_script_steering_changes, initialize_cc3d_sim, print_profiling_report)


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

    initialize_cc3d_sim(sim, simthread)

    restart_manager = pg.restart_manager
    init_using_restart_snapshot_enabled = restart_manager.restart_enabled()

    # simthread.waitForInitCompletion()
    # simthread.waitForPlayerTaskToFinish()

    if steppable_registry is not None:
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
