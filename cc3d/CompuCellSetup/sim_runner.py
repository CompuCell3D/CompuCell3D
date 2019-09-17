import traceback
import cc3d
import sys
from os.path import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.readers import readCC3DFile
from cc3d.CompuCellSetup.simulation_utils import CC3DCPlusPlusError


def handle_error(exception_obj):
    """

    :return:
    """
    simthread = CompuCellSetup.persistent_globals.simthread
    sim = CompuCellSetup.persistent_globals.simulator
    steppable_registry = CompuCellSetup.persistent_globals.steppable_registry

    if sim is not None:
        sim.cleanAfterSimulation()

    # printing c++ error
    if isinstance(exception_obj, CC3DCPlusPlusError):
        simthread.emitErrorFormatted('Error: ' + str(exception_obj.message))

        print("CALLING UNLOAD MODULES NEW PLAYER")
        if simthread is not None:
            simthread.sendStopSimulationRequest()
            simthread.simulationFinishedPostEvent(True)
        if steppable_registry is not None:
            steppable_registry.clean_after_simulation()

        return

    # printing python stack trace for Python errors
    tb = traceback.format_exc()

    formatted_lines = tb.splitlines()
    error_description_line = formatted_lines[-1]

    traceback_text = "Error: " + error_description_line + "\n"
    traceback_text += tb

    if simthread is not None:
        simthread.emitErrorFormatted(traceback_text)


def run_cc3d_project(cc3d_sim_fname):
    """

    :param cc3d_sim_fname:
    :return:
    """

    try:
        cc3dSimulationDataHandler = readCC3DFile(fileName=cc3d_sim_fname)
    except FileNotFoundError:
        print('Could not find cc3d_sim_fname')
        return

    CompuCellSetup.cc3dSimulationDataHandler = cc3dSimulationDataHandler
    # todo - need to find a better solution ot append and remove pythonpath of the simulation object
    sys.path.insert(0, join(dirname(cc3d_sim_fname), 'Simulation'))

    # execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
    with open(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript) as sim_fh:
        try:
            code = compile(sim_fh.read(), cc3dSimulationDataHandler.cc3dSimulationData.pythonScript, 'exec')
        except:
            code = None
            traceback.print_exc(file=sys.stdout)
            handle_error()

        # exec(code)
        if code is not None:
            try:
                exec(code, globals(), locals())
                # exec(sim_fh.read())
                # exec(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)

            except CC3DCPlusPlusError as e:
                # handling C++ error
                handle_error(e)
            except Exception as e:
                traceback.print_exc(file=sys.stdout)
                handle_error(e)
