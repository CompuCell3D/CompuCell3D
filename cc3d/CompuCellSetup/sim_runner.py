import traceback
import sys
from os.path import *
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.readers import readCC3DFile
from cc3d.CompuCellSetup.simulation_utils import CC3DCPlusPlusError
from cc3d.cpp.CompuCell import CC3DException

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

        if simthread is not None:
            print("CALLING UNLOAD MODULES NEW PLAYER")
            simthread.emitErrorFormatted('Error: ' + str(exception_obj.message))
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
        cc3d_simulation_data_handler = readCC3DFile(fileName=cc3d_sim_fname)
    except FileNotFoundError:
        print('Could not find cc3d_sim_fname')
        return

    CompuCellSetup.cc3dSimulationDataHandler = cc3d_simulation_data_handler
    # todo - need to find a better solution ot append and remove pythonpath of the simulation object
    sys.path.insert(0, join(dirname(cc3d_sim_fname), 'Simulation'))

    with open(cc3d_simulation_data_handler.cc3dSimulationData.pythonScript) as sim_fh:
        try:
            code = compile(sim_fh.read(), cc3d_simulation_data_handler.cc3dSimulationData.pythonScript, 'exec')
        except Exception as e:
            code = None
            traceback.print_exc(file=sys.stdout)
            handle_error(e)

        output_directory = CompuCellSetup.persistent_globals.output_directory

        if cc3d_simulation_data_handler and output_directory is not None:
            cc3d_simulation_data_handler.copy_simulation_data_files(output_directory)

        if code is not None:
            try:
                exec(code, globals(), locals())

            except RuntimeError as e:
                # note, we have fixture in SWIG that Converts CC3DException into RuntimeError
                traceback.print_exc(file=sys.stderr)
                handle_error(e)

                # we will exit with code 1 only in the non-player mode
                if not CompuCellSetup.persistent_globals.player_type:
                    sys.exit(1)

            except Exception as e:
                if str(e).startswith("Unknown exception"):
                    print("Likely exception from C++ function that was not marked to throw an exception")
                traceback.print_exc(file=sys.stdout)
                handle_error(e)

                # we will exit with code 1 only in the non-player mode
                if not CompuCellSetup.persistent_globals.player_type:
                    sys.exit(1)
