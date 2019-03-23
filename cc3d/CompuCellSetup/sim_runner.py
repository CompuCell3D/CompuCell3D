import argparse
import traceback
import cc3d
import sys
from os.path import *

# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup

from cc3d.CompuCellSetup.readers import readCC3DFile
#
def handle_error():
    """

    :return:
    """

    tb = traceback.format_exc()

    formatted_lines = tb.splitlines()
    error_description_line = formatted_lines[-1]

    traceback_text = "Error: " + error_description_line + "\n"
    traceback_text += tb

    simthread = CompuCellSetup.persistent_globals.simthread

    if simthread is not None:
        # simthread.emitErrorOccured('Python Error', tb)
        simthread.emitErrorFormatted( traceback_text)


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
    # sys.path.append(join(dirname(cc3d_sim_fname),'Simulation'))
    sys.path.insert(0,join(dirname(cc3d_sim_fname), 'Simulation'))

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
                exec(code)
                # exec(sim_fh.read())
                # exec(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
            except:
                traceback.print_exc(file=sys.stdout)
                handle_error()



            # traceback.format_stack()
            # # traceback.format_exc()
            # # print(traceback.format_stack())
            # traceback.print_tb()


