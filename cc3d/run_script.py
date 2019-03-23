import argparse
import traceback
import cc3d
import sys
from os.path import *

# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.RollbackImporter import RollbackImporter


def process_cml():
    """

    :return:
    """


    cml_parser = argparse.ArgumentParser(description='CompuCell3D Player 5')
    cml_parser.add_argument('-i', '--input', required=False, action='store',
                            help='path to the CC3D project file (*.cc3d)')



    return cml_parser.parse_args()

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




if __name__ =='__main__':
    args = process_cml()
    cc3d_sim_fname = args.input

    rollbackImporter = RollbackImporter()

    cc3d.CompuCellSetup.persistent_globals.simulation_file_name = cc3d_sim_fname

    run_cc3d_project(cc3d_sim_fname=cc3d_sim_fname)

    # cc3dSimulationDataHandler = readCC3DFile(fileName=sim_fname)
    #
    # CompuCellSetup.cc3dSimulationDataHandler = cc3dSimulationDataHandler
    # import sys
    # from os.path import *
    # # todo - need to find a better solution ot append and remove pythonpath of the simulation object
    # sys.path.append(join(dirname(sim_fname),'Simulation'))
    #
    # # execfile(CompuCellSetup.simulationPaths.simulationPythonScriptName)
    # with open(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript) as sim_fh:
    #     try:
    #         code = compile(sim_fh.read(), cc3dSimulationDataHandler.cc3dSimulationData.pythonScript, 'exec')
    #     except:
    #         code = None
    #         traceback.print_exc(file=sys.stdout)
    #
    #     # exec(code)
    #     if code is not None:
    #         try:
    #             exec(code)
    #             # exec(sim_fh.read())
    #             # exec(cc3dSimulationDataHandler.cc3dSimulationData.pythonScript)
    #         except:
    #             traceback.print_exc(file=sys.stdout)
    #
    #         # traceback.format_stack()
    #         # # traceback.format_exc()
    #         # # print(traceback.format_stack())
    #         # traceback.print_tb()
    # execfile()
    # print



"""
--input=d:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_project\cellsort_2D.cc3d
--input=d:\CC3D_PY3_GIT\CompuCell3D\tests\test_data\cellsort_project_py_step\cellsort_2D.cc3d
"""
