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

    cml_parser.add_argument('--currentDir', required=False, action='store',
                            help='path to current directory')


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

    rollbackImporter.uninstall()
