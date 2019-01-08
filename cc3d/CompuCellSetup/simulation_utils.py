from cc3d import CompuCellSetup

def stop_simulation():
    """
    Stops simulation
    :return:
    """
    CompuCellSetup.persistent_globals.user_stop_simulation_flag = True

# legacy api
stopSimulation = stop_simulation