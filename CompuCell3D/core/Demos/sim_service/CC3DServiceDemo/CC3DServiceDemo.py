import multiprocessing
from os.path import dirname, join
import random
import statistics
import time

# Either of these works
# The first is the underlying process class of the server process presented by the second as a proxy
# However, only the second allows extensions of the service interface from within the simulation service specification
# These extensions are called "service functions". They define internal service methods that can be employed through
# the service interface and can be accessed as attributes on the service proxy.
from cc3d.CompuCellSetup.CC3DCaller import CC3DSimService
# from cc3d.core.simservice import service_cc3d as CC3DSimService

from cc3d.CompuCellSetup.CC3DCaller import CC3DCallerWorker
from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
"""
This demo teaches usage and some applications of CC3DSimService, including
    - How to interactively run a CC3D simulation in Python
    - How to pass information to and from an interactive CC3D simulation
    - How to retrofit an existing simulation without modifying simulation source code
    - How to define a non-standard simulation control sequence in parallel CC3D applications
"""
__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"


# Demo options
run_basic = True  # Run the basic demo: execute an interactive simulation
run_parallel = False  # Run the parallel demo: execute multiple simulations in parallel


simulation_fname = join(dirname(__file__), 'steppableBasedMitosis', 'steppableBasedMitosis.cc3d')
root_output_folder = join(dirname(__file__), 'steppableBasedMitosis', 'Output')
output_frequency = None
population_threshold = 10

# Parallel execution specifications; only relevant if run_parallel == True
num_workers = 5
number_of_runs = 5


# This steppable will be injected into the simulation defined in
# CompuCell3D/core/Demos/CompuCellPythonTutorial/steppableBasedMitosis
# Note that this is just a CC3D steppable in the ordinary sense; it returns
# the current number of cells through the CC3D Python API
# We're just defining it here so that we don't need to modify
# the source code of an existing simulation
class CellCounterSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        # Return the number of cells to the outside world
        pg = CompuCellSetup.persistent_globals
        # This shows up as the property "sim_output" on the controlling CC3DSimService instance
        pg.return_object = {'num_cells': len([cell for cell in self.cell_list])}


# This shows how to run and interact with an interactive CC3D simulation
def main_basic():
    # Instantiate CC3D simulation service
    cc3d_sim = CC3DSimService(cc3d_sim_fname=simulation_fname,
                              output_frequency=output_frequency,
                              output_dir=root_output_folder)
    # Inject our counting steppable
    cc3d_sim.register_steppable(steppable=CellCounterSteppable)

    # Initialize simulation
    cc3d_sim.run()

    # Perform interactive simulation run procedures
    # Normally CC3D carries out our simulations for us.
    # Here we'll perform them manually and terminate the simulation
    # according to a criterion specified in this script

    # Call startup routines
    cc3d_sim.init()
    # Call start routines
    cc3d_sim.start()
    # Run simulation until the population exceeds the initial population by a factor of population_threshold
    # Remember that we injected CellCounterSteppable into the steppableBasedMitosis
    # demo simulation to tell us how many cells there are at each simulation step
    num_cells = 1
    num_cells_initial = 1
    get_initial = True
    while num_cells / num_cells_initial < population_threshold:
        # Step simulation
        cc3d_sim.step()
        # Every 100 steps, get the current number of cells and check for end of simulation
        if cc3d_sim.current_step % 100 == 0:
            num_cells = cc3d_sim.sim_output['num_cells']
            if get_initial:
                get_initial = False
                num_cells_initial = num_cells

    # Call finish routines
    cc3d_sim.finish()
    print(cc3d_sim.profiler_report)
    print(f"{cc3d_sim.current_step} steps required to increase the  population by a factor of {population_threshold}.")


# This function is the same as the interactive run procedures in main_basic, but wraps them so that we can
# run lots of instances in parallel
# It will be called inside CC3DSimService.run()
# We'll be giving our CC3DSimService instances to CC3DCallerWorker, rather than manually controlling the simulation.
# A CC3DCallerWorker calls CC3DSimService.run(), so we'd better define our entire simulation control sequence here
def inside_run(cc3d_sim):
    # Call startup routines
    cc3d_sim.init()
    # Call start routines
    cc3d_sim.start()
    # Run simulation until the population doubles
    num_cells = 1
    num_cells_initial = 1
    get_initial = True
    while num_cells / num_cells_initial < population_threshold:
        # Step simulation
        cc3d_sim.step()
        # Every 100 steps, get the current number of cells and check for end of simulation
        if cc3d_sim.current_step % 100 == 0:
            num_cells = cc3d_sim.sim_output['num_cells']
            if get_initial:
                get_initial = False
                num_cells_initial = num_cells

    # Call finish routines
    from cc3d import CompuCellSetup
    cc3d_sim.finish()


# This shows how to define a non-standard simulation control sequence in parallel CC3D applications
def main_parallel():
    # Set up parallel execution

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start workers
    workers = [CC3DCallerWorker(tasks, results) for _ in range(num_workers)]
    [w.start() for w in workers]

    # Make some sims to run

    for i in range(number_of_runs):
        # Instantiate CC3D simulation service
        cc3d_sim = CC3DSimService(cc3d_sim_fname=simulation_fname,
                                  output_frequency=output_frequency,
                                  output_dir=root_output_folder,
                                  sim_name=str(i))
        # Inject our counting steppable
        cc3d_sim.register_steppable(steppable=CellCounterSteppable)
        # Load custom simulation control sequence
        cc3d_sim.set_inside_run(inside_run)
        # Enqueue job
        tasks.put(cc3d_sim)

    # Add a stop task "poison pill" for each of the workers
    for i in range(num_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Retrieve our sims
    cc3d_sim_list = [None] * number_of_runs
    while any([cc3d_sim is None for cc3d_sim in cc3d_sim_list]):
        result = results.get()
        sim_name = result['name']
        cc3d_sim = result['sim']
        cc3d_sim_list[int(sim_name)] = cc3d_sim

    # Post-process results
    for i, cc3d_sim in enumerate(cc3d_sim_list):
        print(f"Instance {i} measured {cc3d_sim.current_step} steps")
    avg_step = statistics.mean([cc3d_sim.current_step for cc3d_sim in cc3d_sim_list])
    print(f"{avg_step} average steps required to increase the population by a factor of {population_threshold}.")


if __name__ == '__main__':
    if run_basic:
        main_basic()
    if run_parallel:
        main_parallel()
