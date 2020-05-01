# This optimization routine calibrates chemotactic Lagrange multiplier according to a specified chemotaxing speed

import multiprocessing
from os.path import dirname, join
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller
from cc3d.CompuCellSetup.CC3DCaller import CC3DCallerWorker
import time

# General setup

# Specify location of simulation file
simulation_fname = join(dirname(__file__), 'ChemotaxisRunModel', 'ChemotaxisRunModel.cc3d')
# Specify root directory of results
res_output_root = join(dirname(__file__), 'res')
# Specify number of workers
num_workers = 5


def run_trials(num_runs, iteration_num, lam_trial):
    """
    Runs simulation and store simulation results
    :param num_runs: number of simulation runs
    :param iteration_num: integer label for storing results according to an iteration
    :param lam_trial: chemotactic Lagrange multiplier to simulate
    :return: mean horizontal center of mass over all runs
    """

    root_output_folder = join(res_output_root, f'iteration_{iteration_num}')

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start workers
    workers = [CC3DCallerWorker(tasks, results) for i in range(num_workers)]
    [w.start() for w in workers]

    # Enqueue jobs
    for i in range(num_runs):
        cc3d_caller = CC3DCaller(cc3d_sim_fname=simulation_fname,
                                 output_frequency=10,
                                 screenshot_output_frequency=10,
                                 output_dir=join(root_output_folder, f'trial_{i}'),
                                 result_identifier_tag=i,
                                 sim_input=lam_trial
                                 )
        tasks.put(cc3d_caller)

    # Add a stop task for each of worker
    for i in range(num_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Return mean result
    trial_results = []
    while num_runs:
        result = results.get()
        trial_results.append(result['result'])
        num_runs -= 1

    return np.average(trial_results)


def run_trials_no_store(num_runs, lam_trial):
    """
    Runs simulation without storing simulation results
    :param num_runs: number of simulation runs
    :param lam_trial: chemotactic Lagrange multiplier to simulate
    :return: mean horizontal center of mass over all runs
    """

    root_output_folder = join(res_output_root, f'tmp')

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start workers
    workers = [CC3DCallerWorker(tasks, results) for i in range(num_workers)]
    [w.start() for w in workers]

    # Enqueue jobs
    for i in range(num_runs):
        cc3d_caller = CC3DCaller(cc3d_sim_fname=simulation_fname,
                                 output_dir=join(root_output_folder, f'trial_{i}'),
                                 result_identifier_tag=i,
                                 sim_input=lam_trial
                                 )
        tasks.put(cc3d_caller)

    # Add a stop task for each of worker
    for i in range(num_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Return mean result
    trial_results = []
    while num_runs:
        result = results.get()
        trial_results.append(result['result'])
        num_runs -= 1

    return np.average(trial_results)


def main():
    # Number of simulation runs per evaluation
    num_runs = 10
    # Maximum number of optimization iterations
    num_iterations = 50
    # Target horizontal center of mass after 1k MCS
    target_xcom = 150

    # Cost function of optimization
    def cost_fun(x):
        res = run_trials_no_store(num_runs, x)
        return (res - target_xcom) ** 2

    # Bounds for chemotactic Lagrange multiplier
    lam_min = 1E2
    lam_max = 1E4

    # Solve!

    solve_time = time.time()

    opt_res = minimize_scalar(cost_fun,
                              bounds=(lam_min, lam_max),
                              method='bounded',
                              options={'maxiter': num_iterations, 'disp': True})

    solve_time = time.time() - solve_time

    # Output results with optimal solution
    run_trials(num_runs, 0, opt_res.x)

    # Print and store optimization results summary
    print(opt_res)
    print(solve_time)

    res_out_summ = join(dirname(__file__), 'opt_summary.dat')
    with open(res_out_summ, 'w') as fout:
        fout.write(str(opt_res))
        fout.write('\n\n')
        fout.write('Solution time: {} s'.format(solve_time))


if __name__ == '__main__':
    main()
