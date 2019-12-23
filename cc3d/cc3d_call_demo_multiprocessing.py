import multiprocessing
from cc3d.CC3DCaller import CC3DCaller
from cc3d.CC3DCaller import CC3DCallerWorker


def main():
    num_workers = 4
    num_jobs = 10

    sim_fnames = [r'd:\CC3DProjects\demo_py3\cellsort_2D\cellsort_2D.cc3d'] * num_jobs

    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # Start workers
    num_consumers = multiprocessing.cpu_count() * 2
    print('Creating %d consumers' % num_consumers)
    workers = [CC3DCallerWorker(tasks, results) for i in range(num_workers)]
    for w in workers:
        w.start()

    # Enqueue jobs

    for i, sim_fname in enumerate(sim_fnames):
        cc3d_caller = CC3DCaller(cc3d_sim_fname=sim_fname,
                                 screenshot_output_frequency=10,
                                 output_dir=r'c:\Users\m\CC3DWorkspace\cellsort_' + f'{i}',
                                 result_identifier_tag=i
                                 )
        tasks.put(cc3d_caller)

    # Add a stop task "poison pill" for each of the workers
    for i in range(num_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    # Start printing results
    while num_jobs:
        result = results.get()
        print('Result:', result)
        num_jobs -= 1


if __name__ == '__main__':
    main()

