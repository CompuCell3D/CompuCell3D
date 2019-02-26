from test_utils.RunSpecs import RunSpecs
from test_utils.RunExecutor import RunExecutor

from test_utils.common import find_file_in_dir
from os.path import *
import os
import shutil
import sys

# windows settings:
# if sys.platform.startswith('win'):
#     cc3d_run_script = r'C:\CompuCell3D\compucell3d.bat'
#     cc3d_demo_dir = r'C:\CompuCell3D\Demos'
#     demo_script = r'C:\CompuCell3D\Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d'
#     test_output_root = 'rC:\CompuCell3D\CompuCell3D_test_output'

if sys.platform.startswith('win'):
    cc3d_run_script = r'C:\CompuCell3D-64bit\compucell3d.bat'
    cc3d_demo_dir = r'C:\CompuCell3D-64bit\Demos'
    demo_script = r'C:\CompuCell3D-64bit\Demos\Models\cellsort\cellsort_2D\cellsort_2D.cc3d'
    test_output_root = r'C:\CompuCell3D-64bit\CompuCell3D_test_output'

# if sys.platform.startswith('win'):
#     cc3d_run_script = r'd:\Program Files\3710\compucell3d.bat'
#     cc3d_demo_dir = r'd:\Program Files\3710\Demos'
#     demo_script = r'd:\Program Files\3710\Demos\Models\cellsort\cellsort_2D\cellsort_2D.cc3d'
#     test_output_root = r'd:\Program Files\3710\CompuCell3D_test_output'


if sys.platform.startswith('darwin'):
    #osx settings
    cc3d_run_script = r'/Users/m/Demo/CC3D_3.7.6/cc3d_test.command'
    cc3d_demo_dir = r'/Users/m/Demo/CC3D_3.7.6/Demos'
    demo_script = r'/Users/m/Demo/CC3D_3.7.6/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d'
    test_output_root = 'r/Users/m/cc3d_tests'


# l = ['/Users/m/Demo/CC3D_3.7.6/cc3d_test.command','/Users/m/Demo/CC3D_3.7.6/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d']
l = [cc3d_run_script, demo_script]
common_pref = commonprefix(l)

for p in l:
    print relpath(p, common_pref)

# cc3d_projects = find_file_in_dir('/Users/m/Demo/CC3D_3.7.6/Demos', '*.cc3d')
cc3d_projects = find_file_in_dir(cc3d_demo_dir, '*.cc3d')
cc3d_projects_common_prefix = commonprefix(cc3d_projects)

rs = RunSpecs()

# rs.run_command = '/Users/m/Demo/CC3D_3.7.6/cc3d_test.command'
rs.run_command = cc3d_run_script
rs.cc3d_project = ''
rs.num_steps = 100
# rs.test_output_root = '/Users/m/cc3d_tests'
rs.test_output_root = test_output_root
rs.test_output_dir = ''

# clean test_output_dir
try:
    shutil.rmtree(rs.test_output_root)
except OSError:
    pass

try:
    os.makedirs(rs.test_output_root)
except OSError:
    pass

# writing to the file list of files to be tested
with open(join(test_output_root,'cc3d_simulation_test_plan.txt'),'a') as fout:
    for i, cc3d_project in enumerate(cc3d_projects):
        fout.write('{}\n'.format(cc3d_project))



error_runs = []

for i, cc3d_project in enumerate(cc3d_projects):

    rs.cc3d_project = cc3d_project
    rs.test_output_dir = relpath(cc3d_project, cc3d_projects_common_prefix)
    run_executor = RunExecutor(run_specs=rs)
    run_executor.run()
    run_status = run_executor.get_run_status()
    if run_status:
        error_tuple = (rs.cc3d_project, run_status)
        error_runs.append(error_tuple)
        with open(join(test_output_root,'cc3d_simulation_tests.txt'),'a') as fout:
            fout.write('{}\n'.format(error_tuple))

    # if i > 10:
    #     break

        # break

if not len(error_runs):
    print()
    print('-----------------ALL SIMULATIONS RUN SUCCESSFULLY----------------------')
    print()

else:
    print()
    print('-----------------THERE WERE ERRORS IN THE SIMULATIONS----------------------')
    print()

    for error_run in error_runs:
        print (error_run)

# run_executor = RunExecutor(run_specs=rs)
# run_executor.run()
