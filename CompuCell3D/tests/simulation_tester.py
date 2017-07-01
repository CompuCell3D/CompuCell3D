from test_utils.RunSpecs import RunSpecs
from test_utils.RunExecutor import RunExecutor

rs = RunSpecs()

rs.run_command = '/Users/m/Demo/CC3D_3.7.6/cc3d_test.command'
rs.cc3d_project = '/Users/m/Demo/CC3D_3.7.6/Demos/Models/cellsort/cellsort_2D/cellsort_2D.cc3d'
rs.num_mcs = 100
rs.test_output_root = '/Users/m/cc3d_tests'

run_executor = RunExecutor(run_specs=rs)
run_executor.run()
