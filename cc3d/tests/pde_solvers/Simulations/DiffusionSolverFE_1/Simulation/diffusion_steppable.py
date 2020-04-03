from cc3d.core.PySteppables import *
from cc3d.tests.test_utils.UnittestSteppables import *


class DiffusionSolverSteppable(UnittestSteppablePdeSolver):
    def __init__(self, frequency=100):
        UnittestSteppablePdeSolver.__init__(self, frequency)
        self.generate_reference_flag = False
        self.test_mcs = 500
        self.model_output_csv = Path(__file__).parent.parent.joinpath('reference_output', 'model.output.csv')




# from cc3d.core.PySteppables import *
# import numpy as np
# from os.path import *
# import os
# import sys
# import pandas as pd
# from collections import OrderedDict
# import numpy.testing as npt
# from inspect import currentframe, getframeinfo
# from pathlib import Path
#
#
# def get_test_output_dir():
#
#     try:
#         test_output_dir = os.environ['CC3D_TEST_OUTPUT_DIR']
#     except KeyError:
#         test_output_dir = Path().home().joinpath('CC3D_test_output_dir')
#
#     return test_output_dir
#
#
# def get_linenumber_fname():
#     cf = currentframe()
#     return f'line: {cf.f_back.f_lineno} in {getframeinfo(cf).filename} '
#
# class demo(SecretionBasePy):
#     pass
#
# class DiffusionSolverSteppable(SteppableBasePy):
#     def __init__(self, frequency=100):
#         SteppableBasePy.__init__(self, frequency)
#         self.generate_reference_flag = False
#
#     def start(self):
#         """
#         """
#
#     def run_regression_tests(self):
#
#         model_output_csv = Path(__file__).parent.parent.joinpath('reference_output', 'model.output.csv')
#         in_df = pd.read_csv(model_output_csv)
#         in_df = in_df.sort_values(by=['x', 'y', 'z'])
#
#         reference_output_df = self.generate_reference_output()
#         reference_output_df = reference_output_df.sort_values(by=['x', 'y', 'z'])
#
#         test_log = ''
#         try:
#             npt.assert_array_almost_equal(in_df.x.values, reference_output_df.x.values, decimal=2)
#         except AssertionError as e:
#             test_log += f'{str(e)}\n'
#
#         npt.assert_array_almost_equal(in_df.y.values, reference_output_df.y.values, decimal=2)
#         npt.assert_array_almost_equal(in_df.z.values, reference_output_df.z.values, decimal=2)
#         try:
#             npt.assert_array_almost_equal(in_df.val.values - 1, reference_output_df.val.values, decimal=10)
#         except AssertionError as e:
#             test_log += f'{get_linenumber_fname()} \n {str(e)}\n'
#
#         test_output_dir = get_test_output_dir()
#         print(sys.stderr, f'test_output_dir={test_output_dir}')
#         print(test_log)
#
#
#     def generate_reference_output(self):
#
#         field = self.field.FGF
#
#         x_list = []
#         y_list = []
#         z_list = []
#         val_list = []
#
#         for i, j, k in self.every_pixel():
#             x_list.append(i)
#             y_list.append(j)
#             z_list.append(k)
#             val_list.append(field[i, j, k])
#
#         out_df = pd.DataFrame(
#             OrderedDict(
#                 [
#                     ('x', x_list),
#                     ('y', y_list),
#                     ('z', z_list),
#                     ('val', val_list),
#                 ]
#             )
#
#         )
#         return out_df
#
#     def step(self, mcs):
#         if mcs == 100:
#             print(f'get_test_output_dir()={get_test_output_dir()}')
#             print(sys.stderr, 'exiting')
#             sys.exit()
#
#         if mcs == 500:
#
#             if self.generate_reference_flag:
#                 out_df = self.generate_reference_output()
#                 output_basename = 'model.output.csv'
#                 output_path = output_basename
#                 output_dir = self.output_dir
#                 if output_dir is not None:
#                     output_path = Path(output_dir).joinpath(output_basename)
#                     # create folder to store data
#                     output_path.parent.mkdir(parents=True, exist_ok=True)
#                 out_df.to_csv(output_path, index=False)
#
#             else:
#                 self.run_regression_tests()
#
#             self.stop_simulation()
