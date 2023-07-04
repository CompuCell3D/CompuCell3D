from cc3d.core.PySteppables import *
import numpy as np
from os.path import *
import os
import sys
import pandas as pd
from collections import OrderedDict
import numpy.testing as npt
from inspect import currentframe, getframeinfo
from pathlib import Path


class UnittestSteppablePdeSolver(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)
        self.generate_reference_flag = False
        self.test_mcs = 500
        self.fields_to_store = ['FGF']

        # override it in actual simulation code
        self.model_output_core = 'model.output'
        # override it in actual simulation code
        self.model_path = None

    @staticmethod
    def get_test_output_dir():

        try:
            test_output_dir = os.environ['CC3D_TEST_OUTPUT_DIR']
        except KeyError:
            test_output_dir = Path().home().joinpath('CC3D_test_output_dir')

        return test_output_dir

    @staticmethod
    def get_test_output_summary():

        try:
            test_output_summary = os.environ['CC3D_TEST_OUTPUT_SUMMARY']
        except KeyError:
            test_output_summary = join(UnittestSteppablePdeSolver.get_test_output_dir(), 'test_summary.csv')

        return test_output_summary

    @staticmethod
    def get_regression_error_summary():

        try:
            regression_errors_summary = Path(
                os.environ['CC3D_TEST_OUTPUT_SUMMARY']).parent.joinpath('test_regression_errors.csv')
        except KeyError:
            regression_errors_summary = join(UnittestSteppablePdeSolver.get_test_output_dir(), 'test_regression_errors.csv')

        return regression_errors_summary


    @staticmethod
    def get_linenumber_fname():
        cf = currentframe()
        return f'line: {cf.f_back.f_lineno} in {getframeinfo(cf).filename} '

    def run_pde_solver_regression_tests(self):

        for field_to_store in self.fields_to_store:
            found_regression_error = False
            reference_field_csv = f'{self.model_output_core}.{field_to_store}.csv'
            in_df = pd.read_csv(reference_field_csv)

            in_df = in_df.sort_values(by=['x', 'y', 'z'])

            reference_output_df = self.generate_pde_solver_reference_output(field_to_store=field_to_store)
            reference_output_df = reference_output_df.sort_values(by=['x', 'y', 'z'])

            test_log = ''
            try:
                npt.assert_array_almost_equal(in_df.x.values, reference_output_df.x.values, decimal=2)
            except AssertionError as e:
                test_log += f'Field: {field_to_store} {self.get_linenumber_fname()} \n {str(e)}\n'

            try:
                npt.assert_array_almost_equal(in_df.y.values, reference_output_df.y.values, decimal=2)
            except AssertionError as e:
                test_log += f'Field: {field_to_store} {self.get_linenumber_fname()} \n {str(e)}\n'

            try:
                npt.assert_array_almost_equal(in_df.z.values, reference_output_df.z.values, decimal=2)
            except AssertionError as e:
                test_log += f'Field: {field_to_store} {self.get_linenumber_fname()} \n {str(e)}\n'

            try:
                npt.assert_array_almost_equal(in_df.val.values, reference_output_df.val.values, decimal=4)
            except AssertionError as e:
                test_log += f'Field: {field_to_store} {self.get_linenumber_fname()} \n {str(e)}\n'

            test_output_dir = self.get_test_output_dir()

            msg = f'---------------------------------------\n' \
                  f'field: {field_to_store} : simulation: {self.model_path} \n'
            if test_log.strip():
                msg += f'error log: {test_log} \n'
                found_regression_error = True
            else:
                msg += f'OK\n'
            msg += f'---------------------------------------\n'

            try:
                with open(self.get_test_output_summary(), 'a') as out_file:

                    out_file.write(msg)
            except FileNotFoundError:
                print(msg)
            if found_regression_error:

                try:
                    with open(self.get_regression_error_summary(), 'a') as out_file:
                        out_file.write(msg)
                except FileNotFoundError:
                    print(msg)
    def generate_pde_solver_reference_output(self, field_to_store):

        field = CompuCell.getConcentrationField(self.simulator, field_to_store)

        x_list = []
        y_list = []
        z_list = []
        val_list = []

        for i, j, k in self.every_pixel():
            x_list.append(i)
            y_list.append(j)
            z_list.append(k)
            val_list.append(field[i, j, k])

        out_df = pd.DataFrame(
            OrderedDict(
                [
                    ('x', x_list),
                    ('y', y_list),
                    ('z', z_list),
                    ('val', val_list),
                ]
            )

        )
        return out_df

    def step_pde_solver_test(self, mcs):
        if mcs == self.test_mcs:

            if self.generate_reference_flag:
                for field_to_store in self.fields_to_store:
                    out_df = self.generate_pde_solver_reference_output(field_to_store=field_to_store)
                    output_basename = Path(f'{self.model_output_core}.{field_to_store}.csv').name
                    output_path = output_basename
                    output_dir = self.output_dir
                    if output_dir is not None:
                        output_path = Path(output_dir).joinpath(output_basename)
                        # create folder to store data
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    out_df.to_csv(output_path, index=False)

            else:
                self.run_pde_solver_regression_tests()

            self.stop_simulation()

    def step(self, mcs):
        self.step_pde_solver_test(mcs=mcs)
