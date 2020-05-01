
from cc3d.core.Validation.PDESuiteTest import PDESuiteTestSteppable
import math


class PDETestPeriodicFESteppable(PDESuiteTestSteppable):

    def __init__(self, frequency=1):

        super().__init__(frequency)

        self.error_contours = True
        self.compare_solvers = True

        # These names correspond to those given in XML solver specification
        self.fields_list_names = [
            "Field_DiffusionSolverFE",
            "Field_ReactionDiffusionSolverFE"
        ]

        self.dir_sel = 'y'

        # For initial distributions
        self.__mean_val = 1.0
        self.__amp_val = 0.5
        self.__wave_num = 2

        # Convenience function for setting analytic solution
        self.load_transient_sine_coeffs(_dir=self.dir_sel,
                                        _diff_coeff=0.1,
                                        _mean_val=self.__mean_val,
                                        _ampl_val=self.__amp_val,
                                        _wave_num=self.__wave_num)

        self.load_inits()

    def start(self):

        # Set sample points
        if self.dir_sel == 'x':
            self.sample_set_pixels = [[x, int(self.dim.y/2), int(self.dim.z/2)] for x in range(self.dim.x)]
        elif self.dir_sel == 'y':
            self.sample_set_pixels = [[int(self.dim.x/2), y, int(self.dim.z/2)] for y in range(self.dim.y)]
        elif self.dir_sel == 'z':
            self.sample_set_pixels = [[int(self.dim.x/2), int(self.dim.y/2), z] for z in range(self.dim.z)]

        PDESuiteTestSteppable.start(self)

        # Set initial fields in numerical solutions
        for x, y, z in self.every_pixel():
            this_val = self.analytic_lambda(x, y, z, 0)
            self.field.Field_DiffusionSolverFE[x, y, z] = this_val
            self.field.Field_ReactionDiffusionSolverFE[x, y, z] = this_val

    def step(self, mcs):

        PDESuiteTestSteppable.step(self, mcs)

    def finish(self):
        
        PDESuiteTestSteppable.finish(self)
