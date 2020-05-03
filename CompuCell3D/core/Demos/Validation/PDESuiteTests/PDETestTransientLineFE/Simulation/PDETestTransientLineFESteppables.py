
from cc3d.core.Validation.PDESuiteTest import PDESuiteTestSteppable


class PDETestTransientLineFESteppable(PDESuiteTestSteppable):
    """
    This steppable is designed for testing transient solutions of DiffusionSolverFE and ReactionDiffusionSolverFE
    for Neumann conditions along two directions, Dirichlet conditions along one direction, and homogeneous initial
    conditions.
    """

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
        self.init_field_val = 0.25

        # Convenience function for setting analytic solution
        self.load_transient_line_coeffs(_dir=self.dir_sel,
                                        _initial_val=self.init_field_val,
                                        _diff_coeff=0.1,
                                        _lower_val=0.1,
                                        _upper_val=1.0)

        self.load_inits()

    def start(self):

        # Set sample points
        if self.dir_sel == 'x':
            self.sample_set_pixels = [[x, int(self.dim.y/2), int(self.dim.z/2)] for x in range(self.dim.x)]
        elif self.dir_sel == 'y':
            self.sample_set_pixels = [[int(self.dim.x/2), y, int(self.dim.z/2)] for y in range(self.dim.y)]
        elif self.dir_sel == 'z':
            self.sample_set_pixels = [[int(self.dim.x/2), int(self.dim.y/2), z] for z in range(self.dim.z)]

        # Set initial fields in numerical solutions
        self.field.Field_ReactionDiffusionSolverFE[:, :, :] = self.init_field_val
        self.field.Field_DiffusionSolverFE[:, :, :] = self.init_field_val

        PDESuiteTestSteppable.start(self)

    def step(self, mcs):
        # Don't take measurements over first 200 MCS, since approximation of analytic solution is unstable
        if mcs < 500:
            return

        PDESuiteTestSteppable.step(self, mcs)

    def finish(self):
        pass
