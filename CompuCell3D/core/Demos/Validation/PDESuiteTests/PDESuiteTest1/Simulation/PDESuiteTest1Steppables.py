
from cc3d.core.Validation.PDESuiteTest import PDESuiteTestSteppable


class PDESuiteTest1Steppable(PDESuiteTestSteppable):

    def __init__(self, frequency=1):

        super().__init__(frequency)

        self.error_contours = True
        self.compare_solvers = True

        # These names correspond to those given in XML solver specification
        self.fields_list_names = [
            "Field_DiffusionSolverFE",
            "Field_ReactionDiffusionSolverFE",
            "Field_SteadyStateDiffusionSolver2D"
        ]

        self.analytic_setup = 'line_x'

        self.load_inits()

    def start(self):
        """
        any code in the start function runs before MCS=0
        """

        self.sample_set_pixels = [[x, 0, 0] for x in range(self.dim.x)]

        PDESuiteTestSteppable.start(self)

    def finish(self):
        """
        Finish Function is called after the last MCS
        """
