from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell


class PDESuiteTestSteppable(SteppableBasePy):
    """
    Base class for testing solvers in the PDE Solver Suite
    Usage: In a subclass, *load_inits* should be called after specifying all *init* options
    *self.fields_list_names*: a mandatory string list of fields names specified in XML for all solvers
    *self.analytic_setup*: specify a steady-state analytic solution mode
        == None     ->  no analytic solution
        == 'manual' ->  manually specify a solution field in *self.analytic_sol*
        == 'lambda' ->  specify using a lambda function with signature (x, y, z);
                        lambda function goes in *self.analytic_lambda*
        See source code for method *start* for built-in solutions
    *self.error_contours* = True -> generate contours of error distributions w.r.t. a specified analytic solution
    *self.compare_solvers* = True -> calculate maximum difference between each solver at every call to *step*
    *self.sample_set_pixels*:   an optional list of points (*[x, y, z]* or *Point3D*) where results should be sampled
                                Should be specified in *start* before calling *PDESuiteTestSteppable.start*
    """
    def __init__(self, frequency=1):
        super().__init__(frequency=frequency)

        # These names correspond to those given in XML solver specification
        self.fields_list_names = []

        self.analytic_setup = None
        self.analytic_lambda = None

        self.error_contours = False
        self.compare_solvers = True

        # Specify a sample set here
        self.sample_set_pixels = None

        self.sample_set_values = None

        self.analytic_sol = None

        self.plot_win_comp = None
        self.plot_win_sol = None
        self.plot_win_err = None

        self.error_fields_names_dict = {}

        self.fields_list = []
        self.fields_dict = {}
        self.error_fields_dict = {}

        self.max_error_dict = {}
        self.max_diff_dict = {}

    def start(self):
        """
        Loads specifications described in subclasses' start()
        :return:
        """
        # Check specified sample pixels
        if self.sample_set_pixels is not None:
            if not self.sample_set_pixels:
                self.sample_set_pixels = None

            for i in range(len(self.sample_set_pixels)):
                if isinstance(self.sample_set_pixels[i], CompuCell.Point3D):
                    self.sample_set_pixels[i] = [self.sample_set_pixels[i].x,
                                                 self.sample_set_pixels[i].y,
                                                 self.sample_set_pixels[i].z]

            self.sample_set_pixels = [pixel for pixel in self.sample_set_pixels if (0 <= pixel[0] <= self.dim.x and
                                                                                    0 <= pixel[1] <= self.dim.y and
                                                                                    0 <= pixel[2] <= self.dim.z)]

        # Generate analytic solution if requested
        if self.analytic_setup is not None:
            self.analytic_sol = self.field.analytic_sol

            if self.analytic_setup == 'line_x':  # 0 on -x bounadry, 1 on +x boundary
                self.analytic_lambda = lambda _x, _y, _z: (_x + 1.0) / (self.dim.x + 1.0)
            elif self.analytic_setup == 'line_y':  # 0 on -y bounadry, +y on top boundary
                self.analytic_lambda = lambda _x, _y, _z: (_y + 1.0) / (self.dim.y + 1.0)
            elif self.analytic_setup == 'line_z':  # 0 on -z bounadry, +z on top boundary
                self.analytic_lambda = lambda _x, _y, _z: (_z + 1.0) / (self.dim.z + 1.0)
            elif self.analytic_setup == 'manual':  # Use this option to specify manually
                pass
            elif self.analytic_setup == 'lambda':  # Use this option to specify with a lambda expression
                pass
            else:
                self.analytic_lambda = lambda _x, _y, _z: 0.0

            if self.analytic_setup != 'manual':
                for x, y, z in self.every_pixel():
                    self.analytic_sol[x, y, z] = self.analytic_lambda(x, y, z)

        # Get solver fields and error fields if requested
        for field_name in self.fields_list_names:
            this_field = CompuCell.getConcentrationField(self.simulator, field_name)
            self.fields_list.append(this_field)
            self.fields_dict[this_field] = field_name
            if self.error_contours:
                self.error_fields_dict[this_field] = getattr(self.field, self.error_fields_names_dict[field_name])
            self.max_error_dict[this_field] = 0.0

        color_list = ['red', 'green', 'blue', 'black']

        # Setup for comparing solver results if requested
        if self.compare_solvers:
            for field_s in self.fields_list:
                self.max_diff_dict[field_s] = {}
                for field_t in [this_field for this_field in self.fields_list if this_field is not field_s]:
                    self.max_diff_dict[field_s][field_t] = 0.0

            self.plot_win_comp = self.add_new_plot_window(
                title='Solver differences',
                x_axis_title='Solver label',
                y_axis_title='Max. relative difference',
                x_scale_type='Linear',
                y_scale_type='Linear',
                grid=True,
                config_options={'legend': True}
            )
            idx = 0
            for field_name in self.fields_dict.values():
                self.plot_win_comp.add_plot(plot_name=field_name, style='Steps', color=color_list[idx], alpha=50)
                idx += 1

        # Prep for post-processing of sample set if requested
        if self.sample_set_pixels is not None:
            self.sample_set_values = {k: [] for k in self.fields_list}

            self.plot_win_sol = self.add_new_plot_window(
                title='Solution sample set',
                x_axis_title='Sample point number',
                y_axis_title='Sample value',
                x_scale_type='Linear',
                y_scale_type='Linear',
                grid=True,
                config_options={'legend': True}
            )
            idx = 0
            for field_name in self.fields_dict.values():
                self.plot_win_sol.add_plot(field_name, style='Dots', color=color_list[idx])
                idx += 1

            if self.analytic_setup is not None:
                self.plot_win_sol.add_plot("analytic_sol", style='Dots', color=color_list[idx])

                self.plot_win_err = self.add_new_plot_window(
                    title='Solution sample set error',
                    x_axis_title='Sample point number',
                    y_axis_title='Sample value error',
                    x_scale_type='Linear',
                    y_scale_type='Linear',
                    grid=True,
                    config_options={'legend': True}
                )
                self.plot_win_err.pW.setXRange(0, len(self.sample_set_pixels) - 1)
                self.plot_win_err.pW.setYRange(-1, 1)
                idx = 0
                for field_name in self.fields_dict.values():
                    idx += 1
                    self.plot_win_err.add_plot(field_name, style='Dots', color=color_list[idx])

    def step(self, mcs):
        # Max errors with analytic solution go here when available
        if self.analytic_sol is not None:
            self.max_error_dict = {k: 0.0 for k in self.max_error_dict.keys()}

        # Max differences between solvers go here when requested
        if self.compare_solvers:
            for field_s in self.max_diff_dict.keys():
                for field_t in self.max_diff_dict[field_s].keys():
                    self.max_diff_dict[field_s][field_t] = 0.0

        for x, y, z in self.every_pixel():
            # Compare with analytic solution if available
            if self.analytic_sol is not None:
                this_analytic_sol = self.analytic_sol[x, y, z]

                if this_analytic_sol != 0:
                    for this_field in self.fields_list:
                        this_error = (this_field[x, y, z] - this_analytic_sol) / this_analytic_sol
                        if self.error_contours:
                            self.error_fields_dict[this_field][x, y, z] = this_error
                        if abs(this_error) > abs(self.max_error_dict[this_field]):
                            self.max_error_dict[this_field] = this_error
                else:
                    this_error = 0.0
                    for this_field in self.fields_list:
                        if self.error_contours:
                            self.error_fields_dict[this_field][x, y, z] = this_error
                        if abs(this_error) > abs(self.max_error_dict[this_field]):
                            self.max_error_dict[this_field] = this_error

            # Do solver comparison if requested
            if self.compare_solvers:
                for field_s in self.fields_list:
                    this_vals_s = field_s[x, y, z]
                    if this_vals_s == 0:
                        continue
                    for field_t in self.fields_list:
                        if field_s is not field_t:
                            this_diff = (field_t[x, y, z] - this_vals_s) / this_vals_s
                            this_max_diff = self.max_diff_dict[field_s][field_t]
                            if abs(this_diff) > abs(this_max_diff):
                                self.max_diff_dict[field_s][field_t] = this_diff

        # Fulfill sample set request w/ or w/o available analytic solution
        if self.sample_set_pixels is not None:
            self.plot_win_sol.erase_all_data()

            sample_num_x = [i for i in range(len(self.sample_set_pixels))]
            if self.analytic_sol is not None:
                self.plot_win_err.erase_all_data()
                sample_num_y = {this_field: [0.0] * len(self.sample_set_pixels)
                                for this_field in [self.analytic_sol] + self.fields_list}
                error_num_y = {this_field: [0.0] * len(self.sample_set_pixels) for this_field in self.fields_list}
            else:
                sample_num_y = {this_field: [0.0] * len(self.sample_set_pixels) for this_field in self.fields_list}
                error_num_y = None

            min_err = -1
            max_err = 1
            this_analytic_sol = 0
            for sample_num in range(len(self.sample_set_pixels)):
                x, y, z = self.sample_set_pixels[sample_num]

                if self.analytic_sol is not None:
                    this_analytic_sol = self.analytic_sol[x, y, z]

                for this_field in sample_num_y.keys():
                    this_sample = this_field[x, y, z]
                    sample_num_y[this_field][sample_num] = this_sample
                    # Compare with analytic solution if available
                    if self.analytic_sol is not None and this_field is not self.analytic_sol:
                        if this_analytic_sol != 0:
                            this_err = this_sample / this_analytic_sol - 1
                            error_num_y[this_field][sample_num] = this_err
                            min_err = min([min_err, this_err])
                            max_err = max([max_err, this_err])

            for this_field in sample_num_y.keys():
                if this_field is not self.analytic_sol:
                    self.plot_win_sol.add_data_series(self.fields_dict[this_field],
                                                      sample_num_x, sample_num_y[this_field])
                    if self.analytic_sol is not None:
                        self.plot_win_err.add_data_series(self.fields_dict[this_field],
                                                          sample_num_x, error_num_y[this_field])
                elif self.analytic_sol is not None:
                    self.plot_win_sol.add_data_series("analytic_sol", sample_num_x, sample_num_y[this_field])

            if self.analytic_sol is not None:
                self.plot_win_err.pW.setYRange(min_err, max_err)

        # Print error metric if analytic solution is available
        if self.analytic_sol is not None:
            print('Error report (analytic): Step', mcs)
            for this_field in self.fields_list:
                print('\t', self.fields_dict[this_field], ': ', self.max_error_dict[this_field])

        # Solver comparison post-processing, if requested
        if self.compare_solvers:
            # Render histogram of solver differences
            self.plot_win_comp.erase_all_data()
            solver_labels = [i for i in range(len(self.max_diff_dict.keys()))]
            for field_s in self.max_diff_dict.keys():
                idx = 0
                solver_diffs = [0.0] * len(self.max_diff_dict.keys())
                for field_t in self.max_diff_dict.keys():
                    if field_s is not field_t:
                        solver_diffs[idx] = self.max_diff_dict[field_s][field_t]

                    idx += 1

                self.plot_win_comp.add_data_series(self.fields_dict[field_s], solver_labels, solver_diffs)

            # Print solver comparison
            print('Difference report: Step', mcs)
            for field_s in self.max_diff_dict.keys():
                print('\tSolver: ', self.fields_dict[field_s])
                for field_t, max_diff in self.max_diff_dict[field_s].items():
                    print('\t\tSolver {}: {}'.format(self.fields_dict[field_t], max_diff))

    def finish(self):
        pass

    def load_inits(self):
        """
        Loads specifications described in subclasses' init()
        :return:
        """
        if self.analytic_setup is not None:
            self.create_scalar_field_py("analytic_sol")

        # Check: comparison requires more than one solver
        if len(self.fields_list_names) < 2:
            self.compare_solvers = False

        # Setup for plotting error contours if requested
        if self.error_contours:
            self.error_fields_names_dict = {field_name: "err_" + field_name for field_name in self.fields_list_names}
            [self.create_scalar_field_py(err_field_name) for err_field_name in self.error_fields_names_dict.values()]
