from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCell

import math


def analytic_lambda_factory_0(_d, _l, _n_terms, _dir: str, _resl, _resu, _resi):
    """
    Generates lambda expression for the transient solution to the diffusion equation with Dirichlet conditions 0 on
    the lower coordinate boundary and 1 on the coordinate upper boundary, and Neumann 0 conditions everywhere else
    :param _d: diffusion coefficient
    :param _l: length of domain
    :param _n_terms: number of summation terms
    :param _dir: direction label of the coordinate along which diffusion occurs
    :param _resl: value at lower boundary (e.g., x=0)
    :param _resu: value at upper boundary (e.g., x=L)
    :param _resi: value of initial homogoeneous solution
    :return: lambda expression
    """
    assert _dir in ['x', 'y', 'z']

    _lf = float(_l)
    _dir_map = {'x': 0, 'y': 1, 'z': 2}

    def _analytic_k(_x, _y, _z, _t, _k):
        _xf, _yf, _zf, _tf, _kf = float(_x), float(_y), float(_z), float(_t), float(_k)
        _cf = [_xf, _yf, _zf][_dir_map[_dir]]
        if _k == 0:
            return _resl + (_resu - _resl) * (_cf + 1.0) / _l
        else:
            _kpi = _k * math.pi
            _e = _kpi / _lf
            _a = 2.0 * ((-1.0) ** _k * (_resu - _resi) + _resi - _resl) / _kpi
            return _a * math.exp(-_d * _tf * _e ** 2.0) * math.sin(_e * (_cf + 1.0))

    def _analytic_lambda(_x, _y, _z, _t):
        if _t == 0:
            return _resi

        res = 0.0
        for k in range(0, _n_terms):
            res += _analytic_k(_x, _y, _z, _t, k)
        return res

    return lambda x, y, z, t: _analytic_lambda(x, y, z, t)


def analytic_lambda_factory_1(_d, _l, _dir: str, _m, _a, _n):
    """
    Generates lambda expression for the transient solution to the diffusion equation with periodic conditions along
    one direction, and Neumann 0 conditions everywhere else
    :param _d: diffusion coefficient
    :param _l: length of domain
    :param _dir: direction label of the coordinate along which diffusion occurs
    :param _m: mean value of initial sine wave
    :param _a: amplitude of initial sine wave
    :param _n: wave number of initial sine wave
    :return: lambda expression
    """
    assert _dir in ['x', 'y', 'z']

    _lf = float(_l)
    _dir_map = {'x': 0, 'y': 1, 'z': 2}

    def _analytic_lambda(_x, _y, _z, _t):
        _p = [_x, _y, _z][_dir_map[_dir]]
        _e = 2 * math.pi * _n / _lf
        return _m + _a * math.exp(-_d * _t * _e ** 2.0) * math.sin(_e * _p)

    return lambda x, y, z, t: _analytic_lambda(x, y, z, t)


def lambda_analytic_factory_2(_d, _l, _dir: str, _m, _a, _n):
    """
    Generates lambda expression for the transient solution to the diffusion equation with Neumann conditions along
    all directions
    :param _d: diffusion coefficient
    :param _l: length of domain
    :param _dir: direction label of the coordinate along which diffusion occurs
    :param _m: mean value of initial cosine wave
    :param _a: amplitude of initial cosine wave
    :param _n: wave number of initial cosine wave
    :return: lambda expression
    """
    assert _dir in ['x', 'y', 'z']

    _lf = float(_l)
    _dir_map = {'x': 0, 'y': 1, 'z': 2}

    def _analytic_lambda(_x, _y, _z, _t):
        _p = [_x, _y, _z][_dir_map[_dir]]
        _e = 2 * math.pi * _n / _lf
        return _m + _a * math.exp(- _d * _e ** 2.0 * _t) * math.cos(_e * _p)

    return lambda x, y, z, t: _analytic_lambda(x, y, z, t)


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
        Some built-in solutions require a diffusion coefficient; declare it in self.analytic_diff_coeff
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
        self.analytic_diff_coeff = None

        self.error_contours = False
        self.compare_solvers = True

        # Specify a sample set here
        self.sample_set_pixels = None

        self.sample_set_values = None

        self.analytic_sol = None

        # Coefficients used in analytic lambda generators
        self.analytic_coeffs = None

        self.plot_win_comp = None
        self.plot_win_comp_hist = None
        self.plot_win_sol = None
        self.plot_win_err = None

        self.error_fields_names_dict = {}

        self.fields_list = []
        self.fields_dict = {}
        self.error_fields_dict = {}

        self.max_error_dict = {}
        self.max_diff_dict = {}
        self.max_diff_hist = {}
        self.max_diff_legend_str = {}

    def start(self):
        """
        Loads specifications described in subclasses' start()
        :return: None
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
            self.__lambda_generator()
            if self.analytic_setup != 'manual':
                for x, y, z in self.every_pixel():
                    self.analytic_sol[x, y, z] = self.analytic_lambda(x, y, z, 0)

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

                if self.analytic_setup is not None:
                    self.max_diff_dict[field_s][self.analytic_sol] = 0.0

            # For plotting differences at a particular step
            self.plot_win_comp = self.add_new_plot_window(
                title='Solver differences',
                x_axis_title='Solver label',
                y_axis_title='Max. relative difference',
                x_scale_type='Linear',
                y_scale_type='Linear',
                grid=True,
                config_options={'legend': True}
            )

            # For plotting history of solver differences
            self.plot_win_comp_hist = self.add_new_plot_window(
                title='Solver differences history',
                x_axis_title='Step',
                y_axis_title='Max. relative difference',
                x_scale_type='Linear',
                y_scale_type='log',
                grid=True,
                config_options={'legend': True}
            )

            idx = 0
            for field, field_name in self.fields_dict.items():
                self.plot_win_comp.add_plot(plot_name=field_name, style='Steps', color=color_list[idx], alpha=50)

                self.max_diff_legend_str[field] = {}
                for field_t, field_name_t in self.fields_dict.items():
                    if field is not field_t:
                        field_comp_str = field_name + ' - ' + field_name_t
                        self.max_diff_legend_str[field][field_t] = field_comp_str
                        self.plot_win_comp_hist.add_plot(plot_name=field_comp_str, style='Dots')

                    if self.analytic_setup is not None:
                        field_comp_str = field_name + ' - analytic'
                        self.max_diff_legend_str[field][self.analytic_sol] = field_comp_str
                        self.plot_win_comp_hist.add_plot(plot_name=field_comp_str, style='Dots')

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
        """
        Performs requested analysis and reporting
        :param mcs: current Monte Carlo step
        :return: None
        """
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
                # Update analytic solution
                if self.analytic_setup != 'manual':
                    self.analytic_sol[x, y, z] = self.analytic_lambda(x, y, z, mcs)

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
                for field_s in self.max_diff_dict.keys():
                    this_vals_s = field_s[x, y, z]
                    if this_vals_s == 0:
                        continue
                    for field_t in self.max_diff_dict[field_s].keys():
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

            # Store and render difference history
            self.max_diff_hist[mcs] = self.max_diff_dict
            for field_s in self.max_diff_legend_str.keys():
                for field_t, comp_str in self.max_diff_legend_str[field_s].items():
                    this_max_diff = abs(self.max_diff_dict[field_s][field_t])
                    if this_max_diff > 0:
                        self.plot_win_comp_hist.add_data_point(comp_str, mcs, this_max_diff)

            # Print solver comparison
            print('Difference report: Step', mcs)
            for field_s in self.max_diff_dict.keys():
                print('\tSolver: ', self.fields_dict[field_s])
                for field_t, max_diff in self.max_diff_dict[field_s].items():
                    if field_t is self.analytic_sol:
                        print('\t\tAnalytic: {}'.format(max_diff))
                    else:
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

    def __lambda_generator(self):
        if self.analytic_setup not in ['manual', 'lambda']:
            self.analytic_lambda = self.lambda_generator(self.analytic_setup)

    def lambda_generator(self, _model_str: str):
        if _model_str == 'line_x':
            # 0 on -x bounadry, 1 on +x boundary, steady-state
            return lambda _x, _y, _z, _t: (_x + 1.0) / (self.dim.x + 1.0)
        elif _model_str == 'line_y':
            # 0 on -y bounadry, +y on top boundary, steady-state
            return lambda _x, _y, _z, _t: (_y + 1.0) / (self.dim.y + 1.0)
        elif _model_str == 'line_z':
            # 0 on -z bounadry, +z on top boundary, steady-state
            return lambda _x, _y, _z, _t: (_z + 1.0) / (self.dim.z + 1.0)
        elif _model_str == 'line_x_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_0(self.analytic_diff_coeff, self.dim.x + 1, 20, 'x',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'line_y_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_0(self.analytic_diff_coeff, self.dim.y + 1, 20, 'y',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'line_z_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_0(self.analytic_diff_coeff, self.dim.z + 1, 20, 'z',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'sine_x_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_1(self.analytic_diff_coeff, self.dim.x, 'x',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'sine_y_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_1(self.analytic_diff_coeff, self.dim.y, 'y',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'sine_z_t':
            assert self.analytic_diff_coeff is not None
            return analytic_lambda_factory_1(self.analytic_diff_coeff, self.dim.z, 'z',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'flat_end_x_t':
            assert self.analytic_diff_coeff is not None
            return lambda_analytic_factory_2(self.analytic_diff_coeff, self.dim.x - 1, 'x',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'flat_end_y_t':
            assert self.analytic_diff_coeff is not None
            return lambda_analytic_factory_2(self.analytic_diff_coeff, self.dim.y - 1, 'y',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'flat_end_z_t':
            assert self.analytic_diff_coeff is not None
            return lambda_analytic_factory_2(self.analytic_diff_coeff, self.dim.z - 1, 'z',
                                             self.analytic_coeffs[0], self.analytic_coeffs[1], self.analytic_coeffs[2])
        elif _model_str == 'manual':  # Use this option to specify manually
            return None
        elif _model_str == 'lambda':  # Use this option to specify with a lambda expression
            return None
        else:
            return lambda _x, _y, _z, _t: 0.0

    def load_transient_line_coeffs(self, _dir: str, _diff_coeff, _lower_val, _upper_val, _initial_val=0.0):
        """
        Convenience function to use a transient line analytic solution
        Boundary conditions along direction of diffusive transport are Dirichlet
        :param _dir: string specifying direction along which steady-state line will form
        :param _diff_coeff: diffusion coefficient
        :param _initial_val: value of initially homogeneous solution (default 0.0)
        :param _lower_val: boundary value of minimum coordinate position along line
        :param _upper_val: boundary value of maximum coordinate position along line
        :return: None
        """

        self.analytic_diff_coeff = _diff_coeff
        self.analytic_setup = {'x': 'line_x_t', 'y': 'line_y_t', 'z': 'line_z_t'}[_dir]
        self.analytic_coeffs = [_lower_val, _upper_val, _initial_val]

    def load_transient_sine_coeffs(self, _dir: str, _diff_coeff, _mean_val, _ampl_val, _wave_num: int):
        """
        Convenience function to test periodic conditions with an initial sine wave distribution
        :param _dir: string specifying direction along which the initial sine wave distribution is specified
        :param _diff_coeff: diffusion coefficient
        :param _mean_val: mean value of initial sine wave (must be greater than zero)
        :param _ampl_val: amplitude of initial sine wave (must be less than mean value and greater than zero)
        :param _wave_num: wave number of initial distribution
        :return: None
        """
        assert 0 < _ampl_val < _mean_val
        assert _wave_num > 0
        self.analytic_diff_coeff = _diff_coeff
        self.analytic_setup = {'x': 'sine_x_t', 'y': 'sine_y_t', 'z': 'sine_z_t'}[_dir]
        self.analytic_coeffs = [_mean_val, _ampl_val, _wave_num]

    def load_flat_line_coeffs(self, _dir: str, _diff_coeff, _mean_val, _ampl_val, _wave_num: int):
        """
        Convenience function to test Neumann conditions with an initial cosine wave distribution
        :param _dir: string specifying direction along which the initial cosine wave distribution is specified
        :param _diff_coeff: diffusion coefficient
        :param _mean_val: mean value of initial cosine wave (must be greater than zero)
        :param _ampl_val: amplitude of initial cosine wave (must be less than mean value and greater than zero)
        :param _wave_num: wave number of initial distribution
        :return: None
        """
        assert 0 < _ampl_val < _mean_val
        assert _wave_num > 0
        self.analytic_diff_coeff = _diff_coeff
        self.analytic_setup = {'x': 'flat_end_x_t', 'y': 'flat_end_y_t', 'z': 'flat_end_z_t'}[_dir]
        self.analytic_coeffs = [_mean_val, _ampl_val, _wave_num]
