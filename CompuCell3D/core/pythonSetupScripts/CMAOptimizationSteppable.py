from PySteppables import *
try:
    from cma import CMAEvolutionStrategy
except ImportError:
    print 'To allow CMA optimization please install cma python module - https://pypi.python.org/pypi/cma'
    raise ImportError('Could not find cma module')

from copy import deepcopy
import CompuCellSetup


# decorator that enables seamless call of the optimization housekeeping fcn
def optimization_step_decorator(step_fcn):
    def wrapper(self, *args, **kwargs):
        step_fcn(self, *args, **kwargs)
        self.optimization_step(*args, **kwargs)

    return wrapper


class CMAOptimizationSteppable(SteppableBasePy):
    def __init__(self, _simulator, _frequency=1):
        SteppableBasePy.__init__(self, _simulator, _frequency)

        self.optim = None
        self.sim_length_mcs = self.simulator.getNumSteps() - 1
        self.f_vec = []
        self.X_vec = []
        self.X_vec_check = []
        self.num_fcn_evals = -1

    def minimized_fcn(self, *args, **kwds):
        """
        this function needs to be overloaded in the subclass - it implements simulation fitness metric
        :return {float}: number describing the "fitness" of the simulation
        """
        return 0.0

    def initial_condition_fcn(self, *args, **kwds):
        """
        This function prepares initial condition for the simulaiton. Typically it creates cell field and initializes
        all cell and field properties
        :param args: first argument is a vector of parameters that are being optimized. The rest are up to the user
        :param kwds: key words arguments - those are are up to the user
        :return: None
        """
        pass

    def init_optimization_strategy(self, *args, **kwds):
        """
        init_optimization_strategy initializes optimizer object. Its argument depend on the specific initializer used
        IN the case of the CMA optimizer the options are described here: https://pypi.python.org/pypi/cma
        :param args: see https://pypi.python.org/pypi/cma
        :param kwds: see https://pypi.python.org/pypi/cma
        :return: None
        """

        self.optim = CMAEvolutionStrategy(*args, **kwds)

    def optimization_step(self, mcs):
        """
        THis function implements houklsekeeping associated with running optimization algorithm in a steppable
        :param mcs {int}: current mcs
        :return: None
        """

        if not mcs % self.sim_length_mcs:

            if self.optim.stop():
                self.stopSimulation()
                print('termination by', self.optim.stop())
                print('best f-value =', self.optim.result()[1])
                print('best solution =', self.optim.result()[0])

            if not len(self.X_vec):
                self.X_vec = self.optim.ask()

                if len(self.f_vec):
                    # print 'self.X_vec_check=', self.X_vec_check
                    # print 'self.f_vec=', self.f_vec
                    self.optim.tell(self.X_vec_check, self.f_vec)  # do all the real "update" work

                    self.optim.disp(20)  # display info every 20th iteration
                    self.optim.logger.add()  # log another "data line"

                    self.f_vec = []

                    self.num_fcn_evals = len(self.X_vec)

                self.X_vec_check = deepcopy(self.X_vec)

            self.X_current = self.X_vec[0]

            if len(self.X_vec_check) != self.num_fcn_evals:
                fcn_target = self.minimized_fcn()
                self.f_vec.append(fcn_target)
                self.X_vec.pop(0)

            self.num_fcn_evals -= 1

            CompuCellSetup.reset_current_step(0)
            self.simulator.setStep(0)
            self.clean_cell_field(reset_inventory=True)
            self.initial_condition_fcn(self.X_current)
