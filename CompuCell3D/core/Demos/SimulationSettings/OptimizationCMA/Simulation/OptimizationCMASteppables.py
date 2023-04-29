# from PySteppables import *
from CMAOptimizationSteppable import *
import CompuCell
import sys
from random import randint, random


class OptimizationCMASteppable(CMAOptimizationSteppable):
    def __init__(self, _simulator, _frequency=1):
        CMAOptimizationSteppable.__init__(self, _simulator, _frequency)


        # those variables are used by the prepare_initial_condition function
        self.cell_type_list = []
        self.cell_type_list_initialized = False

        # init_optimization_strategy takes same arguments as CMAEvolutionStrategy class from cma module
        # first argument is a starting parameter vector from which optimization begins
        # second argument determines standard deviation of parameters vector
        # third argument (a dictionary) defines optimizer options - here we limit parameter range to be between 0 and 15
        # NOTICE that we use same rang efor both parameters. If you want to have different parameters have different range
        # you will need to rescale parameters inside initial_condition_fcn - here see implementation of
        # self.prepare_initial_condition where we rescale second optimized parameter (temperature)

        self.init_optimization_strategy([1.0, 1.0], 10, {'bounds': [0, 15]})


    def set_xml_params(self, param_vec):
        """
        Convenience function used to change xml entries while the simulation is running
        For more information see Demos/SteeringVolumeFlex demo and check out
        Python Scripting Manual v 3.7.5 chapter "Steering - changing xml parameters on the fly" page 36
        :param param_vec: vector of parameters
        :return: None
        """

        contact_ab = param_vec[0]
        temperature = param_vec[1]

        access_path = [['Plugin', 'Name', 'Contact'], ['Energy', 'Type1', 'A', 'Type2', 'B']]
        self.setXMLElementValue(contact_ab, *access_path)

        temperature_access_path = [['Potts'], ['Temperature']]
        self.setXMLElementValue(temperature, *temperature_access_path)

        self.updateXML()

    def initial_condition_fcn(self, *args, **kwds):

        param_vec = args[0]  # first argument is a vector of parameters that are being optimized

        """
        THis fcn creates initial cell layoput - ideally it should be the same for each evaluation of mimized fcn.
        It takes vector of optimized parameters and applies them to the running simulation - here we are manipulating
        xml parameters only. Changing Python parameters is much easier so I decided to present the ard part instead

        :param param_vec: vector of optimized parameters that should be used to evaluate minimized function
        :return: None
        """

        # if parameter vector is None - we do no manipulate xml or Python parameters.
        # This may happen at the beginning of the optimization run
        if param_vec is not None:
            # self.set_xml_params(param_vec[0],3*param_vec[1])  # allowing temperature to vary between 0 and 45
            self.set_xml_params([param_vec[0], 3 * param_vec[1]])  # allowing temperature to vary between 0 and 45

        # convenitnce runtion that rounds floating point number and converts it to integer
        ir = lambda x: int(round(x))

        # We populate rectangular regiong with 5x5 cells . Cell types are chosen randomly.
        # NOTICE: after we create initial cell layout we store the information necessary to recreate it again
        # in the subsequent optimization steps - see how we use
        #  self.cell_type_list and self.cell_type_list_initialized variables

        x_min, x_max = ir(0.2 * self.dim.x), ir(0.8 * self.dim.y)
        y_min, y_max = ir(0.2 * self.dim.y), ir(0.8 * self.dim.y)
        cell_size = 5

        types = [self.A, self.B]

        cell_counter = 0
        for x in xrange(x_min, x_max, cell_size):
            for y in xrange(y_min, y_max, cell_size):

                if not self.cell_type_list_initialized:
                    new_cell_type = types[randint(0, len(types) - 1)]
                    self.cell_type_list.append(new_cell_type)
                else:
                    new_cell_type = self.cell_type_list[cell_counter]
                    cell_counter += 1

                new_cell = self.newCell(new_cell_type)

                self.cellField[x:x + cell_size - 1, y:y + cell_size - 1, 0] = new_cell

        self.cell_type_list_initialized = True

        for cell in self.cellList:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0

    def minimized_fcn(self, *args, **kwds):
        """
        Simulation fitness fcn - here we use total boundary length betwqeen two cell types
        :param args: not used with eh CMA - placeholder for future optimization algorithms
        :param kwds: not used with eh CMA - placeholder for future optimization algorithms
        :return: number describing simulation fitness
        """

        boundary_mat = self.compute_intertype_boundary()
        print boundary_mat
        fcn_target = boundary_mat[1, 2]
        return fcn_target


    def compute_intertype_boundary(self):
        """
        Computes a matrix of intertype boundary lengths. When we have 3 cell types (Medium A, B) the matrix is 3x3
        Each entry of the matrix determines the total length between cell types of different types.
        Notice that the matrix is symmetric. If we want total boundary length betwen types A and B we  use element (1,2)
        If we wanted total length between cells of type A and Medium we would use element (0,1) and if we want
        total length between cells of type A and cellccells ofg type A we would use element (1,1)
        NOTISE: we need to set manually max cell type id - since we have 3 types max type id is 2
        type numbering starts at 0 - see XML
        :return: matrix of intertype boundary lengths
        """

        max_type_id = 2
        intertype_boundary = np.zeros((max_type_id + 1, max_type_id + 1), dtype=float)

        for cell in self.cellList:
            for neighbor, commonSurfaceArea in self.getCellNeighborDataList(cell):
                if neighbor:
                    intertype_boundary[cell.type, neighbor.type] += commonSurfaceArea
                else:
                    intertype_boundary[cell.type, 0] += commonSurfaceArea

        # correcting double-counting for homotypic boundaries
        for i in xrange(max_type_id + 1):
            intertype_boundary[i, i] /= 2.0

            # setting total area betwwen Medium and non-Medium
            # be the same as between non-Medium and Medium
            # symetrizing contact area matrix
            intertype_boundary[0, i] = intertype_boundary[i, 0]

        return intertype_boundary


    # NOTE: using optimization_step_decorator is essential if you want optimization to happen
    # optimization_step_decorator internally calls housekeeping function that performs optimization and
    # manages the flow of the simulation . Among other things it allows continuous run of the simulation
    @optimization_step_decorator
    def step(self, mcs):
        # PUT YOUR CODE HERE
        print 'optimizer will run after mcs=', mcs

