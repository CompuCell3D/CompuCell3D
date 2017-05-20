
from PySteppables import *
import CompuCell
import CompuCellSetup
import sys
class optimization_demoSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):

        for cell in self.cellList:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0
                
    def step(self,mcs):        
        pass

    def fcn(self, x):
        return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    # def finish(self):
    #
    #     # Finish Function gets called after the last MCS
    #     # CompuCellSetup.set_simulation_return_value(322.0)
    #
    #
    #     temperature_access_path = [['Potts'], ['Temperature']]
    #     temp = float(self.getXMLElementValue(*temperature_access_path))
    #
    #     access_path = [['Plugin', 'Name', 'Contact'], ['Energy', 'Type1', 'A','Type2','B']]
    #     contact_energy = float(self.getXMLElementValue(*access_path))
    #     print 'temp,contact_energy=',(temp, contact_energy)
    #
    #     x = [temp, contact_energy]
    #
    #     target_val = self.fcn(x)
    #
    #     CompuCellSetup.set_simulation_return_value(target_val)

    def finish(self):

        # Finish Function gets called after the last MCS
        # CompuCellSetup.set_simulation_return_value(322.0)


        temperature_access_path = [['Potts'], ['Temperature']]
        temp = float(self.getXMLElementValue(*temperature_access_path))

        access_path = [['Plugin', 'Name', 'Contact'], ['Energy', 'Type1', 'A','Type2','B']]
        contact_energy = float(self.getXMLElementValue(*access_path))
        print 'temp,contact_energy=',(temp, contact_energy)

        x = [temp, contact_energy]

        
        target_val = self.fcn(x)
#         target_val = self.fcn(x)+3***2
        target_val = float(int(self.fcn(x)))

        CompuCellSetup.set_simulation_return_value(target_val)

    # def finish(self):
    #
    #     # Finish Function gets called after the last MCS
    #
    #     print 'Computing heterotypic_boundary_length '
    #     import time
    #     t0 = time.time()
    #     print 't0=',t0
    #
    #     try:
    #         heterotypic_boundary_length = self.minimized_fcn()
    #     except StandardError as e:
    #         print e
    #
    #     print 'Took %f to compute heterotypic_boundary_length ' % (time.time()-t0)
    #
    #     CompuCellSetup.set_simulation_return_value(heterotypic_boundary_length)
    #
    #     # CompuCellSetup.broadcast_simulation_return_value(322.0, tag='heterotypic_boundary_length')

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
        intertype_boundary = np.zeros((max_type_id + 1, max_type_id + 1), dtype=np.float)

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
        