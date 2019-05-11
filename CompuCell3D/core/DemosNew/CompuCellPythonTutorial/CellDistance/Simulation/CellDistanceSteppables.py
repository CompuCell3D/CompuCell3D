from cc3d.core.PySteppables import *


class CellDistanceSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.cellA = None
        self.cellB = None

    def start(self):
        self.cellA = self.potts.createCell()
        self.cellA.type = self.A
        self.cell_field[10:12, 10:12, 0] = self.cellA

        self.cellB = self.potts.createCell()
        self.cellB.type = self.B
        self.cell_field[92:94, 10:12, 0] = self.cellB

    def step(self, mcs):
        dist_vec = self.invariant_distance_vector_integer(p1=[10, 10, 0], p2=[92, 12, 0])

        print('dist_vec=', dist_vec, ' norm=', self.vector_norm(dist_vec))

        dist_vec = self.invariant_distance_vector(p1=[10, 10, 0], p2=[92.3, 12.1, 0])
        print('dist_vec=', dist_vec, ' norm=', self.vector_norm(dist_vec))

        print('distance invariant=', self.invariant_distance(p1=[10, 10, 0], p2=[92.3, 12.1, 0]))

        print('distance =', self.distance(p1=[10, 10, 0], p2=[92.3, 12.1, 0]))

        print('distance vector between cells =', self.distance_vector_between_cells(self.cellA, self.cellB))
        print('invariant distance vector between cells =',
              self.invariant_distance_vector_between_cells(self.cellA, self.cellB))
        print('distanceBetweenCells = ', self.distance_between_cells(self.cellA, self.cellB))
        print('invariantDistanceBetweenCells = ', self.invariant_distance_between_cells(self.cellA, self.cellB))
