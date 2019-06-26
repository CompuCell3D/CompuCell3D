from cc3d.core.PySteppables import *


class AdhesionMoleculesSteppables(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
        if mcs == 0:
            for cell in self.cell_list:
                print("CELL ID=", cell.id, " CELL TYPE=", cell.type)

                # accessing entire vector of adhesion molecule densities for non-medium cell
                adhesion_molecule_vector = self.adhesionFlexPlugin.getAdhesionMoleculeDensityVector(cell)
                print("adhesion_molecule_vector=", adhesion_molecule_vector)

            # Medium density adhesion vector
            # accessing entire vector of adhesion molecule densities for medium cell
            medium_adhesion_molecule_vector = self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
            print("medium_adhesion_molecule_vector=", medium_adhesion_molecule_vector)
        else:
            for cell in self.cell_list:
                print("CELL ID=", cell.id, " CELL TYPE=", cell.type)
                if cell.type == 1:
                    # accessing adhesion molecule density using its name
                    print("NCad=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell, "NCad"))

                    # accessing adhesion molecule density using its index - molecules are indexed
                    # in the same order they are listed in the xml file
                    print("Int=", self.adhesionFlexPlugin.getAdhesionMoleculeDensityByIndex(cell, 1))

                    # One can use either setAdhesionMoleculeDensityVector or assignNewAdhesionMoleculeDensityVector
                    # the difference is that setAdhesionMoleculeDensityVector will check if
                    # the new vector has same size as existing one and this is not a good option when initializing
                    # childCell after mitosis
                    # assignNewAdhesionMoleculeDensityVector simply assigns vector
                    # and does not do any checks.
                    # It is potentially error prone but also is a good option to initialize child cell after mitosis

                    # setting entire vector of adhesion molecule densities for non-medium cell
                    # self.adhesionFlexPlugin.setAdhesionMoleculeDensityVector(cell,[3.4,2.1,12.1])

                    # # setting entire vector of adhesion molecule densities for non-medium cell
                    self.adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(cell, [3.4, 2.1, 12.1])

                    print("NEW VALUE OF INT ", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell, "Int"))
                if cell.type == 2:
                    print("NCam=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell, "NCam"))

                    # setting adhesion molecule density using its name
                    self.adhesionFlexPlugin.setAdhesionMoleculeDensity(cell, "NCad", 11.2)

                    print("NEW VALUE OF NCad=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell, "NCad"))

                    # setting adhesion molecule density using its index - molecules
                    # are indexed in the sdame order they are listed in the xml file
                    self.adhesionFlexPlugin.setAdhesionMoleculeDensityByIndex(cell, 2, 11.1)
                    print("NEW VALUE OF Int=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell, "Int"))

            # Medium density adhesion vector
            # One can use either setMediumAdhesionMoleculeDensityVector or assignNewMediumAdhesionMoleculeDensityVector
            # the difference is that setMediumAdhesionMoleculeDensityVector will check if the new vector
            # has same size as existing one and this is not a good option when initializing childCell after mitosis
            # setting entire vector of adhesion molecule densities for medium cell
            # self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensityVector([1.4,3.1,18.1])

            # setting entire vector of adhesion molecule densities for medium cell
            self.adhesionFlexPlugin.assignNewMediumAdhesionMoleculeDensityVector([1.4, 3.1, 18.1])

            # accessing entire vector of adhesion molecule densities for medium cell
            medium_adhesion_molecule_vector = self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
            print("medium_adhesion_molecule_vector=", medium_adhesion_molecule_vector)

            # setting adhesion molecule density using its name - medium cell
            self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensity("NCam", 2.8)

            # accessing entire vector of adhesion molecule densities for medium cell
            medium_adhesion_molecule_vector = self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
            print("medium_adhesion_molecule_vector=", medium_adhesion_molecule_vector)

            self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensityByIndex(2, 16.8)

            # accessing entire vector of adhesion molecule densities for medium cell
            medium_adhesion_molecule_vector = self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector()
            print("medium_adhesion_molecule_vector=", medium_adhesion_molecule_vector)
