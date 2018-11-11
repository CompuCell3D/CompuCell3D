from PySteppables import *
import CompuCell
import sys



class AdhesionMoleculesSteppables(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator, _frequency)
        
    def start(self):
        pass

    def step(self,mcs):
        if mcs==0:
            for cell in self.cellList:
                print "CELL ID=",cell.id, " CELL TYPE=",cell.type
                adhesionMoleculeVector=self.adhesionFlexPlugin.getAdhesionMoleculeDensityVector(cell) # accessing entire vector of adhesion molecule densities for non-medium cell
                print "adhesionMoleculeVector=",adhesionMoleculeVector
                
            # Medium density adhesion vector
            mediumAdhesionMoleculeVector=self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector() # accessing entire vector of adhesion molecule densities for medium cell
            print "mediumAdhesionMoleculeVector=",mediumAdhesionMoleculeVector
        else:
            for cell in self.cellList:
                print "CELL ID=",cell.id, " CELL TYPE=",cell.type
                if cell.type==1:
                    print "NCad=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"NCad") # accessing adhesion molecule density using its name
                    print "Int=", self.adhesionFlexPlugin.getAdhesionMoleculeDensityByIndex(cell,1) # accessing adhesion molecule density using its index - molecules are indexed in the sdame order they are listed in the xml file                   
                    
                    # One can use either setAdhesionMoleculeDensityVector or assignNewAdhesionMoleculeDensityVector
                    # the difference is that setAdhesionMoleculeDensityVector will check if the new vector has same size as existing one and this is not a good option when initializing childCell after mitosis
                    # assignNewAdhesionMoleculeDensityVector simply assigns vector and does not do any checks. It is potentially error prone but also is a good option to initialize child cell after mitosis
                    
                    # self.adhesionFlexPlugin.setAdhesionMoleculeDensityVector(cell,[3.4,2.1,12.1]) # setting entire vector of adhesion molecule densities for non-medium cell
                    
                    self.adhesionFlexPlugin.assignNewAdhesionMoleculeDensityVector(cell,[3.4,2.1,12.1]) # setting entire vector of adhesion molecule densities for non-medium cell
                    
                    print "NEW VALUE OF INT ",self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"Int")
                    
                if cell.type==2:
                    print "NCam=", self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"NCam")
                    self.adhesionFlexPlugin.setAdhesionMoleculeDensity(cell,"NCad",11.2) # setting adhesion molecule density using its name
                    print "NEW VALUE OF NCad=",self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"NCad") # 
                    self.adhesionFlexPlugin.setAdhesionMoleculeDensityByIndex(cell,2,11.1) # setting adhesion molecule density using its index - molecules are indexed in the sdame order they are listed in the xml file
                    print "NEW VALUE OF Int=",self.adhesionFlexPlugin.getAdhesionMoleculeDensity(cell,"Int")
                
            # Medium density adhesion vector
            # One can use either setMediumAdhesionMoleculeDensityVector or assignNewMediumAdhesionMoleculeDensityVector
            # the difference is that setMediumAdhesionMoleculeDensityVector will check if the new vector has same size as existing one and this is not a good option when initializing childCell after mitosis      
            # self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensityVector([1.4,3.1,18.1]) # setting entire vector of adhesion molecule densities for medium cell
            self.adhesionFlexPlugin.assignNewMediumAdhesionMoleculeDensityVector([1.4,3.1,18.1]) # setting entire vector of adhesion molecule densities for medium cell
            
            mediumAdhesionMoleculeVector=self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector() # accessing entire vector of adhesion molecule densities for medium cell
            print "mediumAdhesionMoleculeVector=",mediumAdhesionMoleculeVector
            
            self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensity("NCam",2.8) # setting adhesion molecule density using its name - medium cell         
            mediumAdhesionMoleculeVector=self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector() # accessing entire vector of adhesion molecule densities for medium cell
            print "mediumAdhesionMoleculeVector=",mediumAdhesionMoleculeVector
            self.adhesionFlexPlugin.setMediumAdhesionMoleculeDensityByIndex(2,16.8) # setting adhesion molecule density using its index - medium cell  - molecules are indexed in the sdame order they are listed in the xml file      
            
            mediumAdhesionMoleculeVector=self.adhesionFlexPlugin.getMediumAdhesionMoleculeDensityVector() # accessing entire vector of adhesion molecule densities for medium cell
            print "mediumAdhesionMoleculeVector=",mediumAdhesionMoleculeVector
    