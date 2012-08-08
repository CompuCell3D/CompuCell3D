     #Steppables

"This module contains examples of certain more and less useful steppables written in Python"
from CompuCell import NeighborFinderParams
import CompuCell
from random import random
from random import randint
import types
from PySteppables import *
import CompuCell
import sys
import math

class ElasticityLocalSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
        self.linksInitialized=False
        
            
    def initializeElasticityLocal(self):
        
        for cell in self.cellList:
            
            elasticityDataList=self.getElasticityDataList(cell)
            for elasticityData in elasticityDataList: # visiting all elastic links of 'cell'

                targetLength=elasticityData.targetLength               
                elasticityData.targetLength=6.0
                elasticityData.lambdaLength=200.0
                elasticityNeighbor=elasticityData.neighborAddress
                
                # now we set up elastic link data stored in neighboring cell
                neighborElasticityData=None
                neighborElasticityDataList=self.getElasticityDataList(elasticityNeighbor)
                for neighborElasticityDataTmp in neighborElasticityDataList:
                    if not CompuCell.areCellsDifferent(neighborElasticityDataTmp.neighborAddress,cell):
                        neighborElasticityData=neighborElasticityDataTmp
                        break
                
                if neighborElasticityData is None:
                    print "None Type returned. Problems with FemDataNeighbors initialization or sets of elasticityNeighborData are corrupted"
                    sys.exit()
                neighborElasticityData.targetLength=6.0
                neighborElasticityData.lambdaLength=200.0
        

        
    def step(self,mcs):
        if not self.linksInitialized:        
            self.initializeElasticityLocal()
            # adding link between cell.id=1 and cell.id=3
            cell1=None
            cell3=None
            for cell in self.cellList:
                if cell.id==1:
                    cell1=cell
                if cell.id==3:
                    cell3=cell
            self.elasticityTrackerPlugin.addNewElasticLink(cell1,cell3,200.0, 6.0)
        
	 

