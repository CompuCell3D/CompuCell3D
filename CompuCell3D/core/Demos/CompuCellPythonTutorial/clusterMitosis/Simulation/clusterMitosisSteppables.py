from PySteppables import *
from PySteppablesExamples import MitosisSteppableClustersBase
import CompuCell
import sys
from random import uniform
import math

class VolumeParamSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        for cell in self.cellList:
            cell.targetVolume=25
            cell.lambdaVolume=2.0
    def step(self,mcs):
        for cell in self.cellList:
            cell.targetVolume+=1

class MitosisSteppableClusters(MitosisSteppableClustersBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableClustersBase.__init__(self,_simulator, _frequency)           
        
    def step(self,mcs):        
        
        for cell in self.cellList:            
            clusterCellList=self.getClusterCells(cell.clusterId)
            print "DISPLAYING CELL IDS OF CLUSTER ",cell.clusterId,"CELL. ID=",cell.id
            for cellLocal in clusterCellList:
                print "CLUSTER CELL ID=",cellLocal.id," type=",cellLocal.type
                
                
        
        mitosisClusterIdList=[]
        for compartmentList in self.clusterList:
            # print "cluster has size=",compartmentList.size()
            clusterId=0
            clusterVolume=0            
            for cell in CompartmentList(compartmentList):
                clusterVolume+=cell.volume            
                clusterId=cell.clusterId
            
            
            if clusterVolume>250: # condition under which cluster mitosis takes place
                mitosisClusterIdList.append(clusterId) # instead of doing mitosis right away we store ids for clusters which should be divide. This avoids modifying cluster list while we iterate through it
        for clusterId in mitosisClusterIdList:
            # to change mitosis mode leave one of the below lines uncommented
            
            # self.divideClusterOrientationVectorBased(clusterId,1,0,0)             # this is a valid option
            self.divideClusterRandomOrientation(clusterId)
            # self.divideClusterAlongMajorAxis(clusterId)                                # this is a valid option
            # self.divideClusterAlongMinorAxis(clusterId)                                # this is a valid option
            

    def updateAttributes(self):
        # compartments in the parent and child clusters arel listed in the same order so attribute changes require simple iteration through compartment list  
        compartmentListParent=self.getClusterCells(self.parentCell.clusterId)

        for i in xrange(compartmentListParent.size()):
            compartmentListParent[i].targetVolume/=2.0
        self.cloneParentCluster2ChildCluster()
        

    def changeFlips(self):
        # get Potts section of XML file 
        pottsXMLData=self.simulator.getCC3DModuleData("Potts")
        # check if we were able to successfully get the section from simulator
        if pottsXMLData:            
            flip2DimRatioElement=pottsXMLData.getFirstElement("Flip2DimRatio")
            # check if the attempt was succesful
            if flip2DimRatioElement:
                flip2DimRatioElement.updateElementValue(str(0.0))
            self.simulator.updateCC3DModule(pottsXMLData)            
         
