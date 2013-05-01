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
        print    
        for compartments in self.clusters:
#             print "cluster has size=",len(compartments)
#             print ' ------ compartmentList=',compartments.clusterId()
#             print 'compartments=',compartments
            for cell in compartments:                
#                 print 'cell.id=',cell.id                
                self.clusterSurfacePlugin.setTargetAndLambdaClusterSurface(cell,80, 2.0);
                break
            
class MitosisSteppableClusters(MitosisSteppableClustersBase):
    def __init__(self,_simulator,_frequency=1):
        MitosisSteppableClustersBase.__init__(self,_simulator, _frequency)           
                
    def step(self,mcs):        
        
        for cell in self.cellList:            
            clusterCellList=self.getClusterCells(cell.clusterId)
            print "DISPLAYING CELL IDS OF CLUSTER ",cell.clusterId,"CELL. ID=",cell.id
            for cellLocal in clusterCellList:
                print "CLUSTER CELL ID=",cellLocal.id," type=",cellLocal.type
                print 'clusterSurface=',cellLocal.clusterSurface
                    
        for compartments in self.clusters:
            clusterId=-1
            clusterCell=None
            clusterSurface=0.0
            for cell in compartments:
                clusterCell=cell
                clusterId=cell.clusterId
                for pixelTrackerData in self.getCellPixelList(cell):
                    for neighbor in self.getPixelNeighborsBasedOnNeighborOrder(_pixel=pixelTrackerData.pixel,_neighborOrder=1):
                        nCell = self.cellField.get(neighbor.pt)
                        if not nCell: # only medium contributes in this case
                            clusterSurface+=1.0
                        elif cell.clusterId != nCell.clusterId:
                            clusterSurface+=1.0
                            
            print 'MANUAL CALCULATION clusterId=',clusterId,' clusterSurface=',clusterSurface
            print 'AUTOMATIC UPDATE clusterId=',clusterId, ' clusterSurface=',clusterCell.clusterSurface
            
        
        if mcs==400:
            cell1=None
            for cell in self.cellList:
                cell1=cell
                break
            self.reassignClusterId(cell1,2)
        
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
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
                
        compartmentListChild=self.inventory.getClusterCells(childCell.clusterId)
        compartmentListParent=self.inventory.getClusterCells(parentCell.clusterId)
        print "compartmentListChild=",compartmentListChild 
        for i in xrange(compartmentListChild.size()):
            compartmentListParent[i].targetVolume/=2.0
            # compartmentListParent[i].targetVolume=25
            compartmentListChild[i].targetVolume=compartmentListParent[i].targetVolume
            compartmentListChild[i].lambdaVolume=compartmentListParent[i].lambdaVolume


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
         
