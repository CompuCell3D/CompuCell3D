#Steppables 

"This module contains examples of certain more and less useful steppables written in Python"
from CompuCell import NeighborFinderParams


import CompuCell

class InventoryIteration:
   def __init__(self,_inventory):
      self.inventory=_inventory
   def iterate(self):
      invItr=CompuCell.STLPyIteratorCINV()
      invItr.initialize(self.inventory.getContainer())
      invItr.setToBegin()
      cell=invItr.getCurrentRef()
      while (1):
         if invItr.isEnd():
            break
         cell=invItr.getCurrentRef()
         print "CELL ID=",cell.id
         invItr.next()


   
from PySteppables import *

class MitosisSteppableBase(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        # self.simulator=_simulator
        # self.inventory=self.simulator.getPotts().getCellInventory()
        # self.cellList=CellList(self.inventory)
        # self.clusterList=ClusterList(self.inventory)
        # self.clusterInventory=self.inventory.getClusterInventory().getContainer()
        self.mitosisSteppable=CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parentCell=self.mitosisSteppable.parentCell
        self.childCell=self.mitosisSteppable.childCell  
        self.mitosisDone=False        
        
    # def getClusterCells(self,_clusterId):
        # return self.inventory.getClusterInventory().getClusterCells(_clusterId)       
        # #works too
        # # return ClusterCellList(self.inventory.getClusterInventory().getClusterCells(_clusterId))        
        
    def setParentChildPositionFlag(self,_flag):
        '''
            0 - parent child position will be randomized between mitosis event
            negative integer - parent appears on the 'left' of the child
            positive integer - parent appears on the 'right' of the child
        '''
        
        self.mitosisSteppable.setParentChildPositionFlag(int(_flag))
        
    def getParentChildPositionFlag(self,_flag):
        return self.mitosisSteppable.getParentChildPositionFlag()
        
    def updateAttributes(self):
        self.childCell.targetVolume=self.parentCell.targetVolume
        self.childCell.lambdaVolume=self.parentCell.lambdaVolume
        self.childCell.type=self.parentCell.type        
        
    def step(self,mcs):
        print "MITOSIS STEPPABLE BASE"

        
    def divideCellRandomOrientation(self, _cell):
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisRandomOrientation(_cell)       
        if self.mitosisDone:
            self.updateAttributes()
        return self.mitosisDone
        
    def divideCellOrientationVectorBased(self, _cell, _nx, _ny, _nz):        
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisOrientationVectorBased(_cell, _nx, _ny, _nz)
        if self.mitosisDone:
            self.updateAttributes()            
        return self.mitosisDone

    def divideCellAlongMajorAxis(self, _cell):        
        # orientationVectors=self.mitosisSteppable.getOrientationVectorsMitosis(_cell)
        # print "orientationVectors.semiminorVec=",(orientationVectors.semiminorVec.fX,orientationVectors.semiminorVec.fY,orientationVectors.semiminorVec.fZ)
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisAlongMajorAxis(_cell)
        if self.mitosisDone:
            self.updateAttributes()            
        return self.mitosisDone
        

    def divideCellAlongMinorAxis(self, _cell):        
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisAlongMinorAxis(_cell)
        if self.mitosisDone:
            self.updateAttributes()            
        return self.mitosisDone
    
        
            
class MitosisSteppableClustersBase(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        # self.simulator=_simulator
        # self.inventory=self.simulator.getPotts().getCellInventory()
        # self.cellList=CellList(self.inventory)
        # self.clusterList=ClusterList(self.inventory)        
        # self.clusterInventory=self.inventory.getClusterInventory().getContainer()
        self.mitosisSteppable=CompuCell.MitosisSteppable()
        self.mitosisSteppable.init(self.simulator)
        self.parentCell=self.mitosisSteppable.parentCell
        self.childCell=self.mitosisSteppable.childCell  
        self.mitosisDone=False        
        
    # def getClusterCells(self,_clusterId):
        # return self.inventory.getClusterInventory().getClusterCells(_clusterId)       
        # #works too
        # # return ClusterCellList(self.inventory.getClusterInventory().getClusterCells(_clusterId))         
        
    def updateAttributes(self):
        parentCell=self.mitosisSteppable.parentCell
        childCell=self.mitosisSteppable.childCell
                
        compartmentListChild=self.inventory.getClusterCells(childCell.clusterId)
        compartmentListParent=self.inventory.getClusterCells(parentCell.clusterId)
        # compartments in the parent and child clusters arel listed in the same order so attribute changes require simple iteration through compartment list  
        for i in xrange(compartmentListChild.size()):
            compartmentListChild[i].type=compartmentListParent[i].type
            

    def step(self,mcs):
        print "MITOSIS STEPPABLE Clusters BASE"
                
    def divideClusterRandomOrientation(self, _clusterId):
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisRandomOrientationCompartments(_clusterId)       
        if self.mitosisDone:
            self.updateAttributes()
        return self.mitosisDone            
        
    def divideClusterOrientationVectorBased(self, _clusterId, _nx, _ny, _nz):        
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisOrientationVectorBasedCompartments(_clusterId, _nx, _ny, _nz)
        if self.mitosisDone:
            self.updateAttributes()
        return self.mitosisDone    
        
    def divideClusterAlongMajorAxis(self, _clusterId):        
        # orientationVectors=self.mitosisSteppable.getOrientationVectorsMitosis(_cell)
        # print "orientationVectors.semiminorVec=",(orientationVectors.semiminorVec.fX,orientationVectors.semiminorVec.fY,orientationVectors.semiminorVec.fZ)
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisAlongMajorAxisCompartments(_clusterId)
        if self.mitosisDone:
            self.updateAttributes()            
        return self.mitosisDone
        

    def divideClusterAlongMinorAxis(self, _clusterId):        
        self.mitosisDone=self.mitosisSteppable.doDirectionalMitosisAlongMinorAxisCompartments(_clusterId)
        if self.mitosisDone:
            self.updateAttributes()            
        return self.mitosisDone
        


class TargetVolumeSteppable(SteppablePy):
   def __init__(self,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.inventory=None
      self.inc=0
   def setInitialTargetVolume(self,_tv):
      self.tv=_tv
   def init(self,_simulator):
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
      
   def setIncrement(sel,_inc):
     self.inc=_inc
 
   def start(self):
      for cell in self.cellList:
         print "CELL ID=",cell.id
         cell.targetVolume=self.tv
   
   def step(self,mcs):
      for cell in self.cellList:
         print "CELL ID=",cell.id, "targetVolume=",cell.targetVolume
         cell.targetVolume+=self.inc


import CompuCell

class ConcentrationFieldDumper(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.dim=self.simulator.getPotts().getCellFieldG().getDim()
      self.fieldNameList=[]
   def setFieldName(self,_fieldName):
      self.fieldName=_fieldName
      self.fieldNameList.append(_fieldName)
   def step(self,mcs):
      for name in self.fieldNameList:
         fileName=name+"_"+str(mcs)+".dat"
         print "Field from the list:",fileName
         self.outputField(name,fileName)
      #self.outputField(self.FieldName)
   
   def outputField(self,_fieldName,_fileName):
      field=CompuCell.getConcentrationField(self.simulator,_fieldName)
      pt=CompuCell.Point3D()
      if field:
         try:
            fileHandle=open(_fileName,"w")
         except IOError:
            print "Could not open file ", _fileName," for writing. Check if you have necessary permissions"

         print "dim.x=",self.dim.x
         for i in xrange(self.dim.x):
            for j in xrange(self.dim.y):
               for k in xrange(self.dim.z):
                  pt.x=i
                  pt.y=j
                  pt.z=k
                  fileHandle.write("%d\t%d\t%d\t%f\n"%(pt.x,pt.y,pt.z,field.get(pt)))
                  #print "concentration @ ",pt,"=",field.get(pt)

import CompuCell

class TargetVolumeDrosoSteppable(SteppablePy):
   def __init__(self,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.inventory=None
   #def __name__(self):
      #self.name
   def setInitialTargetVolume(self,_tv):
      self.tv=_tv
   def setInitialLambdaVolume(self,_lambdaVolume):
      self.lambdaVolume=_lambdaVolume
   def init(self,_simulator):
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
      
   def start(self):

      for cell in self.cellList:
         print "CELL ID=",cell.id
         cell.targetVolume=self.tv
         cell.lambdaVolume=self.lambdaVolume
   
   def step(self,mcs):
      xCM=0
      yCM=0
      yCM=0
      
      for cell in self.cellList:
         print "Cell.id=",cell.id," volume=",cell.volume
         xCM=cell.xCM/float(cell.volume)
         yCM=cell.yCM/float(cell.volume)
         zCM=cell.zCM/float(cell.volume)
         if ((xCM-100)**2+(yCM-100)**2) < 400:
            cell.targetVolume+=1

class BlobSimpleTypeInitializer(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def start(self):
      for cell in self.cellList:
         print "Blob Initializer CELL ID=",cell.id
         cell.type=1

#wendy
from random import randint
class BlobSimpleTypeInitializerRandom(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
      self.maxType=1
   def setMaxType(self,_maxType):
      if _maxType>=1:
         self.maxType=_maxType
   def start(self):
      for cell in self.cellList:
         print "Blob Initializer CELL ID=",cell.id
         cell.type=randint(1,self.maxType)
         #print "Random Type",randint(1,self.maxType)
         invItr.next()

#import PlayerPython

#class ConcentrationFillerSteppable(SteppablePy):
   #def __init__(self,_frequency=1):
      #SteppablePy.__init__(self,_frequency)
   #def init(self,_simulator):
      #self.simulator=_simulator
      #self.inventory=self.simulator.getPotts().getCellInventory()
      #self.dim=self.simulator.getPotts().getCellFieldG().getDim()
   #def setScalarField(self,_scalarField):
      #self.scalarField=_scalarField
   #def start(self):
      #conc=0.0
      #for i in xrange(self.dim.x):
         #for j in xrange(self.dim.y):
            #for k in xrange(self.dim.z):
               #conc=float(i*j)
               #PlayerPython.fillScalarValue(self.scalarField,i,j,k,conc)
   #def step(self,mcs):pass



import sys
import CompuCell
class ModifyAttribute(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def start(self):
      for cell in self.cellList:
         print " MODIFY ATTRIB CELL ID=",cell.id
         #print "ref count=",sys.getrefcount(cell.pyAttrib)
         #print "another ref count=",sys.getrefcount(cell.pyAttrib)
         #pyAttrib=cell.pyAttrib
         pyAttrib=CompuCell.getPyAttrib(cell)
         print "ref count=",sys.getrefcount(pyAttrib)
         #print " after assign ref count=",sys.getrefcount(cell.pyAttrib)
         
         print "length=",len(pyAttrib)
         pyAttrib[0:1]=[cell.id*4]
         print "Cell attrib=",pyAttrib[0]
         print "length=",len(pyAttrib)
         pyAttrib.append(14)
         print "Cell attrib=",pyAttrib[1]
         print "length=",len(pyAttrib)
         
         #accessing cell.pyAttrib directly causes ref count decrease and objects might be garbage collected prematurely.
         #always use CompuCell.getPyAttrib(cell) function call. 
         #print "length=",len(cell.pyAttrib)
         #cell.pyAttrib=4
         #list_temp=cell.pyAttrib
         #cell.pyAttrib[0]=11
         #cell.pyAttrib[0]=12
         #cell.pyAttrib.append(cell.id*5)
         #cell.pyAttrib.append(cell.id*5)
         #print "After length=",len(cell.pyAttrib)
         #print "Cell attrib=",cell.pyAttrib[0]


   def step(self,mcs):
      for cell in self.cellList:
         #pyAttrib=cell.pyAttrib
         pyAttrib=CompuCell.getPyAttrib(cell)
         pyAttrib[0]=[cell.id*mcs,cell.id*(mcs-1)]
         if not mcs % 20:
            print "CELL ID modified=",pyAttrib," true=",cell.id,
            print " ref count=",sys.getrefcount(pyAttrib),
            print " ref count=",sys.getrefcount(pyAttrib)


class ModifyDictAttribute(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def start(self):
      for cell in self.cellList:
         print " MODIFY ATTRIB CELL ID=",cell.id
         dictionary=CompuCell.getPyAttrib(cell)
         dictionary["newID"]=cell.id*2
         print "NewID=",dictionary["newID"]


   def step(self,mcs):
      for cell in self.cellList:
         if not mcs % 20:
            dictionary=CompuCell.getPyAttrib(cell)
            print "NewID=",dictionary["newID"]


class CellKiller(SteppablePy):
   def __init__(self,_simulator,_frequency=10):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)

   def step(self,mcs):
      for cell in self.cellList:
         print "Step cell.targetVolume=",cell.targetVolume
         if mcs==10:
            cell.targetVolume=0
            print "cell.targetVolume=",cell.targetVolume


import CompuCell

class ContactLocalFlexPrinter(SteppablePy):
   def __init__(self,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.inventory=None
      self.inc=0
   #def __name__(self):
      #self.name
   def init(self,_simulator):
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.ContactLocalFlexPlugin=CompuCell.getContactLocalFlexPlugin()
      self.cellList=CellList(self.inventory)
   def setIncrement(sel,_inc):
     self.inc=_inc
    
   def step(self,mcs):
      if(mcs<2):
         return
      if not (mcs %10):
         containerAccessor=self.ContactLocalFlexPlugin.getContactDataContainerAccessorPtr()
         clfdItr=CompuCell.clfdSetPyItr()
         for cell in self.cellList:
            container=containerAccessor.get(cell.extraAttribPtr)
            
            clfdItr.initialize(container.contactDataContainer)
            clfdItr.setToBegin()
            
            while not clfdItr.isEnd():
               print "is End=",clfdItr.isEnd()
               neighborCell=clfdItr.getCurrentRef().neighborAddress
               if neighborCell:
                  print "neighbor.id",neighborCell.id," contact energy=",clfdItr.getCurrentRef().J
               else:
                  print "neighbor.id=0 contact energy=",clfdItr.getCurrentRef().J
               clfdItr.next()





#ariel
from CompuCell import Point3D

class AirInjector(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.potts=self.simulator.getPotts()
      self.inventory=self.potts.getCellInventory()
      self.cellField=self.potts.getCellFieldG()
      self.cellList=CellList(self.inventory)
      self.volumeIncrement=0
   def start(self):pass

   def setInjectionPoint(self,_x,_y,_z):
      self.injectionPoint=CompuCell.Point3D(int(_x),int(_y),int(_z))

   def setVolumeIncrement(self,_increment):
      self.volumeIncrement=_increment
   def step(self,mcs):
      cell=self.cellField.get(self.injectionPoint)
      if cell:
         cell.targetVolume+=self.volumeIncrement
         print "INCREASED TARGET VOLUME TO:",cell.targetVolume



class BubbleCellRemover(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
      self.coordName='X'
      self.cutoffValue=0;
   def start(self):pass
   
   def setCutoffCoordinate(self,_coordName,_cutoffValue):
      
      if _coordName=='x' or _coordName=='X':
         self.coordName='X'
      elif _coordName=='y' or _coordName=='Y':
         self.coordName='Y'
      elif _coordName=='z' or _coordName=='Z':
         self.coordName='Z'
      else:
         print "Coordinate Name must be of x or y or z"
         return
      
      self.cutoffValue=_cutoffValue
      
   def checkIfOKToRemove(self,cell):
      if not cell: #do nothing if cell is medium
         return 0
      elif cell.volume==0:
         return 1	 
      if self.coordName=='X':
         xCM=cell.xCM/float(cell.volume)
         if xCM>=self.cutoffValue:
            return 1
         else:
            return 0
      
      if self.coordName=='Y':
         yCM=cell.yCM/float(cell.volume)
         if yCM>=self.cutoffValue:
            return 1
         else:
            return 0
      
      if self.coordName=='Z':
         zCM=cell.zCM/float(cell.volume)
         if zCM>=self.cutoffValue:
            return 1
         else:
            return 0
         
   def step(self,mcs):
      for cell in self.cellList:
         #print "Step cell.targetVolume=",cell.targetVolume
         if self.checkIfOKToRemove(cell):
            cell.targetVolume=0

from CompuCell import Point3D
from random import randint

class BubbleNucleator(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
      self.coordName='X'
      self.numNewBubbles=0
      self.initCellType=0
   def start(self):
      self.Potts=self.simulator.getPotts()
      self.dim=self.Potts.getCellFieldG().getDim()
      print "Got here"
   def setNucleationAxis(self,_coordName):
      if _coordName=='x' or _coordName=='X':
         self.coordName='X'
      elif _coordName=='y' or _coordName=='Y':
         self.coordName='Y'
      elif _coordName=='z' or _coordName=='Z':
         self.coordName='Z'
      else:
         print "Coordinate Name must be of x or y or z"
         return
   def setNumberOfNewBubbles(self,_numNewBubbles):
      if _numNewBubbles>0:
         self.numNewBubbles=int(_numNewBubbles)
   def setInitialTargetVolume(self,_initTargetVolume):
      self.initTargetVolume=_initTargetVolume
   
   def setInitialLambdaVolume(self,_initLambdaVolume):
      self.initLambdaVolume=_initLambdaVolume
      
   def setInitialCellType(self,_initCellType):
      self.initCellType=_initCellType
   
   def createNewCell(self,pt):
      print "Nucleated bubble at ",pt
      cell=self.Potts.createCellG(pt)
      cell.targetVolume=self.initTargetVolume
      cell.type=self.initCellType
      cell.lambdaVolume=self.initLambdaVolume
      
   def nucleateBubble(self):
      pt=Point3D(0,0,0)
      if self.coordName=='X':
         pt.x=randint(0,self.dim.x-1)
         pt.y=3
         self.createNewCell(pt)
      if self.coordName=='Y':
         pt.y=randint(0,self.dim.y-1)
         pt.x=3
         self.createNewCell(pt)
         
      if self.coordName=='Z':
         pt.z=randint(0,self.dim.z-1)
         self.createNewCell(pt)
         
   def step(self,mcs):
      for i in xrange(self.numNewBubbles):
         self.nucleateBubble()


class InitialTargetVolumeSteppable(SteppablePy):
   def __init__(self,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.inventory=None
   def setInitialTargetVolume(self,_tv):
      self.tv=_tv
   def init(self,_simulator):
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def start(self):
      for cell in self.cellList:
         print "CELL ID=",cell.id
         cell.targetVolume=self.tv


#Dave Larson VolumeLocalFlex
import CompuCell
class VolumeLocalFlexSteppableEye(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.volumeLocalFlexPlugin=CompuCell.getVolumeLocalFlexPlugin()
      
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def start(self):
      for cell in self.cellList:
         #print "Step cell.targetVolume=",cell.targetVolume
         #volLocalFlexData=lambdaAccessor.get(cell.extraAttribPtr)
         #initialize targer volume:and lambda         
         if cell.type==1:
            cell.lambdaVolume=50.0
            cell.targetVolume=370
	    print "cell.lambdaVolume",cell.lambdaVolume 
         elif cell.type==2:
            cell.lambdaVolume=2.0
            cell.targetVolume=100
         else:
            cell.lambdaVolume=0
            cell.targetVolume=0
         
   def step(self,mcs):
      for cell in self.cellList:
         if cell.type==1 and cell.targetVolume<770:
            cell.targetVolume+=10

import CompuCell
class NeighborInducedKiller(SteppablePy):
   def __init__(self,_simulator,_frequency=10):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin()
      
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)

   def step(self,mcs):
      for cell in self.cellList:
         
         
         killFlag=1
         if cell.type != 1:
            cellNeighborList=CellNeighborListAuto(self.nTrackerPlugin,cell)
            for neighborSurfaceData in cellNeighborList:
               #print "is End=",nsdItr.isEnd()
               if neighborSurfaceData.neighborAddress and neighborSurfaceData.neighborAddress.type==1:
                  killFlag=0
                  break
            if killFlag:
               cell.targetVolume=0

from random import random
import types
class ContactLocalProductSteppable(SteppablePy):
   def __init__(self,_simulator,_frequency=10):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.contactProductPlugin=CompuCell.getContactLocalProductPlugin()
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def setTypeContactEnergyTable(self,_table):
      self.table=_table
      
   def start(self):
      for cell in self.cellList:
	 specificityObj=self.table[cell.type];
	 if isinstance(specificityObj,types.ListType):
            self.contactProductPlugin.setJVecValue(cell,0,(specificityObj[1]-specificityObj[0])*random())
         else:
            self.contactProductPlugin.setJVecValue(cell,0,specificityObj)


from PlayerPython import fillScalarValue as conSpecSet
class ContactSpecVisualizationSteppable(SteppablePy):
   def __init__(self,_simulator,_frequency=10):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.contactProductPlugin=CompuCell.getContactLocalProductPlugin()
      self.cellFieldG=self.simulator.getPotts().getCellFieldG()
      self.dim=self.cellFieldG.getDim()
      
   def setScalarField(self,_field):
      self.scalarField=_field
   def start(self):pass

   def step(self,mcs):
      cell=None
      cellFieldG=self.cellFieldG
      for x in xrange(self.dim.x):
         for y in xrange(self.dim.y):
            for z in xrange(self.dim.z):
               pt=CompuCell.Point3D(x,y,z)
               cell=cellFieldG.get(pt)
               if cell:
                  conSpecSet(self.scalarField,x,y,z,self.contactProductPlugin.getJVecValue(cell,0))
               else:
                  conSpecSet(self.scalarField,x,y,z,0.0)

class PressureDumperSteppable(SteppablePy):
   def __init__(self,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.inventory=None
      self.inc=0
      self.fixedTargetVolumeFlag=0
   #def __name__(self):
      #self.name
   def setFileName(self,_fileName):
      self.fileName=_fileName
   def init(self,_simulator):
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def setTargetVolume(self,_tv):
     self.targetVolume=_tv
     self.fixedTargetVolumeFlag=1
 
   def start(self):
      self.file=open(self.fileName,"w")
      
   def step(self,mcs):
      self.file.write("%d\t" % (mcs) )
      for cell in self.cellList:
         if self.fixedTargetVolumeFlag:
            self.file.write("%f\t" % (self.targetVolume-cell.volume) )
         else:
            self.file.write("%f\t" % (cell.targetVolume-cell.volume) )
      self.file.write("\n")
      
class IncrementPluginTargetVolume(SteppablePy):
   def __init__(self,_simulator,_frequency=1):
      SteppablePy.__init__(self,_frequency)
      self.simulator=_simulator
      self.inventory=self.simulator.getPotts().getCellInventory()
      self.cellList=CellList(self.inventory)
   def setVolumePlugin(self, _volumeEnergy):
      self.volumeEnergy=_volumeEnergy
   def step(self, mcs):
      self.volumeEnergy.vt+=1

# Generates movie
class MovieGenerator(SteppablePy):
    def __init__(self):
        pass # Init movie
    
    def step(self, mcs):
        pass # Take movie
    
    def finish(self):
        pass # Finish movie
        
#allows users to specify which file are to be stored in the simulation output directory             
class SimulationFileStorage(SteppablePy):
    def __init__(self,_simulator,_frequency=100):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.fileNamesList=[]
        self.filesCopiedFlag=False
        
    def addFileNameToStore(self,_name):
        self.fileNamesList.append(_name)
        
    def copyFilesToOutputDirectory(self):
        import CompuCellSetup
        screenshotDirectoryName=CompuCellSetup.getScreenshotDirectoryName()
        import shutil
        import os
        if screenshotDirectoryName=="":
            return
            
        for fileName in self.fileNamesList:
            fileName=CompuCellSetup.simulationPaths.normalizePath(fileName)            
            sourceFileName=os.path.abspath(fileName)
            destinationFileName=os.path.join(screenshotDirectoryName,os.path.basename(sourceFileName))
            
            shutil.copy(sourceFileName,destinationFileName)
            
    # copying will not work properly in the start function due to CC3D variable initialization order             
    # it is better therefore to implement it in a step fcn with appropriate flag determining 
    # if files have been writte or not
    # def start(self):
        # if not self.filesCopiedFlag:    
            # self.copyFilesToOutputDirectory()
            # self.filesCopiedFlag=True        
    
    def step(self,_mcs):
        print "SFS MCS=",_mcs," self.filesCopiedFlag=",self.filesCopiedFlag
        if not self.filesCopiedFlag:
            self.copyFilesToOutputDirectory()
            self.filesCopiedFlag=True        

