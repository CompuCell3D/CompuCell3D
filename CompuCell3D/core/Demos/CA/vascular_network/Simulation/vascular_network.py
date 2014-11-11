import sys,os,time
from random import random,randint,shuffle
print sys.path
print os.environ['PYTHONPATH']


import math

import CoreObjects
from CoreObjects import Point3D, Dim3D
import CA

from CAPySteppables import *

class VascularSteppable(CAPySteppableBase):
    def __init__(self,_caManager,_frequency=1):
        CAPySteppableBase.__init__(self,_caManager,_frequency)        
        import roadrunner
        
    def start(self):
        
        cell=caManager.createAndPositionCellS(Point3D(self.dim.x/2,self.dim.y/2,0))
        cell.type=4
        cell.size=1
        
        
        for x in xrange(30,35):
            for y in xrange(30,35):
                cell=caManager.createAndPositionCellS(Point3D(x,y,0))
                cell.type=1
                cell.size=2
                
        
#         file = open("D:/Program Files (x86)/CA2/vascular_network/Simulation/VesselLocations.dat",'r')
#         for line in file.readlines():
            
#             line=line.strip()
# #             print 'line=',line
#             splits=line.split()
# #             print splits
#             cell=self.caManager.createAndPositionCellS(Point3D(int(splits[0]),int(splits[1]),int(splits[2])))
#             cell.type=3
        
        
    def step(self, mcs):
        pass
#         for cell in self.cellList:
#             if cell.type==3: continue
            
#         for cell in self.cellListByType(3): 
#             print "id=",cell.id," type=",cell.type            
        


class SBMLOscillator(CAPySteppableBase):
    def __init__(self,_caManager,_frequency=1):
        CAPySteppableBase.__init__(self,_caManager,_frequency)        
        import roadrunner
        
    def start(self):
        
        file = open("D:/Program Files (x86)/CA2/vascular_network/Simulation/VesselLocations.dat",'r')
        for line in file.readlines():
            
            line=line.strip()
#             print 'line=',line
            splits=line.split()
#             print splits
            cell=self.caManager.createAndPositionCellS(Point3D(int(splits[0]),int(splits[1]),int(splits[2])))
            cell.type=3
        
        self.pW=self.addNewPlotWindow(_title='S1 concentration',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Variables')
        self.pW.addPlot('S1',_style='Dots',_color='red',_size=5)
        
        options={'relative':1e-10,'absolute':1e-12,'steps':10}
        self.setSBMLGlobalOptions(options)
        
        self.modelFile='Simulation/oscli.sbml' # this can be e.g. partial path 'Simulation/osci.sbml'
        self.stepSize=0.02
            
        self.initialConditions={}
        self.initialConditions['S1']=0.0
        self.initialConditions['S2']=1.0        
# # #         self.addSBMLToCellTypes(_modelFile=self.modelFile,_modelName='OSCIL',_types=[1],_stepSize=self.stepSize,_initialConditions=self.initialConditions)        


#         self.pW.showAllPlots()
        
    def step(self, mcs):

#         for cell in self.cellList:
#             if cell.type==3: continue
            
#         for cell in self.cellListByType(3): 
#             print "id=",cell.id," type=",cell.type            
        
        if mcs==20:
            self.addSBMLToCellTypes(_modelFile=self.modelFile,_modelName='OSCIL',_types=[1],_stepSize=self.stepSize,_initialConditions=self.initialConditions)        
        
        self.timestepSBML()  
        if mcs >20:    
            added=False
            for cell in self.cellList:
                
                if cell.type==1:     
                    state=self.getSBMLState(_modelName='OSCIL',_cell=cell) 
#                     print 'state',state['S1']
                    if not added:
                        self.pW.addDataPoint("S1",mcs,state['S1']) 
                        added=True
                        break
        
        self.pW.showAllPlots()
        
class MitosisSteppable(CAPySteppableBase):
    def __init__(self,_caManager,_frequency=1):
        CAPySteppableBase.__init__(self,_caManager,_frequency)        
        
    def start(self):
        return
        print 'self.dim=',self.dim
        print 'dir(self.dim) = ',dir(self.dim)
        if self.dim.z>1:
#             cell=caManager.createAndPositionCellS(CA.Point3D(self.dim.x/2,self.dim.y/2,self.dim.z/2))
            cell=caManager.createAndPositionCellS(Point3D(self.dim.x/2,self.dim.y/2,self.dim.z/2))
        else:
            cell=caManager.createAndPositionCellS(Point3D(self.dim.x/2,self.dim.y/2,0))
#             cell=caManager.createAndPositionCellS(CA.Point3D(self.dim.x/2,self.dim.y/2,0))
        cell.type=randint(1,2)    
        

    def attribCheck(self,mcs):
# # #         for cell in self.cellListByType(1): 
# # #             print "id=",cell.id," type=",cell.type
            
        
        for cell in self.cellList:
                
            cellDict=self.getDictionaryAttribute(cell)
            cellDict["Double_MCS_ID"]=mcs*2*cell.id
# # #             print "cellDict for cell.id=",cell.id, "is ",cellDict
            
    def step(self,mcs):
        return
        if mcs > 19:
            self.attribCheck(mcs)
            return
            
#         for cell in self.cellListByType(1): 
#             print "CA id=",cell.id," type=",cell.type
                
        
        # a=CellList(None)    
        bs = self.caManager.getBoundaryStrategy()
        if self.dim.z>1:
            self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(3)
        else:
            self.maxNeighborIdx = bs.getMaxNeighborIndexFromNeighborOrder(2)
            
        print '\n\n\n\n maxNeighborIdx=',self.maxNeighborIdx
        cellVector = self.inventory.generateCellVector()
        print 'dir(cellVector)=',dir(cellVector)
        print 'cellVector.size()=',cellVector.size()
        idx = randint(0,cellVector.size()-1)
        cell = cellVector[idx]
        print 'idx=',idx,'cell.id=',cell.id

        
        cells_to_divide=[]    
        for cell in self.cellList:
            # print 'cell.id',cell.id,' cell.type=',cell.type,'cell.xCOM=',cell.xCOM,' yCOM=',cell.yCOM
            cells_to_divide.append(cell)
        
        shuffle(cells_to_divide)
        for cell in cells_to_divide:
            # print 'cell.id=',cell.id
            # print 'xCOM=',cell.xCOM,' yCOM=',cell.yCOM, 'zCOM=',cell.zCOM
            cellStack = self.cellField.get(Point3D(cell.xCOM,cell.yCOM,cell.zCOM))
            comPt = Point3D(cell.xCOM,cell.yCOM,cell.zCOM)
            
            randOffsetIdx = randint(0,self.maxNeighborIdx)
            
            neighbor=bs.getNeighborDirect( comPt , randOffsetIdx)
            # print 'randOffsetIdx=',randOffsetIdx,' neighbor.pt=',neighbor.pt
            if not neighbor.distance:
                continue
                
            cellStackLocal=self.cellField.get(neighbor.pt)
         
            
            if not cellStackLocal or not cellStackLocal.isFull(): # either neighboring site is empty or occupied but not full
                
                cell = caManager.createAndPositionCellS(neighbor.pt)
                cell.type = randint(1,2) 
                
    

# dictAdder=DictAdder()

import CACompuCellSetup

caSimulation = CACompuCellSetup.CASimulationPy()

caSimulation.initialize(dim=CoreObjects.Dim3D(110,110,1),globalCarryingCapacity=3,numSteps=1000) 

fieldDim=caSimulation.dim

solverRegistry=caSimulation.steppableRegistry

caManager = caSimulation.caManager 

#PLUGINS 
#CELL TYPE INFO
caSimulation.setCellTypeInfo({1:'Cancer',2:'Stem',3:'Vascular',4:'Tip'})
caSimulation.setFrozenTypes(['Cancer','Vascular'])

#PROBABILITY FUNCTION
cp = caSimulation.registerProbabilityFunctionByName('CanonicalProbability')
cp.diffCoeff = 0.1
cp.deltaT = 1.0

chemPr = caSimulation.registerProbabilityFunctionByName('ChemotaxisProbability')
chemPr.diffCoeff = 0.1
chemPr.deltaT = 1.0
chemPr.addChemotaxisData(FieldName='VEGF', ChemotaxingType='Tip', Lambda=100.0)


cellTrail = caSimulation.registerFieldChangeWatcherByName('CellTrail')
cellTrail.addMovingCellTrail(MovingCellType='Tip',TrailCellType='Vascular',TrailCellSize=2) 


#SOLVERS  
dSolverFE = caSimulation.registerSolverByName('DiffusionSolverFE')

dSolverFE.addField(\
Name='FGF',\
DiffusionData = {'DiffusionConstant':0.1,'DecayConstant':0.0001},\
SecretionData = {'Vascular':0.1}
)

dSolverFE.addField(\
Name='VEGF',\
DiffusionData = {'DiffusionConstant':0.1,'DecayConstant':0.0001},\
SecretionData = {'Cancer':100.1}
)

# dSolverFE.printConfiguration()

vascularSteppable = VascularSteppable(caManager)
caSimulation.registerSteppable(vascularSteppable)

caSimulation.mainLoop()




