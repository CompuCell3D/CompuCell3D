from PyPlugins import *
from CompuCell import NeighborFinderParams



class VolumeEnergyFunction(EnergyFunctionPy):
    def __init__(self,_energyWrapper):
        EnergyFunctionPy.__init__(self)
        self.energyWrapper=_energyWrapper
        self.vt=0.0
        self.lambda_v=0.0
    def setParams(self,_lambda,_targetVolume):
        self.lambda_v=_lambda;
        self.vt=_targetVolume
    def changeEnergy(self):
        energy=0.0
        if(self.energyWrapper.isNewCellValid()):
            #print "CellVolume=",self.energyWrapper.newCell.volume
            energy+=self.lambda_v*(1+2*(self.energyWrapper.newCell.volume-self.vt))
        if(self.energyWrapper.isOldCellValid()):
            energy+=self.lambda_v*(1-2*(self.energyWrapper.oldCell.volume-self.vt))
            #print "Energy=",energy
        return energy

class SurfaceEnergyFunction(EnergyFunctionPy):
    def __init__(self,_energyWrapper):
        EnergyFunctionPy.__init__(self)
        self.energyWrapper=_energyWrapper
        self.st=0.0
        self.lambda_s=0.0
        self.potts=self.energyWrapper.potts
        self.cellField=self.energyWrapper.potts.getCellFieldG()
        self.nfparams=NeighborFinderParams()
        
    def diffEnergy(self, surface, diff):
        return self.lambda_s *(diff * diff + 2 * diff * (surface - self.st))

    def setParams(self,_lambda,_targetSurface):
        self.lambda_s=_lambda;
        self.st=_targetSurface
     
    def changeEnergy(self):
        energy=0.0
        oldDiff=0
        newDiff=0
        oldCellId=-1
        newCellId=-1
        nCellId=-1
        print_flag=0;
        
        #print "changePoint=",self.energyWrapper.changePt
        #if self.energyWrapper.changePoint.x==35 and self.energyWrapper.changePoint.y==16:
            #print_flag=1;
            #print "got:",self.energyWrapper.changePoint
        if self.energyWrapper.isOldCellValid():
            oldCellId=self.energyWrapper.oldCell.id
            
        if self.energyWrapper.isNewCellValid():
            newCellId=self.energyWrapper.newCell.id
        
        #if print_flag:
            #print "changePoint:",self.energyWrapper.changePoint
        self.nfparams.reset()
        self.nfparams.pt=self.energyWrapper.changePoint
        self.nfparams.checkBounds=0
        #print "nfparams.pt=",self.nfparams.pt," neighbor=",self.cellField.nextNeighbor(self.nfparams)
        while 1:
            n=self.cellField.nextNeighbor(self.nfparams)
            if self.nfparams.distance>1:
                break
            #if print_flag:
                #print "chPt=",self.nfparams.pt," n=",n
            nCell=self.cellField.get(n)
            if self.energyWrapper.isCellMedium(nCell):
                nCellId=-1
            else:
                nCellId=nCell.id;
            
            #print "energyWrapper.newCell=",self.energyWrapper.newCell.id," nCell=",nCell.id
            if newCellId==nCellId:
                newDiff=newDiff-1
            else:
                newDiff=newDiff+1
                
            if oldCellId==nCellId:
                oldDiff=oldDiff+1
            else:
                oldDiff=oldDiff-1
        #if print_flag:
            #print "oldDiff=",oldDiff," newDiff",newDiff
        if(self.energyWrapper.isNewCellValid()):
            #print "newCell.id=",self.energyWrapper.newCell.id
            #print "self.energyWrapper.newCell.surface=",self.energyWrapper.newCell.surface
            #energy+=self.diffEnergy(self.energyWrapper.newCell.surface, newDiff)
            energy+=self.lambda_s *(newDiff**2 + 2 * newDiff * (self.energyWrapper.newCell.surface - self.st))
            #if print_flag:
                #print "self.energyWrapper.newCell.surface=",self.energyWrapper.newCell.surface
        #else:
            #print "NONE newCell.id=",self.energyWrapper.newCell.id
        if(self.energyWrapper.isOldCellValid()):
            #energy+=self.diffEnergy(self.energyWrapper.oldCell.surface, oldDiff)
            energy+=self.lambda_s *(oldDiff**2 + 2 * oldDiff * (self.energyWrapper.oldCell.surface - self.st))
            #if print_flag:
                #print "self.energyWrapper.oldCell.surface=",self.energyWrapper.oldCell.surface
        #print "energy=",energy
        #if print_flag:
            #print "pt=",self.energyWrapper.changePoint," e=",energy
        return energy



class TypeChangeWatcherExample(TypeChangeWatcherPy):
    def __init__(self,_changeWatcher):
        TypeChangeWatcherPy.__init__(self)
        self.changeWatcher=_changeWatcher
    def typeChange(self):
        newCell=self.changeWatcher.newCell
        newType=self.changeWatcher.newType

        if(newCell):
            #Note that in the typeChange function of the watcher we do not actually change type, we rather react to this type change updating whtever is neceessary
            print "Cell ID=",newCell.id," type=",newCell.type," changes type to: ",newType






from CompuCell import MitosisSimplePlugin

class MitosisPy (StepperPy,Field3DChangeWatcherPy):
    def __init__(self,_changeWatcher):
        Field3DChangeWatcherPy.__init__(self,_changeWatcher)

        self.mitosisPlugin=MitosisSimplePlugin()
        self.doublingVolume=50
        self.mitosisPlugin.setDoublingVolume(self.doublingVolume)        
        self.mitosisPlugin.init(self.changeWatcher.sim)
        self.mitosisPlugin.turnOn() #has to be called after init to make sure vectors are allocated
        self.counter=0
        self.mitosisFlag=0
        #self.childCell
        #self.parentCell
    def setPotts(self,potts):
        self.mitosisPlugin.setPotts(potts)
    def field3DChange(self):
        if self.changeWatcher.newCell and self.changeWatcher.newCell.volume>self.doublingVolume:
            print "BIG CELL I WILL DO MITOSIS"
            self.mitosisPlugin.field3DChange(self.changeWatcher.changePoint,self.changeWatcher.newCell,self.changeWatcher.newCell)
            self.mitosisPlugin.setMitosisFlag(1)
            

    def step(self):
        mitosisDoneFlag=False
        if self.mitosisPlugin.getMitosisFlag():
            self.mitosisFlag=self.mitosisPlugin.doMitosis()
            self.childCell=self.mitosisPlugin.getChildCell()
            self.parentCell=self.mitosisPlugin.getParentCell()
            self.updateAttributes()
            self.mitosisPlugin.setMitosisFlag(0)

    def updateAttributes(self):
        print "self.parentCell.type % 2=",self.parentCell.type % 2
        if self.parentCell.type % 2:
            self.childCell.type=2
            #self.childCell.type=self.parentCell.type
        else:
            self.childCell.type=1
            #self.childCell.type=self.parentCell.type

from CompuCell import MitosisSimplePlugin
class MitosisPyPluginBase(StepperPy,Field3DChangeWatcherPy):
    def __init__(self,_simulator,_changeWatcherRegistry,_stepperRegistry):

        Field3DChangeWatcherPy.__init__(self,_changeWatcherRegistry)
        self.simulator=_simulator
        self.mitosisPlugin=MitosisSimplePlugin()
        self.mitosisPlugin.setPotts(self.simulator.getPotts())        
        self.mitosisPlugin.init(self.changeWatcher.sim)
        self.mitosisPlugin.turnOn() #has to be called after init to make sure vectors are allocated        
        self.counter=0
        self.mitosisFlag=0
        _changeWatcherRegistry.registerPyChangeWatcher(self)
        _stepperRegistry.registerPyStepper(self)
        self.directionalMitosisFlagSet=[1,0,0]
        self.useOrientationVectorMitosis=False
        self.nx=1.0
        self.ny=0.0
        self.nz=0.0        
    
    def setPotts(self,potts):
        self.mitosisPlugin.setPotts(potts)
        
    def setDoublingVolume(self,_doublingVolume):
        self.doublingVolume=_doublingVolume;
        self.mitosisPlugin.setDoublingVolume(self.doublingVolume)
    def setDivisionAlongMajorAxis(self):
        self.directionalMitosisFlagSet=[0,1,0]
    def setDivisionAlongMinorAxis(self):
        self.directionalMitosisFlagSet=[0,0,1]
    def setNondirectionalDivision(self):
        self.directionalMitosisFlagSet=[1,0,0]        
    def setMitosisOrientationVector(self,_nx,_ny,_nz):
        self.useOrientationVectorMitosis=True    
        self.nx=_nx
        self.ny=_ny
        self.nz=_nz
    def unsetMitosisOrientationVector(self):
        self.useOrientationVectorMitosis=False    
        
    # these functions show how to extract semiminor axis. MItosis can be done  along major, minor or user specified axis
    # once you know directions of principal axec of cells you may do custom mitosis along vector derived using principal orientation vectors
    def getSemiminorVectorXY(self):        
         orientationVectors=self.mitosisPlugin.getOrientationVectorsMitosis2D_xy(self.changeWatcher.newCell)
         
         print "orientationVectors.semiminorVec=",orientationVectors.semiminorVec.x,",",orientationVectors.semiminorVec.y,",",orientationVectors.semiminorVec.z
         print "orientationVectors.semimajorVec=",orientationVectors.semimajorVec.x,",",orientationVectors.semimajorVec.y,",",orientationVectors.semimajorVec.z
         
         return orientationVectors.semiminorVec
         
    def getSemiminorVectorXY(self):        
         orientationVectors=self.mitosisPlugin.getOrientationVectorsMitosis2D_xy(self.changeWatcher.newCell)
         
         print "orientationVectors.semiminorVec=",orientationVectors.semiminorVec.x,",",orientationVectors.semiminorVec.y,",",orientationVectors.semiminorVec.z
         print "orientationVectors.semimajorVec=",orientationVectors.semimajorVec.x,",",orientationVectors.semimajorVec.y,",",orientationVectors.semimajorVec.z
         
         return orientationVectors.semimajorVec
        
    def field3DChange(self):
    
        self.newCell=self.changeWatcher.getNewCell()        
        if self.newCell and self.newCell.volume>self.doublingVolume:
        
            self.changePoint=self.changeWatcher.getChangePoint()
            self.mitosisPlugin.field3DChange(self.changePoint, \
            self.newCell, \
            self.newCell)
            self.mitosisPlugin.setMitosisFlag(1)

            
    def step(self):
        mitosisDoneFlag=False
        if self.mitosisPlugin.getMitosisFlag():
            if self.useOrientationVectorMitosis:

                mitosisDoneFlag=self.mitosisPlugin.doDirectionalMitosisOrientationVectorBased(self.nx,self.ny,self.nz)
            else:             
                if self.directionalMitosisFlagSet[0]:
                    mitosisDoneFlag=self.mitosisPlugin.doMitosis()
                    
                if self.directionalMitosisFlagSet[1]:#do mitosis along major Axis
                    self.mitosisPlugin.setDivideAlongMajorAxis()
                    # right before doing mitosis you may want to calculate division axis based on orientatin of principal axes. This is the place where you would do these calculations
                    # self.getSemiminorVectorXY()
                    
                    mitosisDoneFlag=self.mitosisPlugin.doDirectionalMitosis()
                    
                if self.directionalMitosisFlagSet[2]:#do mitosis along minor Axis
                    self.mitosisPlugin.setDivideAlongMinorAxis()
                    mitosisDoneFlag=self.mitosisPlugin.doDirectionalMitosis()
            self.childCell=self.mitosisPlugin.getChildCell()
            self.parentCell=self.mitosisPlugin.getParentCell()
            
            if mitosisDoneFlag:
                self.updateAttributes()
            self.mitosisPlugin.setMitosisFlag(0)
            
    def updateAttributes(self):
        self.childCell.targetVolume=self.parentCell.targetVolume
        self.childCell.lambdaVolume=self.parentCell.lambdaVolume
        self.childCell.type=self.parentCell.type
        
        

from CompuCell import ChemotaxisSimpleEnergy

import exceptions
class ChemotaxisPy(EnergyFunctionPy):
    def __init__(self,_energyWrapper):
        EnergyFunctionPy.__init__(self)
        self.energyWrapper=_energyWrapper
        self.chemotaxisEnergy=ChemotaxisSimpleEnergy()
        self.fieldList=[]
        self.lambdaList=[]
    def setFieldAndLambda(self,_fieldName, _lambda):
        field=CompuCell.getConcentrationField(self.energyWrapper.sim,_fieldName)
        if not field:
            message="Field "+_fieldName+" was not registered in Simulator object"
            raise exceptions.AssertionError,message
            
        self.fieldList.append([field,_lambda])
    def changeEnergy(self):
        energy=0.0
        energyWrapperLocal=self.energyWrapper
        chemotaxisEnergyLocal=self.chemotaxisEnergy
        changePt=energyWrapperLocal.changePoint
        flipNeighbor=energyWrapperLocal.flipNeighbor

        #if energyWrapperLocal.newCell:
            #print "newCell type is ",energyWrapperLocal.newCell.type,"  ", energyWrapperLocal.isNewCellValid()
        #else:
            #print "newCell is Medium",energyWrapperLocal.isNewCellValid()
            
        if energyWrapperLocal.newCell:
            for chemotaxisData in self.fieldList:
                conc=chemotaxisData[0].get(changePt)
                concFlipNeighbor=chemotaxisData[0].get(flipNeighbor)
                if energyWrapperLocal.newCell.type==2:
                    energy+=chemotaxisEnergyLocal.simpleChemotaxisFormula(concFlipNeighbor,conc,chemotaxisData[1])
        
        return energy


