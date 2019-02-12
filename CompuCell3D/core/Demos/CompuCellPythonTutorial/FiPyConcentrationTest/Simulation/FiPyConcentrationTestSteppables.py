from fipy import *
from numpy  import *
import fipy
import time,sys
from XMLUtils import dictionaryToMapStrStr as d2mss                                                                                   
from XMLUtils import CC3DXMLListPy
import FiPyInterface

class FiPyDiffuser:
    def __init__(self,_simulator):
	
        self.simulator = _simulator                                                                                    
        self.dim=self.simulator.getPotts().getCellFieldG().getDim()   
	
	self.nx = self.dim.x
        self.ny = self.dim.y
        #self.nz = self.dim.z
        self.dx = 1.
        self.dy = self.dx
        #self.dz = self.dx
        
        L = self.dx * self.nx
        self.mesh = PeriodicGrid2D(dx=self.dx, dy=self.dy, nx=self.nx, ny=self.ny)
        #self.mesh = Grid3D(dx=self.dx, dy=self.dy, dz=self.dz, nx=self.nx, ny=self.ny, nz=self.nz,)
        
        
        
        
        FiPyXMLData=self.simulator.getCC3DModuleData("Steppable","FiPySolver")
        diffusionFieldsElementVec=CC3DXMLListPy(FiPyXMLData.getElements("DiffusionField"))
        for diffusionFieldElement in diffusionFieldsElementVec:
	  if diffusionFieldElement.getFirstElement("DiffusionData").getFirstElement("FieldName").getText()=="Oxygen":
	    diffConstElement=diffusionFieldElement.getFirstElement("DiffusionData").getFirstElement("DiffusionConstant")  
	    self.cc3dDiffConst=float(diffConstElement.getText())
	    decayConstElement=diffusionFieldElement.getFirstElement("DiffusionData").getFirstElement("DecayConstant")  
	    self.cc3dDecayConst=float(decayConstElement.getText())
	#Is there a better way to get diffusion constant?
	
	print self.cc3dDiffConst, self.cc3dDecayConst
        self.D = FaceVariable(mesh=self.mesh, value=self.cc3dDiffConst)
        self.phi = CellVariable(name = "periodic",
                       mesh = self.mesh,
                       value = 0.0)
    def setConcentration(self,_x,_y,_conc):
            self.phi[_x*self.nx+_y] = _conc
    
    def setDoNotDiffuse(self,_doNotDiffuseList):
	
        x,y = self.mesh.getFaceCenters()
	for i in _doNotDiffuseList:
	  self.D.setValue(0.0, where=(i[0] <= x) & (x <= i[1]) & (i[0]<=y) & (y <= i[1]))
	  
	#self.D.setValue(0.0, where=(6. <= x) & (x <= 7.) & (6.<=y) & (y <= 7.))
	#self.D.setValue(0.0, where=(7. <= x) & (x <= 8.) & (6.<=y) & (y <= 7.))
	#self.D.setValue(0.0, where=(8. <= x) & (x <= 9.) & (6.<=y) & (y <= 7.))
	    
    def iterateDiffusion(self):
        x,y = self.mesh.getFaceCenters()

	#self.D.setValue(0.0, where=(6. <= x) & (x <= 7.) & (6.<=y) & (y <= 7.))
	#self.D.setValue(0.0, where=(7. <= x) & (x <= 8.) & (6.<=y) & (y <= 7.))
	#self.D.setValue(0.0, where=(8. <= x) & (x <= 9.) & (6.<=y) & (y <= 7.))
	
	
	#self.D.setValue(0.0, where=(6. <= x) & (x < 15.) & (6.<=y) & (y < 15))
	#D.setValue(0.0, where=(31. <= x) & (x < 35.) & (31.<=y) & (y < 35))
	#D.setValue(100.0, where=((x-15)**2+(y-20)**2 <= 100))
        
        eq = TransientTerm() == DiffusionTerm(coeff=self.D)              
        self.timeStepDuration = 1# 10 * 0.9 * self.dx**2 / (2 * D)
        eq.solve(var=self.phi,
             #boundaryConditions=BCs,
             dt=self.timeStepDuration)



from PySteppables import *
import CompuCell,PlayerPython
import sys
class FiPyConcentrationTestStetppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.solver = FiPyDiffuser(_simulator)
        #self.solver.setConcentration(20,50,2000)
        #self.solver.setConcentration(0,0,2000)
        #print 'FiPy: ', dir(self.solver)
    def start(self):
        
        for cell in self.cellList:
        # any code in the start function runs before MCS=0
            cell.targetVolume = 40
            cell.lambdaVolume = 2.0
      
    def step(self,mcs):
        field=CompuCell.getConcentrationField(self.simulator,"Oxygen")
        
        
        FiPyInteractor = FiPyInterface.FiPyInterfaceBase(2) #dimension of lattice (currently, 2D only works)

        FiPyInteractor.fillArray3D(self.solver.phi._getArray(),field)
        doNotDiffuseVec = FiPyInteractor.getDoNoDiffuseVec()
        self.solver.setDoNotDiffuse(doNotDiffuseVec)
	self.solver.iterateDiffusion()
        
	pt=CompuCell.Point3D(0,0,0)
        print '\n', field.get(pt), 
        sumField = 0 
        for i in xrange(self.dim.x):                                                                                              
                for j in xrange(self.dim.y):                                                                                          
                    for k in xrange(self.dim.z):                                                                                      
                        pt.x=i                                                                                                        
                        pt.y=j                                                                                                        
                        pt.z=k                                                                                                        
                        sumField += field.get(pt)
	print sumField