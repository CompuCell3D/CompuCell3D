from PlayerPython import * 
import CompuCellSetup
#Steppables

import CompuCell
import PlayerPython
from PySteppables import *
from dolfin import *

import ufl
import numpy,math
import time


if has_linear_algebra_backend("uBLAS"):
    parameters["linear_algebra_backend"] = "uBLAS"

class StepFunctionExpressionPython(Expression):
    
    def __init__(self,_cellField=None,_cellTypeToValueMap={},_cellIdsToValueMap={},_defaultValue=0.0):
        
        self.cellField=_cellField
        self.cellTypeToValueMap=_cellTypeToValueMap
        self.cellIdsToValueMap=_cellIdsToValueMap        
        self.defaultValue=_defaultValue       
        self.pt=CompuCell.Point3D()
        
        
    def eval(self,values,x):
        self.pt.x=int(round(x[0]))
        self.pt.y=int(round(x[1]))            
        self.pt.z=int(round(x[2]))                        
        cell=self.cellField.get(self.pt)
        
        if cell :
            #check types
            try:
                values[0]=self.cellTypeToValueMap[cell.type]
                return    
            except LookupError,e:
                pass
            #check cell ids    
            try:
                values[0]=self.cellIdsToValueMap[cell.type]
                return    
            except LookupError,e:
                pass
            # if nothing found use default value    
            values[0]=self.defaultValue
                
        else:
            #check types - Medium
            try:
                values[0]=self.cellTypeToValueMap[0]
                return    
            except LookupError,e:
                pass
            
            # if nothing found use default value    
            values[0]=self.defaultValue
        
        
        
        

cpixel=numpy.array([[0,0]])           
class CleaverDolfinDemoSteppable(DolfinSolverSteppable):
   def __init__(self,_simulator,_frequency=10):
      DolfinSolverSteppable.__init__(self,_simulator,_frequency)
      self.createFields(['DOLFIN'])
      import dolfinCC3D

   def start(self):pass

   def step(self,mcs):
   
   
      self.mesh = Mesh() 
      import dolfinCC3D

      
      dolfinCC3D.buildCellFieldDolfinMeshUsingCleaver(self.cellField,self.mesh ,[1,2])
      
      self.vtk_file = File("cc3d_new.pvd")      
      self.vtk_file << self.mesh
      
      self.V = FunctionSpace(self.mesh, 'CG', 1)
      self.u = Function(self.V)
      
      parameters["linear_algebra_backend"] = "uBLAS"
      subdomains = MeshFunction('uint', self.mesh, 3)
    
      # OmegaCustom1() is defined in pyinterface/dolfinCC3D/CustomSubDomains.h  and SWIG wrapper is in pyinterface/dolfinCC3D/dolfinCC3D.i   
      subdomain1 = dolfinCC3D.OmegaCustom1()
      subdomain1.mark(subdomains, 1)
      
      print'DID OMEGA1---------------------'
      print 'start OMEGA0'
      start = time.time()
      
      subdomain0 = dolfinCC3D.OmegaCustom0()      
      subdomain0.setCellField(self.cellField) # passing cell field to OmegaCustom0()      
      subdomain0.init() # making a list of all pixels belonging to cell type=2       
      
      print 'type(subdomain0)=',type(subdomain0)
      
      subdomain0.mark(subdomains, 0)
      elapsed = (time.time() - start)
      print 'elapsed-OMEGA0',elapsed
    
      
      V0 = FunctionSpace(self.mesh, 'DG', 0)
    
    
    # Define Dirichlet conditions for y=0 boundary
    
      tol = 1E-14   # tolerance for coordinate comparisons
      class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < tol
    
      Gamma_0 = DirichletBC(self.V, Constant(0), BottomBoundary())
    # Define Dirichlet conditions for y=1 boundary
    
      class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
     
      Gamma_1 = DirichletBC(self.V, Constant(0), TopBoundary())
    
      bcs = [Gamma_0, Gamma_1]
      print 'DID BCS'  
      u = Function(self.V)
      u.assign(self.u)
      v = TestFunction(self.V)
      
      
      cellTypeToValue={}
      cellTypeToValue[0]=0.2
      cellTypeToValue[1]=1.0
      cellTypeToValue[2]=10.0

#       cellIdsToValue={}
#       cellIdsToValue[1]=0.3
#       cellIdsToValue[2]=0.5
#       cellIdsToValue[3]=1.0

      # this function takes as an argument dict of {cellType:scalar value}, dict of {cellId:scalar value} and a default scalar value
      # all arguments are optional
      # stepFcn=self.getStepFunctionExpressionFlex(cellTypeToValue,cellIdsToValue,1.0) 
      stepFcn=self.getStepFunctionExpressionFlex(cellTypeToValue)
      
      # Alternative Python implementation - it is mush slower though
#       stepFcn=StepFunctionExpressionPython(self.cellField,cellTypeToValue)
      
      
      
      print 'dir(stepFcn)=',dir(stepFcn)
      
#       F=0.01*v*customExpr*dx
#       F=0.01*v*stepFcn*dx
#       return
      
      f = Constant(-0.01) #Expression("x[0]*sin(x[1])")
      F = stepFcn*inner(grad(u), grad(v))*dx + (0.1*u/(u+0.1))*v*dx-0.01*v*stepFcn*dx
      
      
    
    # Compute solution
      print 'start solve'
      start = time.time()
      #parameters["linear_algebra_backend"] = "Epetra"
      solve(F == 0, u, bcs, solver_parameters={"newton_solver":
                                            {"relative_tolerance": 1e-6}})
      elapsed = (time.time() - start)
      print 'elapsed',elapsed
      print 'u=',type(u)
      print 'u.geometric_dimension=',u.geometric_dimension()  
      
      #dolfinCC3D.extractSolutionValuesAtLatticePoints(self.fieldDict['DOLFIN'], u) 3 this call is ok
      # this is a call to use to extract fields within a box - may be necessary if interpolate function throws excepations usually at the boundaries of the lattice      
      dolfinCC3D.extractSolutionValuesAtLatticePoints(self.fieldDict['DOLFIN'],u,CompuCell.Dim3D(1,1,1),CompuCell.Dim3D(self.dim.x-1,self.dim.y-1,self.dim.z-1))
      
      

