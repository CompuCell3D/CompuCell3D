from PlayerPython import * 
import CompuCellSetup
#Steppables

import CompuCell
import PlayerPython
from PySteppables import *
from dolfin import *
import numpy,math
import time
if has_linear_algebra_backend("uBLAS"):
    parameters["linear_algebra_backend"] = "uBLAS"

    

cpixel=numpy.array([[0,0]])           
class CleaverDolfinDemoSteppable(DolfinSolverSteppable):
   def __init__(self,_simulator,_frequency=10):
      DolfinSolverSteppable.__init__(self,_simulator,_frequency)
      self.createFields(['DOLFIN'])
      
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
      
      subdomain0.mark(subdomains, 0)
      elapsed = (time.time() - start)
      print 'elapsed-OMEGA0',elapsed
    
      
      V0 = FunctionSpace(self.mesh, 'DG', 0)
      k = Function(V0)
      print'DID V0---------------------'
    # Loop over all cell numbers, find corresponding
    
    
    # subdomain number and fill cell value in k
      k_values = [1, 10]  # values of k in the two subdomains
      print 'len(subdomains.array())', len(subdomains.array())
      for cell_no in range(len(subdomains.array())):
        #print 'cell_no', cell_no
        subdomain_no = subdomains.array()[cell_no]
        k.vector()[cell_no] = k_values[subdomain_no]
      print'DID K---------------------'
    # Much more efficient vectorized code
    # (subdomains.array() has elements of type uint32, which
    # must be transformed to plain int for numpy.choose to work)
      help = numpy.asarray(subdomains.array(), dtype=numpy.int32)
      k.vector()[:] = numpy.choose(help, k_values)
      print'DID K2---------------------'
    #print 'k degree of freedoms:', k.vector().array()
    
    #plot(subdomains, title='subdomains')
    
    #V = FunctionSpace(mesh, 'Lagrange', 1)
#       V = FunctionSpace(mesh, 'CG', 1)
    
    
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
      f = Constant(-0.01) #Expression("x[0]*sin(x[1])")
      F = k*inner(grad(u), grad(v))*dx + (0.1*u/(u+0.1))*v*dx-0.01*v*k*dx
    
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
      
      dolfinCC3D.extractSolutionValuesAtLatticePoints(self.fieldDict['DOLFIN'], u)
      
      
