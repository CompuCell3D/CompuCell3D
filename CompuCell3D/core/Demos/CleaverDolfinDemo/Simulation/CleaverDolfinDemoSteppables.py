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
class CleaverDolfinDemoSteppable(SteppableBasePy):
   def __init__(self,_simulator,_frequency=10):
      SteppableBasePy.__init__(self,_simulator,_frequency)
#       global cpixel
#       print '      cpixel=numpy.array([[0,0]])',  cpixel
#       cpixel=numpy.array([[1,1]])
#       print '      cpixel=numpy.array([[0,0]])',  cpixel
      ###nx = 10;  ny = 10; nz=10
      ###parameters["linear_algebra_backend"] = "uBLAS"
      ###self.vtk_file = File("cc3d_1.pvd")
   def start(self):pass

   def step(self,mcs):
       
      ###self.mesh = Mesh("latticeMesh.xml")
      ###self.vtk_file = File("cc3d_1.pvd")
      ###self.vtk_file<<self.mesh       
      ###return
      
      self.mesh = Mesh()      
      
      import dolfinCC3D
      self.dolfinCC3DObj=dolfinCC3D.dolfinCC3D()
      self.dolfinCC3DObj.getX()
      dolfinCC3D.dolfinMeshInfo(self.mesh)
      #dolfinCC3D.simulateCleaverMesh(self.cellField, [1,2])      
      dolfinCC3D.buildCellFieldDolfinMeshUsingCleaver(self.cellField,self.mesh ,[1,2])
      print 'num_edges()=',self.mesh.num_edges()
      #print 'num_entities()=',self.mesh.num_entities()      
      self.vtk_file = File("cc3d_new.pvd")      
      self.vtk_file << self.mesh
      #return
      
      self.V = FunctionSpace(self.mesh, 'CG', 1)
      self.u = Function(self.V)
      

      
      #self.mesh=self.cleaverMeshDumper.getBlankDolfinMesh()      
      #print 'self.mesh.__hex__',self.mesh.__long__()
      print 'self.mesh -',dir(self.mesh)     
      print 'self.mesh._s()=',self.mesh.this
      #print type(self.mesh.__long__())
      
      
      #self.cleaverMeshDumper.dolfinMeshInfo(120)            
      #self.cleaverMeshDumper.dolfinMeshInfo(int(self.mesh.__long__()))      
      #return
      
      #self.cleaverMeshDumper.buildDolfinFromCurrentLatticeSnapshot(mesh)
      
      
      
      #return 
      
      global cpixel
      cpixel=numpy.array([[22,22,22]])
      for cell in self.cellList:
         if cell.type==2:
            pixelList=self.getCellPixelList(cell)
            for pixelTrackerData in pixelList:
                px=pixelTrackerData.pixel
                cpixel=numpy.append(cpixel,[[px.x,px.y,px.z]],axis=0)
                #print "pixel of cell id=",cell.id," type:",cell.type, " = ",pixelTrackerData.pixel," number of pixels=",pixelList.numberOfPixels()
      
      numpy.delete(cpixel,0,axis=0)
#       for i in range(len(cpixel)):
#          print "all pixels",cpixel[i,0],cpixel[i,1]
      print "all pixels"

#       plot(self.mesh)
#       interactive()
    # Define a MeshFunction over two subdomains
      parameters["linear_algebra_backend"] = "uBLAS"
      subdomains = MeshFunction('uint', self.mesh, 3)
    
      class Omega1(SubDomain):
        def inside(self, x, on_boundary):
            return True 
            
      class Omega0(SubDomain):
        
        def inside(self, x, on_boundary):
            #print 'xxxxxxx',x
            global cpixel
            flag=0
            #print "all pixels",len(cpixel)
#             q=10+mcs/10
#             return True if (between(x[1],(0,q)) and between(x[0],(0,25)))  else False
            for i in range(len(cpixel)):
                  #print 'Flagggggggggggggggggggggggggg'
                  yy=cpixel[i,1]*1.0
                  xx=cpixel[i,0]*1.0
                  zz=cpixel[i,2]*1.0
#                 print "all pixels", x[1],yy,yy+1,between(x[1],(yy,yy+1))
#                 return True if (between(x[1],(yy,yy+1)) and between(x[0],(xx,xx+1)))  else False
#                 #return True if (near(x[1],yy) and near(x[0],xx))  else False
                  if (between(x[1],(yy,yy+1)) and between(x[0],(xx,xx+1)) and between(x[2],(zz,zz+1))):
                    flag=1
                    #print 'FlaggggggggggggggggggggggggggQQQ'
            if flag==1:
                #print 'Flagggggggggggggggggggggggggg'
                return True
                
                    
    
    
    # Mark subdomains with numbers 0 and 1
      subdomain1 = Omega1()
      subdomain1.mark(subdomains, 1)
      print'DID OMEGA1---------------------'
      print 'start OMEGA0'
      start = time.time()
      subdomain0 = Omega0()
      subdomain0.mark(subdomains, 0)
      elapsed = (time.time() - start)
      print 'elapsed-OMEGA0',elapsed
#      cell_mesh = SubMesh(self.mesh, subdomains, 0)
      #plot(self.mesh)
      #interactive()
    
      
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
#       plot(u)
#       interactive()
      self.u.assign(u)     
      self.vtk_file << u

      
