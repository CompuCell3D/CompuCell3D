from PySteppables import *
import CompuCell
import sys





class CellInitializer(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        

    def start(self):
        self.mediumCell=self.cellField.get(CompuCell.Point3D())
        fileHandle=open("examples_PythonTutorial/cellInitializer/CellSurfaceCircularFactors.txt","w")
        from math import *
        for radius in range(1,50):
            cellSurface,cellVolume = self.drawCircularCell(radius)
            cellSurfaceManualCalculation=self.calculateHTBL()
            fileHandle.write("%f %f %f %f %f\n"%(radius,2*pi*radius,cellSurface,pi*radius**2,cellVolume))
            print "radius=",radius," surface theor=",2*pi*radius," cellSurface=",cellSurface,"cellSurfaceManualCalculation=",cellSurfaceManualCalculation
            
        fileHandle.close()
        
        
    def drawCircularCell(self,_radius):
        centerPt=CompuCell.Point3D(self.dim.x/2 , self.dim.y/2 ,self.dim.z/2)
        
        pt=CompuCell.Point3D()
        
        # assigning medium to all lattce points
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt.x=x
                    pt.y=y
                    pt.z=z
                    self.cellField.set(pt,self.mediumCell)
        
        # initializing large circular cell in the middle of the lattice
        cell=self.potts.createCellG(centerPt)
        cell.type=1
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    
                    pt.x=x
                    pt.y=y
                    pt.z=z
                    
                    if ((pt.x-centerPt.x)**2+(pt.y-centerPt.y)**2) < _radius**2:
                        self.cellField.set(pt,cell)
                        
        return cell.surface,cell.volume
        
    def calculateHTBL(self):
        cellSurfaceManualCalculation=0
        self.boundaryStrategy=CompuCell.BoundaryStrategy.getInstance()
        self.maxNeighborIndex=self.boundaryStrategy.getMaxNeighborIndexFromDepth(1.1)
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    pt=CompuCell.Point3D(x,y,z)
                    cell=self.cellField.get(pt)
                        
                    if cell:
                        for i in xrange (self.maxNeighborIndex+1):
                            pixelNeighbor=self.boundaryStrategy.getNeighborDirect(pt,i)
                            if pixelNeighbor.distance: #neighbor is valid
                                nCell=self.cellField.get(pixelNeighbor.pt)
                                if CompuCell.areCellsDifferent(nCell,cell):
                                    cellSurfaceManualCalculation+=1
                                
                            
        return cellSurfaceManualCalculation
        
    def step(self,mcs):
        pass
        # print "self.cell.volume=",self.cell.volume

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

