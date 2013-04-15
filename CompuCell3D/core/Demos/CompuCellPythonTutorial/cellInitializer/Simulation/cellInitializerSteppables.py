from PySteppables import *
import CompuCell
import sys
import os

class CellInitializer(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        

    def start(self):
        self.mediumCell=self.cellField[0,0,0]
        
        simulationDir=os.path.dirname (os.path.abspath( __file__ ))
        fileFullName= os.path.join(simulationDir,'CellSurfaceCircularFactors.txt')
        fileFullName=os.path.abspath(fileFullName) # normalizing path
        
        fileHandle=open(fileFullName,"w")
        from math import *
        for radius in range(1,50):
            cellSurface,cellVolume = self.drawCircularCell(radius)
            cellSurfaceManualCalculation=self.calculateHTBL()
            fileHandle.write("%f %f %f %f %f\n"%(radius,2*pi*radius,cellSurface,pi*radius**2,cellVolume))
            print "radius=",radius," surface theor=",2*pi*radius," cellSurface=",cellSurface,"cellSurfaceManualCalculation=",cellSurfaceManualCalculation
            
        fileHandle.close()
        
        
    def drawCircularCell(self,_radius):

        xCenter=self.dim.x/2 
        yCenter=self.dim.y/2 
        zCenter=self.dim.z/2 
        
        # assigning medium to all lattce points
        self.cellField[:,:,:]=self.mediumCell
        
        # initializing large circular cell in the middle of the lattice
        cell=self.potts.createCell()
        cell.type=1
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):
                    
                    if ((x-xCenter)**2+(y-yCenter)**2) < _radius**2:
                        self.cellField[x,y,z]=cell

                        
        return cell.surface,cell.volume
        
    def calculateHTBL(self):

        cellSurfaceManualCalculation=0
        pt=CompuCell.Point3D()   
        
        for x in xrange(self.dim.x):
            for y in xrange(self.dim.y):
                for z in xrange(self.dim.z):

                    cell=self.cellField[x,y,z]
                        
                    if cell:
                        for pixelNeighbor in self.getPixelNeighborsBasedOnNeighborOrder(pt,1):                            
                            nCell=self.cellField[pixelNeighbor.pt.x,pixelNeighbor.pt.y,pixelNeighbor.pt.z]                                
                            if CompuCell.areCellsDifferent(nCell,cell):
                                cellSurfaceManualCalculation+=1
                                
                            
        return cellSurfaceManualCalculation

        
    def step(self,mcs):
        pass
        # print "self.cell.volume=",self.cell.volume

    def outputField(self,_fieldName,_fileName):
        field=CompuCell.getConcentrationField(self.simulator,_fieldName)
        if field:
            try:
                fileHandle=open(_fileName,"w")
            except IOError:
                print "Could not open file ", _fileName," for writing. Check if you have necessary permissions"

            print "dim.x=",self.dim.x
            for i in xrange(self.dim.x):
                for j in xrange(self.dim.y):
                    for k in xrange(self.dim.z):
                        fileHandle.write("%d\t%d\t%d\t%f\n"%(x,y,z,field[x,y,z]))

