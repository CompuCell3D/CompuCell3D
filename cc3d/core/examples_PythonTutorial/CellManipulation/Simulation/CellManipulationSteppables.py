from PySteppables import *
import CompuCell
import sys
from random import uniform
import math
'''
    For detailed explanation how Cell manipulation works see CellManipulationSteppableExplained below
        
'''

class CellManipulationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    
    def step(self,mcs):
        if mcs==10:
        
            shiftVector=CompuCell.Point3D(20,20,0)
            
            for cell in self.cellList:
                self.moveCell(cell,shiftVector)
                
        if mcs==20:
            pt=CompuCell.Point3D(50,50,0)
            self.createNewCell(2,pt,5,5,1)

        if mcs==30:
            for cell in self.cellList:
                self.deleteCell(cell)
        
            

class CellManipulationSteppableExplained(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        
        pass
        # #tryuing to extract pointer to medium cell
        # self.mediumCell=None
        # pt=CompuCell.Point3D()
        # for x in range (self.dim.x):
            # for y in range (self.dim.y):
                # for z in range (self.dim.z):
                    # pt.x=x
                    # pt.y=y
                    # pt.z=z
                    # self.mediumCell=self.cellField.get(pt)
                    # if not self.mediumCell:
                        # break # we have just found medium pixel
                    
    def checkIfInTheLattice(self,_pt):
        if _pt.x>=0 and _pt.x<self.dim.x and  _pt.y>=0 and _pt.y<self.dim.y and _pt.z>=0 and _pt.z<self.dim.z:
            print "RETURNING TRUE"
            return True
        return False 
        
    def createNewCell (self,type,pt,xSize,ySize,zSize=1):
        if not self.checkIfInTheLattice(pt):
            return
        cell=self.potts.createCellG(pt)    
        cell.type=type
        
        ptCell=CompuCell.Point3D()
        
        for x in range(pt.x,pt.x+xSize,1):
            for y in range(pt.y,pt.y+ySize,1):        
                for z in range(pt.z,pt.z+zSize,1):
                    ptCell.x=x
                    ptCell.y=y
                    ptCell.z=z
                    print "ptCell=",ptCell
                    if self.checkIfInTheLattice(ptCell):
                        self.cellField.set(ptCell,cell)
                        
    def  deleteCell(self,cell):
        pixelsToDelete=[] #used to hold pixels to delete        
        pixelList=self.getCellPixelList(cell)
        pt=CompuCell.Point3D()
        print "DELETING CELL WITH ",pixelList.numberOfPixels()," pixels volume=",cell.volume
        
        for pixelTrackerData in pixelList:
            pixelsToDelete.append(CompuCell.Point3D(pixelTrackerData.pixel))
            
            self.mediumCell=CompuCell.getMediumCell()                        
            
        for pixel in pixelsToDelete:
            
            self.cellField.set(pixel,self.mediumCell)        
            print "CELL.volume=",cell.volume
        self.cleanDeadCells()        
                
    def moveCell(self, cell, shiftVector):
                #we have to make two list of pixels :
                pixelsToDelete=[] #used to hold pixels to delete
                pixelsToMove=[] #used to hold pixels to move
                
                # If we try to reassign pixels in the loop where we iterate over pixel data we will corrupt the container so in the loop below all we will do is to populate the two list mentioned above
                pixelList=self.getCellPixelList(cell)
                pt=CompuCell.Point3D()
                print " Moving ",pixelList.numberOfPixels()," pixels of cell.id=",cell.id," . Shift vector=",shiftVector
                for pixelTrackerData in pixelList:
                    pt.x = pixelTrackerData.pixel.x + shiftVector.x
                    pt.y = pixelTrackerData.pixel.y + shiftVector.y
                    pt.z = pixelTrackerData.pixel.z + shiftVector.z
                    # here we are making a copy of the cell 
                    print "adding pt=",pt
                    pixelsToDelete.append(CompuCell.Point3D(pixelTrackerData.pixel))
                    
                    if self.checkIfInTheLattice(pt):
                        pixelsToMove.append(CompuCell.Point3D(pt))
                        # self.cellField.set(pt,cell)
                 
                # Now we will move cell
                for pixel in pixelsToMove:
                    self.cellField.set(pixel,cell)
                 
                # Now we will delete old pixels    
                pixelList=self.getCellPixelList(cell)
                pt=CompuCell.Point3D()
                
                self.mediumCell=CompuCell.getMediumCell()
                print " Deleting ",len(pixelsToDelete)," pixels of cell.id=",cell.id
                for pixel in pixelsToDelete:
                    self.cellField.set(pixel,self.mediumCell)
    
    def step(self,mcs):
        if mcs==10:
        
            shiftVector=CompuCell.Point3D(20,20,0)
            
            for cell in self.cellList:
                self.moveCell(cell,shiftVector)
                
        # if mcs==20:
            # pt=CompuCell.Point3D(50,50,0)
            # self.createNewCell(2,pt,5,5,1)

        if mcs==20:
            for cell in self.cellList:
                self.deleteCell(cell)
        
            