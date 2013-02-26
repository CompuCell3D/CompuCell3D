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
        
    def checkIfInTheLattice(self,_pt):
        if _pt.x>=0 and _pt.x<self.dim.x and  _pt.y>=0 and _pt.y<self.dim.y and _pt.z>=0 and _pt.z<self.dim.z:
            print "RETURNING TRUE"
            return True
        return False 
        
    def createNewCell (self,type,pt,xSize,ySize,zSize=1):
        if not self.checkIfInTheLattice(pt):
            return
        cell=self.potts.createCell()    
        cell.type=type
        self.cellField[pt.x:pt.x+xSize-1,pt.y:pt.y+ySize-1,pt.z:pt.z+zSize-1]=cell
        
                        
    def  deleteCell(self,cell):
#         pixelsToDelete=self.getCopyOfCellPixels(cell,SteppableBasePy.CC3D_FORMAT) # returns list of Point3D
        pixelsToDelete=self.getCopyOfCellPixels(cell,SteppableBasePy.TUPLE_FORMAT) # returns list of tuples
        
        
        self.mediumCell=CompuCell.getMediumCell()    
        for pixel in pixelsToDelete:            
            print "CELL.volume=",cell.volume
            self.cellField[pixel[0],pixel[1],pixel[2]]=self.mediumCell
            

                
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
#             self.cellField.set(pixel,cell)
             self.cellField[pixel.x,pixel.y,pixel.z]=cell
             
        # Now we will delete old pixels    
        pixelList=self.getCellPixelList(cell)
        pt=CompuCell.Point3D()
        
        self.mediumCell=CompuCell.getMediumCell()
        print " Deleting ",len(pixelsToDelete)," pixels of cell.id=",cell.id
        for pixel in pixelsToDelete:            
            self.cellField[pixel.x,pixel.y,pixel.z]=self.mediumCell
#             self.cellField.set(pixel,self.mediumCell)
    
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
        
            