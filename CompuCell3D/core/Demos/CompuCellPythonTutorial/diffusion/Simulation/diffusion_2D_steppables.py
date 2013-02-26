from PySteppables import *
import CompuCell
import sys


class ConcentrationFieldDumperSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
    
    def step(self,mcs):
        fileName="diffusion_output/FGF_"+str(mcs)+".dat"
        field=CompuCell.getConcentrationField(self.simulator,"FGF")        
        if field:
            try:                
                import CompuCellSetup
                fileHandle,fullFileName=CompuCellSetup.openFileInSimulationOutputDirectory(fileName,"w")
            except IOError:
                print "Could not open file ", fileName," for writing. Check if you have necessary permissions"                
                
            for i,j,k in self.everyPixel():
                fileHandle.write("%d\t%d\t%d\t%f\n"%(i,j,k,field[i,j,k]))
        
            fileHandle.close()