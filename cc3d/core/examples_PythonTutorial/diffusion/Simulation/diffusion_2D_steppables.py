from PySteppables import *
import CompuCell
import sys


class ConcentrationFieldDumperSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
    
    def step(self,mcs):
        fileName="diffusion_output/FGF_"+str(mcs)+".dat"
        field=CompuCell.getConcentrationField(self.simulator,"FGF")
        pt=CompuCell.Point3D()
        if field:
            try:                
                import CompuCellSetup
                fileHandle,fullFileName=CompuCellSetup.openFileInSimulationOutputDirectory(fileName,"w")
            except IOError:
                print "Could not open file ", fileName," for writing. Check if you have necessary permissions"
            
            for i in xrange(self.dim.x):
                for j in xrange(self.dim.y):
                    for k in xrange(self.dim.z):
                        pt.x=i
                        pt.y=j
                        pt.z=k
                        fileHandle.write("%d\t%d\t%d\t%f\n"%(pt.x,pt.y,pt.z,field.get(pt)))
        
            fileHandle.close()