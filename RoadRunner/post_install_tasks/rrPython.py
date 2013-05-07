import sys
import os
import time

sys.path.append(os.environ["PYTHON_MODULE_PATH"])
sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

        

from RoadRunner import RoadRunner 
import RoadRunnerSetup        


modelFile=os.path.abspath(os.path.join(os.environ["PREFIX_RR"],'test_1.xml'))

t1 = time.time()
        
rrList=[RoadRunner(RoadRunnerSetup.tempDirPath,RoadRunnerSetup.compilerSupportPath,RoadRunnerSetup.compilerExeFile) for i in xrange(1000)]        

print 'building list of RR ',(time.time()-t1)*1000.0

t1 = time.time()
        
compiled=False
for rr in rrList:
    if not compiled:
        compiled=rr.loadSBMLFromFile(modelFile, True)
        if not compiled:
            raise RuntimeError('COULD NOT COMPILE SBML MODEL '+modelFile)
    else:
        rr.loadSBMLFromFile(modelFile, False)

print 'loading SBML ',(time.time()-t1)*1000.0    

stepSize=0.5;
numSteps=20;

t1 = time.time()

for i in xrange(numSteps+1):
    t=i*stepSize
    idx=0
    for rr in rrList:
        rr.setNumPoints(1)
        rr.setTimeStart(t)
        rr.setTimeEnd(t+stepSize)


        rr.simulate()

        if not idx%1000:
            print 'idx=',idx,'rr t=',t,' S1=',rr.getValue('S1'),' S2=',rr.getValue('S2')
        
        idx+=1
              
print 'solving list of RR ',(time.time()-t1)*1000.0        
