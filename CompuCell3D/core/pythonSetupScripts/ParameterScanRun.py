import sys
import os


def printHelp():
    help_str='''
paramScan script runs parameter scans untill the scan is finished or untill the max number of requested runs is reached
If any single simulation crashes for whatever reason the next simulation will start regardless and the parameter scan will keep going.
For this reason please make sure your simulation is bug free as much as possible to avoid half-finished simulations in your output directory

Usage (on osx use parameScan.command and on Windows use paramScan.bat):

paramScan.sh -i <Simulation File> [--guiScan] [--maxNumberOfRuns=XXX] [other options allowed by runScript or compucell3d scripts]

--guiScan - this option will use player to run simulations. Afer each simulation new instance of player will be opened to run the next simulation

--maxNumberOfRuns - this option speicifies maximuma number of runs allowed. 

example command:
./paramScan.sh -i Demos/cellsort_scam.cc3d --guiScan --maxNumberOfRuns=10 
'''
    print help_str
   
from os import environ
python_module_path=os.environ["PYTHON_MODULE_PATH"]
appended=sys.path.count(python_module_path)
if not appended:
    sys.path.append(python_module_path)

swig_lib_install_path=os.environ["SWIG_LIB_INSTALL_DIR"]
appended=sys.path.count(swig_lib_install_path)
if not appended:
    sys.path.append(swig_lib_install_path)


from ParameterScanUtils import getParameterScanCommandLineArgList   
import ParameterScanEnums
from SystemUtils import getCC3DRunScriptPath,getCC3DPlayerRunScriptPath,getCommandLineArgList

maxNumberOfRuns=-1 # if negative we use infinite while loop to run parameter scan

# For a gui-less run example command could look like 
# ./psrun.sh -i <SIMUNLATION_FILE> 
# this in turn would call ./runScript.sh -i <SIMUNLATION_FILE>
commandLineArgs=getCommandLineArgList()
print 'commandLineArgs=',commandLineArgs
cc3dScriptPath=getCC3DRunScriptPath()

# check if user requests parameter scan to be run from the GUI
usingGui=False
for arg in commandLineArgs:
    if arg=='--guiScan':
        # in a run with gui the example command could look like 
        # ./psrun.sh -i <SIMUNLATION_FILE> --guiScan
        # this in turn would call ./compucell3d.sh -i <SIMUNLATION_FILE> --guiScan
        cc3dScriptPath=getCC3DPlayerRunScriptPath()
        usingGui=True
        
popenArgs =[cc3dScriptPath] + getCommandLineArgList()

# when not using gui we pass --exitWhenDone to the run script to let it know that after each run it is supposed to exit rather than
# keep executing parameter scan. This way the consecutive parameter scan runs willbe managed entirely in this script rasther than using legacy 
# approach as implemened in runScript and CompuCellPythonSimulationCML.py
if not usingGui:
    popenArgs += ['--exitWhenDone']
print 'popenArgs=',popenArgs



import getopt
opts=None
args=None
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:s:o:w:p:f:c:h", ["help","noOutput","exitWhenDone","currentDir=","outputFrequency=","guiScan","maxNumberOfRuns=",'prefs','port','tweditPID='])
    print "opts=",opts
    print "args=",args
except getopt.GetoptError, err:
    # print help information and exit:
    print str(err) # will print something like "option -a not recognized"

    sys.exit(0)

invalid_command = True

for o, a in opts:
    print "o=",o
    print "a=",a
    if o in ("--maxNumberOfRuns"):             
        maxNumberOfRuns=int(a) 
    if o in ("--help"):    
        printHelp()
        sys.exit(0)
        
    if o not in ('--currentDir'): # this argument is passed from paramScan shell script
        invalid_command = False
    
if invalid_command:
    printHelp()
    sys.exit(0)

import subprocess
from subprocess import Popen



if maxNumberOfRuns<0:
    while True:
        try:    
            cc3dProcess = Popen(popenArgs)
            streamdata=cc3dProcess.communicate()        
            rc=cc3dProcess.returncode
            if rc==ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE:
                print 'GOT RETURN CODE 2'
                sys.exit(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)
            elif rc>10:
                continue
                
            print 'THIS IS RETURN CODE=', rc
            
            
        except KeyboardInterrupt:
            print 'User stopped the run'
            sys.exit(0)
else:
    for i in xrange(maxNumberOfRuns):
        try:    
            cc3dProcess = Popen(popenArgs)
            streamdata=cc3dProcess.communicate()        
            rc=cc3dProcess.returncode
            if rc==ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE:
                print 'GOT RETURN CODE 2'
                sys.exit(ParameterScanEnums.SCAN_FINISHED_OR_DIRECTORY_ISSUE)
            elif rc>10:
                continue
                
            print 'THIS IS RETURN CODE=', rc
            
            
        except KeyboardInterrupt:
            print 'User stopped the run'
            sys.exit(0)
        
        
        
