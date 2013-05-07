import sys
import os
import time

sys.path.append(os.environ["PYTHON_MODULE_PATH"])
sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

def getRoadRunnerTempDirectory():
    tempDirPath=os.path.abspath(os.path.join(os.environ["PREFIX_RR"],'temp'))
    
    if os.path.isdir(tempDirPath) :
        return tempDirPath
    else:
        tempDirPath=os.path.abspath(os.path.join(os.path.expanduser('~'),'.RoadRunnerTemp'))
        if os.path.isdir(tempDirPath):
            return tempDirPath
            
        try:
            os.makedirs(tempDirPath)
        except:
            print 'could not create road runner temporary directory : ',tempDirPath
            return None
        
        return tempDirPath
        
def getCompiler():
    if sys.platform.startswith('win'):
        return os.path.abspath(os.path.join(os.environ["PREFIX_RR"],'compilers/tcc/tcc.exe'))
    elif sys.platform.startswith('lin'):
        return os.path.abspath(os.path.join('/usr/bin','gcc'))
    else:
        return os.path.abspath(os.path.join('/usr/bin','gcc'))

global tempDirPath
global compilerSupportPath
global compilerExeFile        

tempDirPath=getRoadRunnerTempDirectory()

compilerSupportPath=os.path.abspath(os.path.join(os.environ["PREFIX_RR"],'rr_support'))

compilerExeFile=getCompiler()


        