import sys
import os
import time

sys.path.append(os.environ["PYTHON_MODULE_PATH"])
sys.path.append(os.environ["SWIG_LIB_INSTALL_DIR"])

# will try to extract prefix environment variable - can be  either PREFIX_RR or PREFIX_CC3D - depending on what we install
global prefixPath
try:
    prefixPath=os.environ["PREFIX_RR"]
except LookupError,e:
    try:
        prefixPath=os.environ["PREFIX_CC3D"]
    except LookupError,e:
        prefixPath=''
# if prefixPath=='':
    # prefixPath=os.environ["PREFIX_CC3D"]


def getRoadRunnerTempDirectory():
    global prefixPath
    tempDirPath=os.path.abspath(os.path.join(prefixPath,'temp'))
    
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
        return os.path.abspath(os.path.join(prefixPath,'compilers/tcc/tcc.exe'))
    elif sys.platform.startswith('lin'):
        return os.path.abspath(os.path.join('/usr/bin','gcc'))
    else:
        return os.path.abspath(os.path.join('/usr/bin','gcc'))

global tempDirPath
global compilerSupportPath
global compilerExeFile        

tempDirPath=getRoadRunnerTempDirectory()

compilerSupportPath=os.path.abspath(os.path.join(prefixPath,'rr_support'))

compilerExeFile=getCompiler()


        