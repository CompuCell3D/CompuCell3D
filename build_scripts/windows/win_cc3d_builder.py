# example command:
# python .\win_cc3d_builder.py  -p D:/install_projects/3.7.1 -s D:/CC3D_GIT  -i D:/CC3D_FILES_SVN/binaries/3.7.1/windows -v 3.7.1.0


import os,sys
import shutil
import re
from os.path import expanduser
import subprocess
import win32api

import time

 

def printRuntime(_timeInterval):
    timeInterval=int(_timeInterval)
    hours = timeInterval/(3600*1000)
    minutesInterval = timeInterval % (3600*1000)    
    minutes = minutesInterval / (60*1000)    
    secondsInterval = minutesInterval % (60*1000) 
    seconds = secondsInterval / (1000)     
    miliseconds = secondsInterval % (1000)
    
    print "Build RUNTIME ",
    if hours:
        print hours," h : ",minutes," m : ",seconds," s : ",miliseconds," ms"
    elif minutes: 
        print minutes," m : ",seconds," s : ",miliseconds," ms"
    elif seconds:
        print seconds," s : ",miliseconds," ms"
    else:
        miliseconds," ms"
    print "EQUIVALENT OF      %0.3f seconds" % (_timeInterval/1000)

 
def shellquote(s):
    return "'" + s.replace("'", "'\\''") + "'"

t1 = time.time()    
    

# this is the path to the NSIS instaler executable
NSIS_EXE_PATH='C:\Program Files (x86)\NSIS\makensis.exe '
CMAKE_PATH=os.path.abspath('C:/Program Files (x86)/CMake 2.8/bin/cmake.exe')

CMAKE_GENERATOR_NAME='NMake Makefiles'

# version has to have format 3.7.1.0 - four numbers otherwise NSIS crashes, strange...

# -------------- parsing command line
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-p", "--prefix", dest="prefix",action="store", type="string",default='D:/install_projects/3.7.1', help="CC3D installation directory")
parser.add_option("-s", "--source-root", dest="source_root",action="store", type="string",default='D:/CC3D_GIT', help="CC3D git repository")
parser.add_option("-c", "--cores", dest="cores",action="store", type="int",default=1,help="Number of compilation threads for make")
parser.add_option('-i' , '--installer-dir', dest='installer_dir', action='store',type='string',default='',help='Location of the place where to store installer')
parser.add_option("-v", "--version", dest="version",action="store", type="string", default='3.7.1.0',help='version of installer')

parser.add_option("--cc3d",  action="store_true", default=False, dest="cc3d",help='this option picks CompuCell3D to be compiled. Using it will require all other project files to be set individually')
parser.add_option("--bionet",  action="store_true", default=False, dest="bionet",help='this option picks BionetSolver to be compiled. Using it will require all other project files to be set individually')
parser.add_option("--celldraw",  action="store_true", default=False, dest="celldraw",help='this option picks BionetSolver to be compiled. Using it will require all other project files to be set individually')
parser.add_option("--rr",  action="store_true", default=False, dest="rr",help='this option picks RR to be compiled. Using it will require all other project files to be set individually')
# parser.add_option('--build-installer',action='store_true', default=False, dest="build_installer",help='this option tells the script to build windows installer')



(options, args) = parser.parse_args()
# -------------- end of parsing command line

CURRENT_DIR=os.getcwd()

BUILD_INSTALLER=False
version=options.version

INSTALLER_DIR=options.installer_dir
if INSTALLER_DIR!='':
    INSTALLER_DIR=os.path.abspath(INSTALLER_DIR)
    BUILD_INSTALLER=True

        
# BUILD_INSTALLER=options.build_installer
INSTALLER_VERSION=options.version


HOME_DIR=expanduser('~')

SOURCE_ROOT=os.path.abspath(options.source_root)
INSTALL_PREFIX=os.path.abspath(options.prefix)
BUILD_ROOT=os.path.abspath(INSTALL_PREFIX+'_build')
DEPENDENCIES_ROOT=os.path.abspath(INSTALL_PREFIX+'_depend')


# these are hard-coded for now
WIN_DEPENDENCIES_ROOT=os.path.abspath('D:/CC3D_FILES_SVN/dependencies/windows/VS2010/dependencies_qt_4.8.5_pyqt_4.10.3_vtk_5.10.1_python27')
LIBSBML_INSTALL_DIR=os.path.abspath('D:/CC3D_FILES_SVN/dependencies/windows/VS2010/BionetSolver/sbml-xml2') # used by bionet solver on windows
SUNDIALS_INSTALL_DIR=os.path.abspath('D:/CC3D_FILES_SVN/dependencies/windows/VS2010/BionetSolver/sundials') # used by bionet solver on windows
RR_BINARIES_DIR=os.path.abspath('D:/CC3D_FILES_SVN/dependencies/windows/VS2010/roadrunner/')

BUILD_CC3D=False
BUILD_BIONET=False
# BUILD_BIONET_DEPEND=False
BUILD_CELLDRAW=False
BUILD_RR=False
BUILD_RR_DEPEND=False
BUILD_ALL=True
MAKE_MULTICORE=1


if options.cc3d:
    BUILD_ALL=False
    BUILD_CC3D=True

if options.bionet:
    BUILD_ALL=False
    BUILD_BIONET=True

if options.celldraw:
    BUILD_ALL=False
    BUILD_CELLDRAW=True
    
if options.rr:
    BUILD_ALL=False
    BUILD_RR=True

if BUILD_ALL:
    BUILD_CC3D=True
    BUILD_BIONET=True
    # BUILD_BIONET_DEPEND=True
    BUILD_CELLDRAW=True
    BUILD_RR=True
    # # # BUILD_RR_DEPEND=True
    



if not os.path.isdir(BUILD_ROOT):
    os.makedirs(BUILD_ROOT)
    
if not os.path.isdir(DEPENDENCIES_ROOT):   
    os.makedirs(DEPENDENCIES_ROOT)

if BUILD_CC3D:    
    ############ building CompuCell3D
    

    CC3D_BUILD_PATH=os.path.abspath(os.path.join(BUILD_ROOT,'CompuCell3D'))
    CC3D_SOURCE_PATH=os.path.abspath(os.path.join(SOURCE_ROOT,'CompuCell3D'))
    if not os.path.isdir(CC3D_BUILD_PATH):   
        os.makedirs(CC3D_BUILD_PATH)
    os. chdir(CC3D_BUILD_PATH)  
    
    subprocess.call([CMAKE_PATH,'-G', CMAKE_GENERATOR_NAME,'-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo','-DCMAKE_INSTALL_PREFIX:PATH='+INSTALL_PREFIX,'-DWINDOWS_DEPENDENCIES:PATH='+WIN_DEPENDENCIES_ROOT, CC3D_SOURCE_PATH ])
    # subprocess.call(['nmake'])
    subprocess.call(['nmake','install'])
    ############ End of building CompuCell3D

    

if BUILD_BIONET:

    ############# BUILDING  BIONET 

    BIONET_SOURCE_PATH=os.path.abspath(os.path.join(SOURCE_ROOT,'BionetSolver/0.0.6'))
    BIONET_BUILD_PATH=os.path.abspath(os.path.join(BUILD_ROOT,'BionetSolver'))
        
    
    if not os.path.isdir(BIONET_BUILD_PATH):   
        os.makedirs(BIONET_BUILD_PATH)
    os. chdir(BIONET_BUILD_PATH)  
    
    subprocess.call([CMAKE_PATH,'-G', CMAKE_GENERATOR_NAME,'-DCMAKE_BUILD_TYPE:STRING=Release','-DCMAKE_INSTALL_PREFIX:PATH='+INSTALL_PREFIX,'-DLIBSBML_INSTALL_DIR:PATH='+LIBSBML_INSTALL_DIR, '-DSUNDIALS_INSTALL_DIR:PATH='+SUNDIALS_INSTALL_DIR, BIONET_SOURCE_PATH ])
    subprocess.call(['nmake','install'])    
    
  ############# END OF BUILDING  BIONET 

    
if  BUILD_CELLDRAW:

    ############# BUILDING  CELLDRAW 
    CELLDRAW_BUILD_PATH=os.path.abspath(os.path.join(BUILD_ROOT,'CellDraw'))
    CELLDRAW_SOURCE_PATH=os.path.abspath(os.path.join(SOURCE_ROOT,'CellDraw/1.5.1'))
    
    if not os.path.isdir(CELLDRAW_BUILD_PATH):   
        os.makedirs(CELLDRAW_BUILD_PATH)
    os. chdir(CELLDRAW_BUILD_PATH)  

    subprocess.call([CMAKE_PATH,'-G', CMAKE_GENERATOR_NAME,'-DCMAKE_BUILD_TYPE:STRING=Release','-DCMAKE_INSTALL_PREFIX:PATH='+INSTALL_PREFIX, CELLDRAW_SOURCE_PATH ])    
    subprocess.call(['nmake','install'])        

  ############# END OF  CELLDRAW 

if BUILD_RR:

    ############# BUILDING RR - actually in this case we are copying prebuilt binaries
    destinationDir=os.path.join(INSTALL_PREFIX,'lib/python/roadrunner')
    if os.path.exists(destinationDir):
        shutil.rmtree(destinationDir)
    shutil.copytree(RR_BINARIES_DIR,destinationDir)
  
if  BUILD_INSTALLER:    
    #revision number 
    from datetime import date
    today=date.today()
    revisionNumber=str(today.year)+str(today.month).zfill(2)+str(today.day).zfill(2)
    version=options.version
    
    INSTALLER_NAME=os.path.abspath(os.path.join(INSTALLER_DIR,'setup-'+version+'v'+revisionNumber+'.exe'))
      
    os.chdir(CURRENT_DIR)    
    subprocess.call(['python','win_cc3d_installer_creator.py','-d',INSTALL_PREFIX,'-v',INSTALLER_VERSION,'-i',INSTALLER_NAME])    
    
  
t2 = time.time()

printRuntime((t2-t1)*1000.0)