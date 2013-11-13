import os,sys
import re

# example command:
# python .\win_cc3d_installer_creator.py -d 'D:\install_projects\3.7.0' -v 3.7.0.0

# this is the path to the NSIS instaler executable
NSIS_EXE_PATH='C:\Program Files (x86)\NSIS\makensis.exe '

# version has to have format 3.7.0.0 - four numbers otherwise NSIS crashes, strange...

# -------------- parsing command line
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-d", "--directory", dest="dirname",action="store", type="string",help="CC3D installation directory")
parser.add_option("-i", "--installer-name", dest="installer_name",action="store", default='', type="string",help="full installer name")
parser.add_option("-v", "--version", dest="version",action="store", type="string",help="CC3D version", default='3.7.0.0')

(options, args) = parser.parse_args()

# -------------- end of parsing command line

cc3d_install_dir=options.dirname
cc3d_install_dir=os.path.abspath(cc3d_install_dir)


cwd=os.path.abspath(os.getcwd())

#revision number 
from datetime import date
today=date.today()
revisionNumber=str(today.year)+str(today.month).zfill(2)+str(today.day).zfill(2)
version=options.version


INSTALLER_NAME=options.installer_name

if INSTALLER_NAME=='':
    INSTALLER_NAME=os.path.abspath(os.path.join(cwd,'setup-'+version+'v'+revisionNumber+'.exe'))
else:    
    INSTALLER_NAME=os.path.abspath(INSTALLER_NAME)
    
INSTALLATION_SOURCE_DIR=cc3d_install_dir


INSTALL_FILES=[] # holds lines for install section of the installer
DELETE_FILES=[]  # holds lines for delete section of the uninstaller

# ----------creating lines for install section of the installer
sub=''
for root, subfolders, files in os.walk(cc3d_install_dir):
    relpath=os.path.relpath(root,cc3d_install_dir )    
    separator='\\'
    if len(relpath)==1:
        INSTALL_FILES.append('SetOutPath '+'"$INSTDIR"')

    else:
        INSTALL_FILES.append('SetOutPath '+'"$INSTDIR'+separator+relpath+'"')
    
    
    for file in files:
        INSTALL_FILES.append('File '+'"'+os.path.join(root,file)+'"')

# ---------- end creating lines for install section of the installer    


# ----------creating lines for install section of the uninstaller

sub=''
dirs_to_remove=[]

for root, subfolders, files in os.walk(cc3d_install_dir):
    relpath=os.path.relpath(root,cc3d_install_dir )    
    
    if len(relpath)!=1:
        dirs_to_remove.insert(0,'"$INSTDIR\\'+relpath+'"')
    
    
    for file in files:
        relpath_file=os.path.relpath(os.path.join(root,file),cc3d_install_dir )   
        DELETE_FILES.append('Delete '+ '"$INSTDIR\\'+relpath_file+'"')        

DELETE_FILES.append('')
        

for dir in dirs_to_remove:
    DELETE_FILES.append('RmDir '+dir)

# ---------- end of creating lines for install section of the uninstaller





installer_path=os.path.dirname(INSTALLER_NAME)

inFile=open('CompuCell3D.nsi.tpl','r')
nsiFilePath=os.path.join(installer_path,'CompuCell3D_installer.nsi')
nsiFileDir=os.path.dirname(nsiFilePath)
if not os.path.exists(nsiFileDir):
    os.makedirs(nsiFileDir)
nsiFile=open(nsiFilePath,'w')
for line in inFile.readlines():
    line=line.rstrip()
    if line.startswith('!define VERSION'):
        line='!define VERSION '+'"'+version+'"'
    elif line.startswith('!define INSTALLER_NAME'):
        line='!define INSTALLER_NAME '+'"'+INSTALLER_NAME+'"'
    elif line.startswith('!define INSTALLATION_SOURCE_DIR'):
        line = '!define INSTALLATION_SOURCE_DIR '+'"'+INSTALLATION_SOURCE_DIR+'"'
    elif  line.startswith('<INSTALL_FILES>') :
        line=''
        for install_line in INSTALL_FILES:
            line+=install_line+'\n'
            
    elif  line.startswith('<DELETE_FILES>') :
        line=''
        for delete_line in DELETE_FILES:
            line+=delete_line+'\n'
            
    print>>nsiFile,line    
    

    
inFile.close()
nsiFile.close()

#executing NSIS command
import subprocess
subprocess.call([NSIS_EXE_PATH,'/V1',nsiFilePath])
# subprocess.call([NSIS_EXE_PATH,'/V1','CompuCell3D_installer.nsi'])

