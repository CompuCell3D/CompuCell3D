import os
import sys
import time


# PREFIX_CC3D=os.getcwd()
PREFIX_CC3D=sys.argv[1]
PYTHON_INSTALL_PATH=sys.argv[2]
os.chdir(PREFIX_CC3D)
# print "CWD=",os.getcwd()

# debugFile=open("debug.txt","w")
# debugFile.write("PREFIX_CC3D %s\n"%(PREFIX_CC3D))
# debugFile.write("PYTHON PATH %s\n"%(PYTHON_INSTALL_PATH))
# debugFile.close()

# print "PREFIX_CC3D=",PREFIX_CC3D
# Modifying run script for new Player
tmpFile=open("tmp.bat","w")
tmpFile.write("%s\n"%("@ECHO OFF"))
tmpFile.write("%s\n"%("@SET PREFIX_CC3D="+PREFIX_CC3D))
tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
tmpFile.close()

# #concatenate
# outFile=open("compucell3d_old.bat","w")

# #fileList to concatenate
# fileList=["tmp.bat","compucell3d_old.bat.in.v2"]

# # print "CWD=",os.getcwd()

# for x in fileList:
    # print "Processing ",x
    # file=open(x,'r')
    # data=file.read()
    # file.close()
    # outFile.write(data)
    # os.remove(x)

# outFile.close()


# Modifying compucell3d.bat run script 
tmpFile=open("tmp.bat","w")
tmpFile.write("%s\n"%("@ECHO OFF"))
tmpFile.write("%s\n"%("@SET PREFIX_CC3D="+PREFIX_CC3D))
tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
tmpFile.close()

#concatenate
outFile=open("compucell3d.bat","w")

#fileList to concatenate
fileList=["tmp.bat","compucell3d.bat.in.v2"]

# print "CWD=",os.getcwd()

for x in fileList:
    print "Processing ",x
    file=open(x,'r')
    data=file.read()
    file.close()
    outFile.write(data)
    os.remove(x)

outFile.close()


# Modifying runScript.bat 
tmpFile=open("tmp.bat","w")
tmpFile.write("%s\n"%("@ECHO OFF"))
tmpFile.write("%s\n"%("@SET PREFIX_CC3D="+PREFIX_CC3D))
tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
tmpFile.close()

#concatenate
outFile=open("runScript.bat","w")

#fileList to concatenate
fileList=["tmp.bat","runScript.bat.in.v2"]

# print "CWD=",os.getcwd()

for x in fileList:
    print "Processing ",x
    file=open(x,'r')
    data=file.read()
    file.close()
    outFile.write(data)
    os.remove(x)

outFile.close()


# Modifying paramScan.bat script 
tmpFile=open("tmp.bat","w")
tmpFile.write("%s\n"%("@ECHO OFF"))
tmpFile.write("%s\n"%("@SET PREFIX_CC3D="+PREFIX_CC3D))
tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
tmpFile.close()

#concatenate
outFile=open("paramScan.bat","w")

#fileList to concatenate
fileList=["tmp.bat","paramScan.bat.in.v2"]

# print "CWD=",os.getcwd()

for x in fileList:
    print "Processing ",x
    file=open(x,'r')
    data=file.read()
    file.close()
    outFile.write(data)
    os.remove(x)

outFile.close()


# Modifying twedit++ run script 
tmpFile=open("tmp.bat","w")
tmpFile.write("%s\n"%("@ECHO OFF"))
tmpFile.write("%s\n"%("@SET PREFIX_CC3D="+PREFIX_CC3D))
tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
tmpFile.close()

#concatenate
outFile=open("twedit++.bat","w")

#fileList to concatenate
fileList=["tmp.bat","twedit++.bat.in.v2"]

# print "CWD=",os.getcwd()

for x in fileList:
    print "Processing ",x
    file=open(x,'r')
    data=file.read()
    file.close()
    outFile.write(data)
    os.remove(x)

outFile.close()


# Modifying celldraw run script

import os.path
if os.path.isfile("celldraw.bat.in.v2"):

    # Modifying pifgenenrator file
    tmpFile=open("tmp.bat","w")
    tmpFile.write("%s\n"%("@ECHO OFF"))
    tmpFile.write("%s\n"%("@SET PREFIX_CELLDRAW="+PREFIX_CC3D))
    tmpFile.write("%s\n"%("@SET PYTHON_INSTALL_PATH="+PYTHON_INSTALL_PATH))
    tmpFile.close()

    #concatenate
    outFile=open("celldraw.bat","w")

    #fileList to concatenate
    fileList=["tmp.bat","celldraw.bat.in.v2"]

    # print "CWD=",os.getcwd()

    for x in fileList:
        print "Processing ",x
        file=open(x,'r')
        data=file.read()
        file.close()
        outFile.write(data)
        os.remove(x)

    outFile.close()





