import os
import sys

PREFIX_CC3D = sys.argv[1]
PYTHON_INSTALL_PATH = sys.argv[2]
os.chdir(PREFIX_CC3D)


def localize_script(name: str):
    # Modifying <name>.bat
    tmpFile = open("tmp.bat", "w")
    tmpFile.write("%s\n" % ("@ECHO OFF"))
    tmpFile.write("%s\n" % ("@SET PREFIX_CC3D=" + PREFIX_CC3D))
    tmpFile.write("%s\n" % ("@SET PYTHON_INSTALL_PATH=" + PYTHON_INSTALL_PATH))
    tmpFile.close()

    # concatenate
    outFile = open(f"{name}.bat", "w")

    # fileList to concatenate
    fileList = ["tmp.bat", f"{name}.bat.in.v2"]

    for x in fileList:
        print("Processing ", x)
        file = open(x, 'r')
        data = file.read()
        file.close()
        outFile.write(data)
        os.remove(x)

    outFile.close()


# Localize scripts
localize_script(name='compucell3d')
localize_script(name='playerParamScan')
