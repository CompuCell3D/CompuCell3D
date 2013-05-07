import sys
import os
import shutil
import rrPython

rr = rrPython

cwd = os.getcwd()

testFileName =  os.path.join(cwd, "..", "testing", "ModelSourceTest")
tempFolder   =  os.path.join(cwd, "..", "temp")

if not rr.setTempFolder(tempFolder) == True:
    print "Failed to set temp folder!!"
    exit()

#Copy test source to temp folder
shutil.copy(testFileName + ".h", tempFolder)
shutil.copy(testFileName + ".c", tempFolder)

sourceFileName = tempFolder + "/ModelSourceTest.c"
val = rr.compileSource(sourceFileName)

if val == True:
    print "It seems that the compilation succeeded. Check for a compiled libray in the temporary folder.."
else:
    print "There was a problem compiling a simple C program."

print "done"
