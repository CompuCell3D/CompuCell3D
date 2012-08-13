import sys
import string
import shutil
import os

# this is very simple and not really robust script to comment out all lines in swig-generated file
# containing "__del__ = lambda self : None"
# apparently any appearance of __del__ in the swig-generated class causes CC3D to segfault during object destruction when calling Py_DECREF(_cell->pyAttrib);
# this script fixes it but we have to look more into reference counting and garbage collection done in Python to make sure we are doing the right thing. For now it shuld work though

def main(argv):
    filename=argv[0]
    print "filename=",filename
    outputFileName=filename+".mod"
    outputFile=open(outputFileName,"w")
    for i,line in enumerate(open(filename)):
        if string.find(line,"#CC3D")!=-1:
            outputFile.write("%s"%(line))    
        elif string.find(line,"__del__ = lambda self : None")!=-1:
            outputFile.write("%s"%("#CC3D CUSTOM MODIFICATION "+line))    
            print "__del__ = lambda self : None in line ",i+1
        else:
            outputFile.write("%s"%(line))    
            
    outputFile.close()
    shutil.copy(outputFileName,filename)
    os.remove(outputFileName)
    
if __name__ == '__main__':
    main(sys.argv[1:])
    