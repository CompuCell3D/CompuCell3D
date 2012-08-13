#!/usr/bin/env python
#
# testXML.py - test .xml configuration files
#
#
# for cleaner output, might also want to comment out following lines in compucell3d.sh:
# echo "Configuration file: $xmlFile"
# echo "CompuCell3D - version $COMPUCELL3D_MAJOR_VERSION.$COMPUCELL3D_MINOR_VERSION.$COMPUCELL3D_BUILD_VERSION"


import sys
import string
import re
#import subprocess
import os, time


def processLine(line, output):
#      line = string.strip(line) + "\n"   # if want to strip blanks

      # for now, let's just do a simple replacement of Steps.
      if string.find(line, "<Steps>") > -1:
#        print 'found one: ' + line
#        line = re.sub('\>','',line)
#        print 'now line: ' + line
        output.write( '<Steps>10</Steps>\n')
      else:
        output.write(line)

def usage():
    msg = """Usage:\n  testXML.py\n
This script attempts to 
The converted code is printed on the standard output.

"""
    print msg

def main():
#    if len(sys.argv) < 2:
#        usage()
#        sys.exit(1)

    count=0
    outfile = 'tmp.xml'   # might want to do something different than copying to this config file
    for id in os.walk('Demos'):
      if count>0 and count<10:  # to test all, remove 'count<10' check
        print count,'-->',id
    #    print id[0],id[2]
        for f in id[2]:
    #      print f[-3:]
          if f[-3:] == 'xml':
    #        print 'yes, found xml:',f
            fullf = os.path.join(id[0],f)
            print fullf
            input = open(fullf,'r')
            output = open(outfile,'w')
            for line in input.readlines():
              processLine(line,output)
            input.close()
            output.close()
      count+=1


    # execute cc3d on this .xml config file
    cmd = "./compucell3d.sh -i " + outfile
    print cmd
#    subprocess.call(cmd)
    os.system(cmd)

    # assuming you've redirected cerr and cout (from the player's main.cpp), concat to a master output file
#    os.system("cat cc3d-out.txt >> cc3d-all.txt")
#    os.system("echo '-------------------------------' >> cc3d-all.txt")
    time.sleep(1)

if __name__ == "__main__":
    main()