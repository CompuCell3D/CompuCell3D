import glob
import os

header_glob=glob.glob('*.h')
for header in header_glob:
    headerFullInclude='    #include <CompuCell3D/Boundary/'+header+'>'
    print headerFullInclude