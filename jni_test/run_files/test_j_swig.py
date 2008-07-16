from java.util import Random
r = Random()
r.nextInt()

for i in range(5):
   print r.nextDouble()

from java.lang import System
# System.loadLibrary("classes")


import java
import os
import sys

pathSeparator=java.lang.System.getProperty('path.separator')

oldPath2=os.environ['PATH']
newPath2="C:\Program Files\classes\CalculatorTry"+pathSeparator+oldPath2
os.environ['PATH']=newPath2

print "newPath2=",newPath2



oldPath=os.getenv("PATH")
print "path=",oldPath


# sys.exit()
#
# newPath="C:\Program Files\classes\CalculatorTry"+pathSeparator+oldPath

# os.setenv("PATH",newPath)




# oldPath1=java.lang.System.getProperty('java.library.path')
# print "oldPath1=",oldPath1
# sys.exit()

#
# pathSeparator=java.lang.System.getProperty('path.separator')
# newPath=oldPath+pathSeparator+"C:\Program Files\classes\CalculatorTry"
# java.lang.System.setProperty('java.library.path',newPath)
#
# print "newPath=",newPath
# print "java.library.path=",java.library.path


# sys.path.append("c:\Program Files\classes\CalculatorTry")
# print 'sys.path=',sys.path




from Geometry import Square

s=Square(10)

print dir(Square)

import CalculatorTry
packageProperties=dir(CalculatorTry)
print "packageProperties=",packageProperties

# classesModule=CalculatorTry.classes()
# print 'classesModule=',dir(classesModule)
System.loadLibrary("classes")
# loader=CalculatorTry.Loader()
# calculator=CalculatorPackage.classesJNI.new_Calculator()
calculator=CalculatorTry.Calculator()
calculator.calculate(9)
# try:
#    calculator=CalculatorTry.Calculator()
#
# except:
#    for val in sys.exc_info():
#       print val


# from CalculatorPackage import Calculator
# from CalculatorPackage import Loader
# l=Loader()
# print "This is dir(Loader)=",dir(Loader)
# # import CalculatorPackage
#


# System.loadLibrary("classes")
# print dir(Calculator)
# c=Calculator()

