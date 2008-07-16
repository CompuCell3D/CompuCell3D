from java.util import Random
r = Random()
r.nextInt()

for i in range(5):
   print r.nextDouble()

from java.lang import System

import java
import os
import sys

sys.path.append("C:\JavaProjects\Shapes\bin")


oldPath=java.lang.System.getProperty('python.path')
print "old python path=",oldPath
pathSeparator=java.lang.System.getProperty('path.separator')
newPath="C:\JavaProjects\Shapes\bin"
java.lang.System.setProperty('python.path',newPath)

print "python.home=",java.lang.System.getProperty('python.home')



import Geometry


from Geometry import Square
from Geometry import ShapeProcessor

s1=Square(10)
s2=Square(10.4)
s3=Square(7.2)

shapeProc=ShapeProcessor()
shapeProc.addShape(s1)
shapeProc.addShape(s2)
shapeProc.addShape(s3)
shapeProc.exploreShapes()

shapeProc.dictObj['test']=10
shapeProc.dictObj['test1']=10.10

print " number=",shapeProc.number
print " testShapeProcessor=",shapeProc.testShapeProcessor1()


# import java.lang

# string=java.lang.String("dupa")

# print "dir string ",dir(string)

# sys.exit()
import CalculatorTry
packageProperties=dir(CalculatorTry)
print "packageProperties=",packageProperties

System.loadLibrary("classes")
# System.loadLibrary("jvm")


calculator=CalculatorTry.CalculatorJava()

calculator.calculate(9)
calculator.calculate1(9)
calculator.callJavaFunction()

print "TRYING CALLING METHOD FROM EXTERNAL OBJECT"
calculator.setShapeObject(s1)
calculator.setShapeObject1(s2)
calculator.runShapeObjects()
print "TRYING CALLING METHOD FROM EXTERNAL OBJECT"
calculator.runShapeObject(s3)





jDict=calculator.pyDict
jDict1=calculator.pyDict

shapeWrapper=calculator.wrapper.shapeObjWrap;
# shapeWrapper=calculator.objectWrapperPtr.shapeObjWrap;
print "TESTING OBJ WRAPPER"
# shapeWrapper.printNumber()
print "TESTING OBJ WRAPPER"

print dir(shapeWrapper)
shapeProc.addShape(shapeWrapper)


shapeObj=calculator.getShapeObj()

print dir(shapeObj)

shapeObj=calculator.getShapeObj()
shapeObj1=calculator.getShapeObj1()


shapeProc.addShape(shapeObj)
shapeProc.addShape(shapeObj1)


shapeProc.exploreShapes()

sys.exit()

print "***************"
shapeObj.printNumber()

dictWrapper=Geometry.JDictWrapper()
dictWrapper['test']=20
dictWrapper['test1']=20.10

for key in dictWrapper.keys():
   print "dictWrapper[",key,"]=",dictWrapper[key]


for key in shapeProc.dictObj.keys():
   print "dictObj[",key,"]=",shapeProc.dictObj[key]

jDict1['a']=10
jDict['b']=10.20

for key in jDict.keys():
   print "jDict[",key,"]=",jDict[key]


jList=calculator.pyList
jList.append(10)
jList.append(10.20)

for i in jList:
   print "List element ",i



# print "METHODS OF SHAPE OBJ=",dir(shapeObj)
#
#
# print "type of shapeObj=", type(shapeObj)
#
# so.printNumber()
# print type(calculator)
# classObj.printNumber()
# calculator.shapeObj.printNumber()
